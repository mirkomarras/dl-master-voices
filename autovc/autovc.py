import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
import librosa
from librosa.filters import mel
from numpy.random import RandomState

import torch
import numpy as np
from math import ceil
from model_vc import Generator

from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

from rtvc.synthesizer import audio as synth_a

from wavenet import build_model as build_wavenet
from wavenet import wavegen

from helpers import audio

_model_speaker_embedding = None
_model_vc = None
_model_vocoder = None
_target_length = 2.58

DEVICE = 'cuda:0'


def decode_audio(fp, sample_rate=16000):
    audio_sf, new_sample_rate = librosa.load(fp, sr=sample_rate, mono=True)
    return audio_sf


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
    
def mel_spectrogram(y):
    
    prng = RandomState(0) 
    
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)
    
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)
    return S.astype(np.float32)


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def make_autovc():
    
    G = Generator(32,256,512,32).eval().to(DEVICE)

    g_checkpoint = torch.load('models/autovc.ckpt', map_location=DEVICE)
    G.load_state_dict(g_checkpoint['model'])
    return G

def make_dvector():
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load('models/3000000-BL.ckpt')
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    return C

def make_wavenet():
    wavenet = build_wavenet().to(DEVICE)
    checkpoint = torch.load("models/checkpoint_step001000000_ema.pth")
    wavenet.load_state_dict(checkpoint["state_dict"])
    return wavenet


def get_speakerembedding(target_speaker):
    
    global _model_speaker_embedding

    if _model_speaker_embedding is None:
        _model_speaker_embedding = make_dvector()
    
    # Target passed as filename
    if isinstance(target_speaker, str):
        target_speaker = audio.decode_audio(target_speaker, target_length=_target_length)
        target_speaker = mel_spectrogram(target_speaker)
        emb_tgt = _model_speaker_embedding(torch.from_numpy(target_speaker[np.newaxis, :, :]).cuda())
    
    # Target passed as speaker embedding
    elif target_speaker.ndim == 1 and len(target_speaker) == 256:
        emb_tgt = torch.from_numpy(target_speaker).cuda()
        
    # Target passed as waveform
    elif target_speaker.ndim == 1 and len(target_speaker) > 256:
        target_speaker = mel_spectrogram(target_speaker)
        emb_tgt = _model_speaker_embedding(torch.from_numpy(target_speaker[np.newaxis, :, :]).cuda())
        
    return emb_tgt


def vc(waveform, target_speaker, vocoder='gl'):
    
    global _model_speaker_embedding
    global _model_vc
    global _model_vocoder
    
    if _model_speaker_embedding is None:
        _model_speaker_embedding = make_dvector()
    
    if _model_vc is None:
        _model_vc = make_autovc()
            
    if isinstance(waveform, str):    
        waveform = audio.decode_audio(waveform, target_length=_target_length)
        
    spectrum = mel_spectrogram(waveform)
    emb_src = _model_speaker_embedding(torch.from_numpy(spectrum[np.newaxis, :, :]).cuda())
    
    emb_tgt = get_speakerembedding(target_speaker)
    
    # Run voice conversion
    spectrum_gpu = torch.from_numpy(pad_seq(spectrum)[0][np.newaxis, :, :]).cuda()

    with torch.no_grad():
        _, spectrum_vc, _ = _model_vc(spectrum_gpu, emb_src, emb_tgt)

    # Setup vocoders
    if vocoder in {'gl', 'wavernn'}:
        from rtvc.synthesizer.hparams import hparams
        hparams.hop_size = 256
        hparams.win_size = 800
        hparams.n_fft = 1024
        hparams.ref_level_db = 16
        hparams.fmin = 90
        hparams.fmax = 7600
        hparams.griffin_lim_iters = 10

        def renormalize(s, hparams):
            return synth_a._normalize(100 * (s - 1) - 16, hparams)
        
        sp = spectrum_vc.cpu().numpy()[0, 0, ...].T
        sp = renormalize(sp, hparams)

    if vocoder == 'gl':        
        waveform_rec = synth_a.inv_mel_spectrogram(sp, hparams)
    
    elif vocoder == 'wavernn':
        
        if _model_vocoder is None:
            from rtvc.vocoder import inference as _model_vocoder
            _model_vocoder.load_model('../rtvc/vocoder/saved_models/pretrained/pretrained.pt')
                    
        waveform_rec = _model_vocoder.infer_waveform(sp) # progress_callback=lambda *args: None
    
    elif vocoder == 'wavenet':
        
        if _model_vocoder is None:
            _model_vocoder = make_wavenet()
                
        waveform_rec = wavegen(_model_vocoder, c=spectrum_vc[0, 0].cpu())
    
    else:
        raise ValueError(f'Invalid vocoder: {vocoder}')
        
    return waveform_rec


def reset():
    global _model_speaker_embedding
    global _model_vc
    global _model_vocoder

    _model_speaker_embedding = None
    _model_vc = None
    _model_vocoder = None