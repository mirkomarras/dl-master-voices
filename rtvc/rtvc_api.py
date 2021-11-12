from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
from audioread.exceptions import NoBackendError

# encoder = None
synthesizer = None
# vocoder = None

def load_models(prefix):
    # global _voc_model_fpath
    # global _enc_model_fpath
    # global _syn_model_fpath
    global synthesizer

    _voc_model_fpath = Path(os.path.join(prefix, 'vocoder/saved_models/pretrained/pretrained.pt'))
    _enc_model_fpath = Path(os.path.join(prefix, 'encoder/saved_models/pretrained.pt'))
    _syn_model_fpath = Path(os.path.join(prefix, 'synthesizer/saved_models/pretrained/pretrained.pt'))

    torch.manual_seed(1234)

    encoder.load_model(_enc_model_fpath)
    synthesizer = Synthesizer(_syn_model_fpath, verbose=False)
    vocoder.load_model(_voc_model_fpath, verbose=False)


def get_embedding(speaker):
    if isinstance(speaker, str):

        # The following two methods are equivalent:
        # - Directly load from the filepath:
        # preprocessed_wav = encoder.preprocess_wav(speaker)
        
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(str(speaker), sr=16000)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)

    elif len(speaker) > 1024:
        preprocessed_wav = encoder.preprocess_wav(speaker, 16000)    
        embed = encoder.embed_utterance(preprocessed_wav)
    else:
        embed = torch.from_numpy(speaker)
    
    if hasattr(embed, 'numpy'):
        return embed.numpy()
    else:
        return embed


def vc(text, speaker, clip_length=200):

    embed = get_embedding(speaker)

    # If seed is specified, reset torch seed and force synthesizer reload
    # torch.manual_seed(1234)
    # synthesizer = Synthesizer(_syn_model)

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
                
    ## Generating the waveform
    # If seed is specified, reset torch seed and reload vocoder
    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     vocoder.load_model(args.voc_model_fpath)

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    # print(spec.shape)

    # Clip very long audio
    if clip_length is not None and spec.shape[1] > clip_length:
        spec = spec[:, :clip_length]

    generated_wav = vocoder.infer_waveform(spec)
    
    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    return generated_wav
