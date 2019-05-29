# Master Voice with WaveGAN (Python script)

### Initial parameters
1. Speaker IR directory: _/beegfs/kp2218/test_runs/conv_test/data/audio/ir_speaker/IR_ClestionBD300.wav_
2. Room IR directory: _/beegfs/kp2218/test_runs/conv_test/data/audio/ir_mic/IR_OktavaMD57.wav_
3. Mic IR directory: _/beegfs/kp2218/test_runs/conv_test/data/audio/ir_room/BRIR.wav_
4. Batch size: _16_
5. Number of iterations: _10_
6. Learning rate: _0.0001_
7. Min similarity factor: _0.25_
8. Max similairty factor: _0.75_
9. Utterance type: _male_
10. Post processing: _yes_

### Running the script
To run the script, execute the following code:
```
python3 Master_Voice_Main_Script.py
```

To run the script (without post-processing), execute the following code:
```
python3 Master_Voice_Main_Script.py --post_processing 'no'
```

To run the script (female utterances), execute the following code:
```
python3 Master_Voice_Main_Script.py --utterance_type 'female'
```

### Parameters
1. Speaker IR directory: _--speaker_ir_
2. Room IR directory: _--room_ir_
3. Mic IR directory: _--mic_ir_
4. Batch size: _--batch_
5. Number of iterations: _--iterations_
6. Learning rate: _--learning_rate_
7. Min similarity factor: _--min_similarity_
8. Max similairty factor: _--max_similarity_
9. Utterance type: _--utterance_type_
10. Post processing: _--post_processing_
