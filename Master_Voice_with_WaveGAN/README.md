# Master Voice with WaveGAN (Python script)

_Note:_ The training loop for the Master Voice model is repeated 10 times for a given set of parameters. This loop is different from the _iterations_ parameter mentioned in the script.

### Default parameters
| Parameter               | Value |
| ----------------------- | ------------- |
| Speaker IR directory    |  _./audio/ir_speaker/IR_ClestionBD300.wav_  |
| Room IR directory       | _./audio/ir_mic/IR_OktavaMD57.wav_  |
| Mic IR directory        | _./audio/ir_room/BRIR.wav_  |
| Batch size              | _16_  |
| Number of iterations    | _1000_  |
| Learning Rate           | _1e-4_  |
| Min similarity factor   | _0.25_  |
| Max similarity factor   | _0.75_  |
| Utterance type          | _male_ |
| Post processing         | _False_  |
| Save directory          | _None_  |
| Total repetitions       | _10_  |

### Running the script
_Note:_ Here, the directory extension in which the training results are stored must be provided

To run the script, execute the following code:
```
python3 Master_Voice_Main_Script.py --save_dir './data'
```

To run the script (with post-processing), execute the following code:
```
python3 Master_Voice_Main_Script.py --save_dir './data' /
                                    --post_processing
```

To run the script (female utterances), execute the following code:
```
python3 Master_Voice_Main_Script.py --save_dir './data' /
                                    --utterance_type 'female'
```
### Running the tests
Here, 4 scripts have been given. Execute each script to get the desired results. The data corresponding to each script will be saved in a separate folder.

```
python3 Master_Voice_Main_Script.py --save_dir './male_yes-PP' /
                                    --post_processing

python3 Master_Voice_Main_Script.py --save_dir './male_no-PP'

python3 Master_Voice_Main_Script.py --save_dir './female_yes-PP' /
                                    --utterance_type 'female' /
                                    --post_processing

python3 Master_Voice_Main_Script.py --save_dir './female_no-PP' /
                                    --utterance_type 'female'
```

### Parameters Extensions
| Parameter               | Value |
| ----------------------- | ------------- |
| Speaker IR directory    |  _--speaker_ir_  |
| Room IR directory       | _--room_ir_ |
| Mic IR directory        | _--mic_ir_  |
| Batch size              | _--batch_  |
| Number of iterations    | _--iterations_  |
| Learning Rate           | _--learning_rate_  |
| Min similarity factor   | _--min_similarity_  |
| Max similarity factor   | _--max_similarity_  |
| Utterance type          | _--utterance_type_ |
| Post processing         | _--post_processing_  |
| Save directory          | _--save_dir_  |
| Total repetitions       | _--total_runs  |

### Model Architecture
// ![Alt text](https://github.com/mirkomarras/mastervoices/blob/master/Master_Voice_with_WaveGAN/model.png?raw=true "Model Summary")
