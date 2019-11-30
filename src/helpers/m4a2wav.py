import os
import argparse
import time

from pydub import AudioSegment

formats_to_convert = ['.m4a']

total = 1092009
counter = 0
for (dirpath, dirnames, filenames) in os.walk('/beegfs/mm10572/voxceleb2/dev'):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):
            filepath = os.path.join(dirpath, filename)
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                t1 = time.time()
                track = AudioSegment.from_file(filepath, file_extension_final)
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = os.path.join(dirpath, wav_filename)
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
                t2 = time.time()
                print(str(counter)  + ' - ' + str((t2-t1)*(total-counter)//60) +  'm - Converting ' + str(filepath) + ' to ' + str(wav_path))
            except:
                print("Error converting " + str(filepath))
        counter += 1