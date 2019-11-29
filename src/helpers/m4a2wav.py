import os
import argparse

from pydub import AudioSegment

formats_to_convert = ['.m4a']

counter = 0
for (dirpath, dirnames, filenames) in os.walk('/beegfs/mm10572/voxceleb2/dev'):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):
            filepath = os.path.join(dirpath, filename)
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                track = AudioSegment.from_file(filepath, file_extension_final)
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = os.path.join(dirpath, wav_filename)
                print(str(counter) + ' - Converting ' + str(filepath) + ' to ' + str(wav_path))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
            except:
                print("Error converting " + str(filepath))
        counter += 1