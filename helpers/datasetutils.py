import os

def getData(source):
    paths = []
    labels = []
    counter = 0
    max_lenght = len(list(os.listdir(source)))
    print('Start loading data from', source)
    for index_user_folder, user_folder in enumerate(os.listdir(source)):
        print('\rLoading utterances from user', counter+1, '/', max_lenght, end='')
        user_folder_path = os.path.join(source, user_folder)
        for index_video_folder, video_folder in enumerate(os.listdir(user_folder_path)):
            video_folder_path = os.path.join(source, user_folder, video_folder)
            for index_audio_folder, audio_folder in enumerate(os.listdir(video_folder_path)):
                paths.append(os.path.join(source, user_folder, video_folder, audio_folder))
                labels.append(counter)
        counter += 1
    print('')
    print('Found', len(paths), 'utterances from', source)
    return {"paths": paths, "labels": labels}