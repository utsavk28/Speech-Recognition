import os
import pandas as pd


def preprocess():
    data_path = "./input/ravdess-emotional-speech-audio"
    emotions, input_paths = extract_emotions_paths(data_path)
    df = get_df(emotions, input_paths)
    save_as_csv(df)


def extract_emotions_paths(data_path):
    dir_list = os.listdir(data_path)
    emotions = []
    input_paths = []
    for i in dir_list:
        actor_path_dir = data_path + '/' + i
        actor = os.listdir(actor_path_dir)
        for f in actor:
            part = f.split('.')[0].split('-')
            emotions.append(int(part[2]))
            input_paths.append(actor_path_dir + '/' + f)

    return emotions, input_paths


def get_df(emotions, input_paths):
    idx_2_emotion = {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust',
                     8: 'surprise'}
    emotions = list(map(lambda x: idx_2_emotion[x], emotions))
    df = pd.DataFrame(
        zip(emotions, input_paths), columns=['Emotions', 'Path'])
    return df


def save_as_csv(df):
    df.to_csv("./output/df.csv", index=False)


if __name__ == "__main__":
    preprocess()
