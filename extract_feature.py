import numpy as np
import pandas as pd
import timeit
from tqdm import tqdm
import librosa


def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(
        y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(
        y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)


def _extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result


def extract_features():
    def get_features(path, duration=2.5, offset=0.6):
        data, sr = librosa.load(path, duration=duration, offset=offset)
        aud = _extract_features(data)
        audio = np.array(aud)

        return audio

    start = timeit.default_timer()
    X, Y = [], []
    df = pd.read_csv('./output/df.csv')
    for path, emotion, index in tqdm(zip(df['Path'], df['Emotions'], range(df.shape[0]))):
        features = get_features(path)
        # for i in features:
        X.append(features)
        Y.append(emotion)
    df_ef = pd.DataFrame(X)
    df_ef['Emotions'] = Y
    df_ef.to_csv('./output/df_ef.csv', index=False)
    print(df_ef.head())
    print("Saved")


if __name__ == "__main__":
    extract_features()
