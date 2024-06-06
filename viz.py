import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns


def viz():
    df = pd.read_csv('./output/df.csv')
    data, sr = get_sample_data(df)
    print(f"Sampling Rate: {sr}")
    viz_emotion_count(df)
    viz_mel_spectrogram(data, sr)
    viz_mfcc(data, sr)


def get_sample_data(df):
    data, sr = librosa.load(df.loc[0, 'Path'])
    return data, sr


def viz_emotion_count(df):
    plt.title('Count of Emotions', size=16)
    sns.countplot(df['Emotions'])
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


def viz_mel_spectrogram(data, sr):
    plt.figure(figsize=(10, 5))
    spectrogram = librosa.feature.melspectrogram(
        y=data, sr=sr, n_mels=128, fmax=8000)
    log_spectrogram = librosa.power_to_db(spectrogram)
    librosa.display.specshow(
        log_spectrogram, y_axis='mel', sr=sr, x_axis='time')
    plt.title('Mel Spectrogram ')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def viz_mfcc(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
    plt.figure(figsize=(16, 10))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    viz()
