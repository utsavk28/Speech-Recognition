import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM, BatchNormalization, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


def loading_data():
    df = pd.read_csv('./output/df_ef.csv')
    return df


def handling_data(df):
    def handling_na_values(df):
        return df.fillna(0)

    def encoding(Y):
        encoder = OneHotEncoder()
        return encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray(), encoder

    df = handling_na_values(df)
    X = df.iloc[:, :-1].values
    Y = df['Emotions'].values
    Y, encoder = encoding(Y)
    return X, Y, encoder


def split_train_test_data(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=42, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


def reshaping_for_lstm(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


def main():
    df = loading_data()
    X, Y, encoder = handling_data(df)
    x_train, x_test, y_train, y_test = split_train_test_data(X, Y)
    x_train = reshaping_for_lstm(x_train)
    x_test = reshaping_for_lstm(x_test)
    print(x_train.shape, x_test.shape)

    early_stop = EarlyStopping(
        monitor='val_acc', mode='auto', patience=5, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(
        monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    model01 = Sequential()
    model01.add(LSTM(128, return_sequences=True,
                input_shape=(x_train.shape[1], 1)))
    model01.add(Dropout(0.2))
    model01.add(LSTM(128, return_sequences=True))
    # model01.add(Dropout(0.2))
    model01.add(LSTM(128, return_sequences=True))
    # model01.add(Dropout(0.2))
    model01.add(LSTM(128, return_sequences=True))
    # model01.add(Dropout(0.2))
    model01.add(LSTM(128, return_sequences=True))
    # model01.add(Dropout(0.2))
    model01.add(LSTM(128, return_sequences=True))
    # model01.add(Dropout(0.3))
    model01.add(LSTM(128))
    # model01.add(Dropout(0.3))
    model01.add(Dense(7, activation='softmax'))
    model01.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    model01.summary()

    hist = model01.fit(x_train, y_train,
                       epochs=1,
                       validation_data=(x_test, y_test), batch_size=64,
                       verbose=1)

    print("Accuracy of our model on test data : ",
          model01.evaluate(x_test, y_test)[1]*100, "%")
    epochs = [i for i in range(20)]
    fig, ax = plt.subplots(1, 2)
    train_acc = hist.history['accuracy']
    train_loss = hist.history['loss']
    test_acc = hist.history['val_accuracy']
    test_loss = hist.history['val_loss']
    
    fig.set_size_inches(20, 6)
    ax[0].plot(epochs, train_loss, label='Training Loss')
    ax[0].plot(epochs, test_loss, label='Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs, train_acc, label='Training Accuracy')
    ax[1].plot(epochs, test_acc, label='Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.show()



if __name__ == "__main__":
    main()
