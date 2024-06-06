import numpy as np
import torch
from torch import nn
import torchaudio
from type import AudioHeadVariant, TextHeadVariant, ClassifierVariant
from OverKill_Model import OverKill_Model
from datasets import load_dataset
from torchinfo import summary
import torch.optim as optim
from AudioHead import AudioHead
from SERDataSet import SERDataset
from torch.utils.data import Dataset, DataLoader


def load_data():
    dataset = SERDataset("./input/ravdess-emotional-speech-audio")
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    # Create train and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    return train_loader, test_loader


def load_model():
    model = OverKill_Model(AudioHeadVariant.WHISPER,
                           TextHeadVariant.BERT_BASE_UNCASED, ClassifierVariant.MULI_HEAD_ATTENTION_NET)
    # model.print_trainable_params()
    return model


def train(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            waveform, sampling_rate, text, labels = data
            optimizer.zero_grad()
            outputs = predict(model, waveform, sampling_rate, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            break
        print(f'[{epoch + 1}] loss: {running_loss / len(train_loader):.3f}')
        running_loss = 0.0
    print('Finished Training')
    PATH = './output/saved_weights/overkill_model.pth'
    torch.save(model.state_dict(), PATH)


def predict(model, waveform, sampling_rate, text):
    op = model.predict(waveform, sampling_rate, text)
    return op


def evaluate(model, test_loader):
    PATH = './output/saved_weights/overkill_model.pth'
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            waveform, sampling_rate, text, labels = data
            outputs = predict(model, waveform, sampling_rate, text)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(labels, predicted)
            break
    print(f'Accuracy of the network : {100 * correct // total} %')
    print('Finished Evaluating')


def main():
    train_loader, test_loader = load_data()
    model = load_model()
    train(model, train_loader)
    evaluate(model,  test_loader)
    # for (waveform, sampling_rate, text) in train_loader:
    #     print(waveform.shape, sampling_rate.shape, len(text))
    # for (waveform, sampling_rate, text) in test_loader:
    #     print(waveform.shape, sampling_rate.shape, len(text))
    # waveform, sampling_rate, text, label = next(iter(train_loader))
    # op = predict(model, waveform, sampling_rate, text)
    # print(op)
    # print(op.shape)


if __name__ == "__main__":
    main()
