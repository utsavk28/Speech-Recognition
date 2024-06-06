from torch import nn
from type import ClassifierVariant
from FeedForwardNetwork import FeedForwardNetwork


class Classifier(nn.Module):
    def __init__(self, variant=None):
        super().__init__()
        self.network = None
        self.attention = None
        if variant in (ClassifierVariant.FEED_FORWARD_NET, ClassifierVariant.MULI_HEAD_ATTENTION_NET):
            self.network = FeedForwardNetwork()
        if variant is ClassifierVariant.MULI_HEAD_ATTENTION_NET:
            self.attention = nn.MultiheadAttention(1024, 8)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        if self.attention is not None:
            input, _ = self.attention(input, input, input)
        output = self.network.forward(input)
        # output = self.softmax(output)
        return output

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred

    def print_trainable_params(self):
        print(f"{'='*50} Classifier {'='*50}")
        total_params, trainable_params = self.network.print_trainable_params()
        if self.attention is not None:
            total_params = total_params + \
                sum(p.numel() for p in self.attention.parameters())
            trainable_params = trainable_params + sum(p.numel()
                                                      for p in self.attention.parameters() if p.requires_grad)
            print(f"{'='*50} Multi Head Self Attention {'='*50}")
            for name, param in self.attention.named_parameters():
                print(name, param.requires_grad)
        return total_params, trainable_params
