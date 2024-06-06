import torch
from torch import nn
from type import TextHeadVariant
from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration


class TextHead:
    def __init__(self, variant=None):
        self.tokenizer = None
        self.model = None
        self.pool = None
        self.variant = variant
        if variant is TextHeadVariant.BERT_BASE_UNCASED:
            self._load_bert_model()
        elif variant is TextHeadVariant.GOOGLE_T5_SMALL:
            self._load_t5_small()
        else:
            raise Exception("Invalid TextHeadVariant Type")
        self._freeze_model()

    def _load_t5_small(self) :
        model_name = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)


    def _load_bert_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.pool = nn.AdaptiveMaxPool1d(512)

    def _aggregate(self, input):
        return torch.mean(input, dim=1)
    
    def _freeze_model(self) :
        for param in self.model.parameters():
            param.requires_grad = False

    def run(self, input):
        encoded_input = self.tokenizer(input, return_tensors='pt')
        output = self.model(**encoded_input).last_hidden_state
        output = self._aggregate(output)
        if self.pool is not None:
            output = self.pool(output)
        return output

    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"{'='*50} {self.variant} {'='*50}")
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        return total_params, trainable_params
