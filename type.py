from enum import Enum


class AudioHeadVariant(Enum):
    WAV2VEC2 = 'wav2vec'
    WAV2VEC2_BERT = 'wav2vec2_bert'
    WHISPER = 'whisper'


class TextHeadVariant(Enum):
    BERT_BASE_UNCASED = 'bert_base_uncased'
    GOOGLE_T5_SMALL = 'google_t5_small'

class ClassifierVariant(Enum):
    FEED_FORWARD_NET = 'feed_forward_net'
    MULI_HEAD_ATTENTION_NET = 'multi_head_attention_net'

