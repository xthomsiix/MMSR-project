# define constants for configuration and rendering
from enum import Enum


class IRMethod(Enum):
    BASELINE = "Baseline"
    TFIDF = "TF-IDF"
    BERT = "BERT"
    BLF_SPECTRAL = "BLF-Spectral"
    MUSIC_NN = "MusicNN"
    RESNET = "ResNet"
    VGG19 = "VGG19"
    BERT_EMBEDDINGS = "BERT-Embeddings"
    EARLY_FUSION = "Early-Fusion"
    LATE_FUSION = "Late-Fusion"
    LLM = "LLM"
