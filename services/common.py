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
    LLM = "LLM"
