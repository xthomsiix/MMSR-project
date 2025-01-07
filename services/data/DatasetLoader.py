""""""

import logging
import os
from typing import Any, Callable, Dict, Literal
import numpy as np
import pandas as pd


class DatasetLoader:
    DATASET_PATH: str = "dataset"

    # avoid tedious typos since it enables autocomplete when calling the "load" method
    FILENAMES = Literal[
        "id_blf_correlation_mmsr.tsv",
        "id_blf_deltaspectral_mmsr.tsv",
        "id_blf_logfluc_mmsr.tsv",
        "id_blf_spectralcontrast_mmsr.tsv",
        "id_blf_spectral_mmsr.tsv",
        "id_blf_vardeltaspectral_mmsr.tsv",
        "id_genres_mmsr.tsv",
        "id_incp_mmsr.tsv",
        "id_information_mmsr.tsv",
        "id_ivec1024_mmsr.tsv",
        "id_ivec256_mmsr.tsv",
        "id_ivec512_mmsr.tsv",
        "id_lyrics_bert_mmsr.tsv",
        "id_lyrics_tf-idf_mmsr.tsv",
        "id_lyrics_word2vec_mmsr.tsv",
        "id_metadata_mmsr.tsv",
        "id_mfcc_bow_mmsr.tsv",
        "id_mfcc_stats_mmsr.tsv",
        "id_musicnn_mmsr.tsv",
        "id_resnet_mmsr.tsv",
        "id_tags_dict.tsv",
        "id_total_listens.tsv",
        "id_url_mmsr.tsv",
        "id_vgg19_mmsr.tsv",
    ]

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, path: str | None = None):
        self.path: str = path or self.DATASET_PATH

        # load all necessary files
        self.id_artist_song_album: pd.DataFrame = self._load("id_information_mmsr.tsv")
        self.id_url: pd.DataFrame = self._load("id_url_mmsr.tsv")
        self.id_genres: pd.DataFrame = self._load("id_genres_mmsr.tsv", {"genre": eval})
        self.id_tags: pd.DataFrame = self._load("id_tags_dict.tsv", {"(tag, weight)": eval})
        self.id_metadata: pd.DataFrame = self._load("id_metadata_mmsr.tsv", {"popularity": eval})
        self.tfidf: np.ndarray[Any, np.dtype[np.float64]] = self._convert_to_numpy(
            self._load("id_lyrics_tf-idf_mmsr.tsv")
        )
        self.bert: np.ndarray[Any, np.dtype[np.float64]] = self._convert_to_numpy(
            self._load("id_lyrics_bert_mmsr.tsv")
        )
        self.blf_spectral: np.ndarray[Any, np.dtype[np.float64]] = self._convert_to_numpy(
            self._load("id_blf_spectral_mmsr.tsv")
        )
        self.music_nn: np.ndarray[Any, np.dtype[np.float64]] = self._convert_to_numpy(
            self._load("id_musicnn_mmsr.tsv")
        )
        self.resnet: np.ndarray[Any, np.dtype[np.float64]] = self._convert_to_numpy(
            self._load("id_resnet_mmsr.tsv")
        )
        self.vgg19: np.ndarray[Any, np.dtype[np.float64]] = self._convert_to_numpy(
            self._load("id_vgg19_mmsr.tsv")
        )

    def _load(
        self, filename: FILENAMES, converters: Dict[str, Callable[[Any], Any]] = {}
    ) -> pd.DataFrame:
        """Load a file from the dataset folder.

        Args:
            filename (FILENAMES): The filename of the file to load.
            converters (Dict[str, Callable[[Any], Any]], optional):
                The converters to use for the columns. Defaults to {}.

        Returns:
            pd.DataFrame: The loaded file as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        file_path: str = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"{file_path} not found")
            raise FileNotFoundError(f"{file_path} not found")

        data: pd.DataFrame = pd.read_csv(  # type: ignore
            filepath_or_buffer=file_path, sep="\t", converters=converters
        )
        self.logger.debug(f"Loaded {filename} from {file_path}")

        return data

    def _convert_to_numpy(
        self, data: pd.DataFrame
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Convert the data to a numpy array.

        Args:
            data (pd.DataFrame): The data to convert.

        Returns:
            np.ndarray: The data as a numpy array.
        """
        # remove all non-numeric columns
        data = data.select_dtypes(include=[np.number])
        return data.to_numpy()  # type: ignore
