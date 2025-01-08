import logging
from typing import Any, Dict, Hashable, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from services.mmrs import MultiMediaRetrievalSystem
from services.data import DatasetLoader


class MetricCalculator:
    logger: logging.Logger = logging.getLogger(__name__)
    DISPLAY_TOL: int = 4
    DEBUG_RUN_NR: int = 100

    def __init__(self, dataset_loader: DatasetLoader, mmrs: MultiMediaRetrievalSystem):
        self.dataset_loader: DatasetLoader = dataset_loader or DatasetLoader()
        self.mmrs: MultiMediaRetrievalSystem = mmrs or MultiMediaRetrievalSystem()
        if not mmrs:
            self.mmrs.prepare_data(
                self.dataset_loader.id_artist_song_album,
                self.dataset_loader.id_url,
                self.dataset_loader.id_genres,
            )

    # TODO: use varargs here and return the dataset
    def _prepare_data(
        self,
        *datasets: pd.DataFrame,
    ):
        """Prepare the data for the search process by merging datasets.

        Args:
            *data (pd.DataFrame): Some kind of data frame as vararg.
        """
        data = None
        for dataset in datasets:
            if data is None:
                data = dataset
            else:
                data = data.merge(dataset, on="id")
        return data

    def compute_cov_at_n(
        self,
        ir_method: str,
        k: int,
        tol: int = DISPLAY_TOL,
    ):
        """Computes the COV@N metric

        Args:
            ir_method (str): the IR method that should be evaluated.
            k (int): the number of items returned per query.
            tfidf: needed for using tfidf function.
            tol (int): the digits after zero to be displayed.

        Return:
            float: The COV@N score.
        """
        data = self._prepare_data(
            self.dataset_loader.id_artist_song_album,
            self.dataset_loader.id_url,
            self.dataset_loader.id_genres,
        )

        results_set = set()
        total_songs = self.DEBUG_RUN_NR  # len(data)

        for i in range(0, total_songs):
            artist = data["artist"].iloc[i]
            song = data["song"].iloc[i]

            match ir_method:
                case "Baseline":
                    results = self.mmrs.baseline(artist, song, k).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case "TF-IDF":
                    results = self.mmrs.tfidf(
                        self.dataset_loader.tfidf, artist, song, k
                    ).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case "BERT":
                    results = self.mmrs.bert(
                        self.dataset_loader.bert, artist, song, k
                    ).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case "BLF-Spectral":
                    results = self.mmrs.blf_spectral(
                        self.dataset_loader.blf_spectral, artist, song, k
                    ).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case "MusicNN":
                    results = self.mmrs.music_nn(
                        self.dataset_loader.music_nn, artist, song, k
                    ).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case "ResNet":
                    results = self.mmrs.resnet(
                        self.dataset_loader.resnet, artist, song, k
                    ).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case "VGG19":
                    results = self.mmrs.vgg19(
                        self.dataset_loader.vgg19, artist, song, k
                    ).get("search_results")
                    for result in results:
                        results_set.add(result["id"])
                case _:
                    self.logger.debug(f"IR Method '{ir_method}' not detected.")
                    return None

        coverage = len(results_set) / total_songs
        return np.round(coverage, tol)

    def compute_div_at_n(
        self,
        ir_method: str,
        k: int,
        tol: int = DISPLAY_TOL,
    ):
        """Computes the DIV@N metric

        Args:
            ir_method (str): the IR method that should be evaluated.
            k (int): the number of items returned per query.
            tfidf: needed for using tfidf function.
            tol (int): the digits after zero to be displayed.

        Return:
            float: The DIV@N score.
        """
        data = self._prepare_data(
            self.dataset_loader.id_artist_song_album,
            self.dataset_loader.id_genres,
            self.dataset_loader.id_tags,
        )
        total_songs = self.DEBUG_RUN_NR  # len(data)
        tag_set = set()

        for i in range(0, total_songs):
            artist = data["artist"].iloc[i]
            song = data["song"].iloc[i]

            match ir_method:
                case "Baseline":
                    results = self.mmrs.baseline(artist, song, k).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case "TF-IDF":
                    results = self.mmrs.tfidf(
                        self.dataset_loader.tfidf, artist, song, k
                    ).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case "BERT":
                    results = self.mmrs.bert(
                        self.dataset_loader.bert, artist, song, k
                    ).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case "BLF-Spectral":
                    results = self.mmrs.blf_spectral(
                        self.dataset_loader.blf_spectral, artist, song, k
                    ).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case "MusicNN":
                    results = self.mmrs.music_nn(
                        self.dataset_loader.music_nn, artist, song, k
                    ).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case "ResNet":
                    results = self.mmrs.resnet(
                        self.dataset_loader.resnet, artist, song, k
                    ).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case "VGG19":
                    results = self.mmrs.vgg19(
                        self.dataset_loader.vgg19, artist, song, k
                    ).get("search_results")
                    tag_set = self._update_tag_set(results, tag_set)
                case _:
                    self.logger.debug(f"IR Method '{ir_method}' not detected.")
                    return None

        # for id in results_set:
        #     entry: pd.DataFrame = data[data["id"] == id]
        #     genres = set(entry["genre"].tolist()[0])
        #     tags = entry["(tag, weight)"].tolist()[0].keys()
        #     tags_without_genre = [tag for tag in tags if tag not in genres]
        #     tag_set = tag_set.union(tags_without_genre)

        diversity = len(tag_set) / total_songs
        return np.round(diversity, tol)

    def _update_tag_set(self, results: List[Dict[str, Hashable | Any]], tag_set: Set):
        for result in results:
            genres = set(result["genre"])  # type: ignore
            tags = list(result["(tag, weight)"].keys())  # type: ignore
            tags_without_genre = [tag for tag in tags if tag not in genres]
            tag_set = tag_set.union(tags_without_genre)
        return tag_set

    def compute_avg_pop_at_n(
        self,
        ir_method: str,
        k: int,
        tol: int = DISPLAY_TOL,
    ):
        """Computes the AVG_POP@N metric

        Args:
            ir_method (str): the IR method that should be evaluated.
            k (int): the number of items returned per query.
            tfidf: needed for using tfidf function.
            tol (int): the digits after zero to be displayed.

        Return:
            float: The AVG_POP@N score = sum of popularity of result songs / number of result songs
        """
        data = self._prepare_data(
            self.dataset_loader.id_artist_song_album,
            self.dataset_loader.id_genres,
            self.dataset_loader.id_metadata,
        )
        total_songs = self.DEBUG_RUN_NR  # len(data)
        pop_list = list()

        for i in range(0, total_songs):
            artist = data["artist"].iloc[i]
            song = data["song"].iloc[i]

            match ir_method:
                case "Baseline":
                    results = self.mmrs.baseline(artist, song, k).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case "TF-IDF":
                    results = self.mmrs.tfidf(
                        self.dataset_loader.tfidf, artist, song, k
                    ).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case "BERT":
                    results = self.mmrs.bert(
                        self.dataset_loader.bert, artist, song, k
                    ).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case "BLF-Spectral":
                    results = self.mmrs.blf_spectral(
                        self.dataset_loader.blf_spectral, artist, song, k
                    ).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case "MusicNN":
                    results = self.mmrs.music_nn(
                        self.dataset_loader.music_nn, artist, song, k
                    ).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case "ResNet":
                    results = self.mmrs.resnet(
                        self.dataset_loader.resnet, artist, song, k
                    ).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case "VGG19":
                    results = self.mmrs.vgg19(
                        self.dataset_loader.vgg19, artist, song, k
                    ).get("search_results")
                    for result in results:
                        pop_list.append(result["popularity"])
                case _:
                    self.logger.debug(f"IR Method '{ir_method}' not detected.")
                    return None

        total_popularity = sum(pop_list)
        # for id in pop_list:
        #     entry: pd.DataFrame = data[data["id"] == id]
        #     popularity = entry["popularity"].tolist()[0]
        #     total_popularity += popularity

        diversity = total_popularity / len(pop_list)
        return np.round(diversity, tol)
