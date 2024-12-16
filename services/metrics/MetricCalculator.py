import logging
from typing import Any

import numpy as np
import pandas as pd

from services.mmrs import MultiMediaRetrievalSystem
from services.data import DatasetLoader


class MetricCalculator:
    logger: logging.Logger = logging.getLogger(__name__)
    DISPLAY_TOL: int = 4

    def __init__(self):
        self.dataset_loader: DatasetLoader = DatasetLoader()
        self.mmrs: MultiMediaRetrievalSystem = MultiMediaRetrievalSystem()
        self.mmrs.prepare_data(
            self.dataset_loader.id_artist_song_album, self.dataset_loader.id_url, self.dataset_loader.id_genres
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
            tfidf: np.ndarray[Any, np.dtype[np.float64]] = None,
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
            self.dataset_loader.id_genres)

        total_songs = len(data)
        results_set = set()

        for i in range(0, total_songs):
            artist = data["artist"].iloc[i]
            song = data["song"].iloc[i]

            match ir_method:
                case "Baseline":
                    results = self.mmrs.baseline(artist, song, k).get("search_results")
                    for result in results:
                        results_set.add(result['id'])
                case "TF-IDF":
                    results = self.mmrs.tfidf(tfidf, artist, song, k).get("search_results")
                    for result in results:
                        results_set.add(result['id'])
                case _:
                    self.logger.debug(f"IR Method '{ir_method}' not detected.")
                    return None

        coverage = len(results_set) / total_songs
        return np.round(coverage, tol)

    def compute_div_at_n(
            self,
            ir_method: str,
            k: int,
            tfidf: np.ndarray[Any, np.dtype[np.float64]] = None,
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
            self.dataset_loader.id_tags)
        total_songs = len(data)
        results_set = set()

        for i in range(0, total_songs):
            artist = data["artist"].iloc[i]
            song = data["song"].iloc[i]

            match ir_method:
                case "Baseline":
                    results = self.mmrs.baseline(artist, song, k).get("search_results")
                    for result in results:
                        results_set.add(result['id'])
                case "TF-IDF":
                    results = self.mmrs.tfidf(tfidf, artist, song, k).get("search_results")
                    for result in results:
                        results_set.add(result['id'])
                case _:
                    self.logger.debug(f"IR Method '{ir_method}' not detected.")
                    return None

        tag_set = set()
        for id in results_set:
            entry: pd.DataFrame = data[data["id"] == id]
            genres = set(entry["genre"].tolist()[0])
            tags = entry["(tag, weight)"].tolist()[0].keys()
            tags_without_genre = [tag for tag in tags if tag not in genres]
            tag_set = tag_set.union(tags_without_genre)

        diversity = len(tag_set) / total_songs
        return np.round(diversity, tol)

    def compute_avg_pop_at_n(
            self,
            ir_method: str,
            k: int,
            tfidf: np.ndarray[Any, np.dtype[np.float64]] = None,
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
            self.dataset_loader.id_metadata)
        total_songs = len(data)
        results_list = list()

        for i in range(0, total_songs):
            artist = data["artist"].iloc[i]
            song = data["song"].iloc[i]

            match ir_method:
                case "Baseline":
                    results = self.mmrs.baseline(artist, song, k).get("search_results")
                    for result in results:
                        results_list.append(result['id'])
                case "TF-IDF":
                    results = self.mmrs.tfidf(tfidf, artist, song, k).get("search_results")
                    for result in results:
                        results_list.append(result['id'])
                case _:
                    self.logger.debug(f"IR Method '{ir_method}' not detected.")
                    return None

        total_popularity = 0
        for id in results_list:
            entry: pd.DataFrame = data[data["id"] == id]
            popularity = entry["popularity"].tolist()[0]
            total_popularity += popularity

        diversity = total_popularity / len(results_list)
        return np.round(diversity, tol)
