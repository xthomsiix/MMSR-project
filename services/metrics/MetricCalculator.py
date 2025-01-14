import logging
import random
from typing import Any, Dict, Hashable, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from services.common import IRMethod
from services.mmrs import MultiMediaRetrievalSystem
from services.data import DatasetLoader


class MetricCalculator:
    logger: logging.Logger = logging.getLogger(__name__)
    DISPLAY_TOL: int = 4
    DEBUG_NR_SAMPLES: int | None = 400

    def __init__(self, dataset_loader: DatasetLoader, mmrs: MultiMediaRetrievalSystem):
        self.dataset_loader: DatasetLoader = dataset_loader
        self.mmrs: MultiMediaRetrievalSystem = mmrs
        if not mmrs:
            self.mmrs.prepare_data(
                self.dataset_loader.id_artist_song_album,
                self.dataset_loader.id_url,
                self.dataset_loader.id_genres,
            )
        self.nr_samples: int = len(self.dataset_loader.id_artist_song_album)

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

    def compute_results(
        self,
        ir_method: IRMethod,
        song: str,
        artist: str,
        k: int,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None] | None:
        match ir_method:
            case IRMethod.BASELINE:
                results = self.mmrs.baseline(artist, song, k)
            case IRMethod.TFIDF:
                results = self.mmrs.tfidf(self.dataset_loader.tfidf, artist, song, k)
            case IRMethod.BERT:
                results = self.mmrs.bert(self.dataset_loader.bert, artist, song, k)
            case IRMethod.BLF_SPECTRAL:
                results = self.mmrs.blf_spectral(
                    self.dataset_loader.blf_spectral, artist, song, k
                )
            case IRMethod.MUSIC_NN:
                results = self.mmrs.music_nn(
                    self.dataset_loader.music_nn, artist, song, k
                )
            case IRMethod.RESNET:
                results = self.mmrs.resnet(self.dataset_loader.resnet, artist, song, k)
            case IRMethod.VGG19:
                results = self.mmrs.vgg19(self.dataset_loader.vgg19, artist, song, k)
            case _:
                self.logger.debug(f"IR Method '{ir_method}' not detected.")
                return None
        return results

    def get_sample(self, sample_size: int):
        assert (
            sample_size <= self.nr_samples
        ), "Sample size is larger than the number of samples"

        random_song_ids: List[int] = random.sample(range(self.nr_samples), sample_size)
        assert len(set(random_song_ids)) == len(random_song_ids)

        artists: List[str] = (
            self.dataset_loader.id_artist_song_album["artist"]
            .iloc[random_song_ids]
            .values.tolist()
        )  # type: ignore
        songs: List[str] = (
            self.dataset_loader.id_artist_song_album["song"]
            .iloc[random_song_ids]
            .values.tolist()
        )  # type: ignore

        return artists, songs

    def compute_cov_at_n(
        self,
        ir_method: IRMethod,
        sample_size: int,
        N: int,
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
        results_set = set()
        precision_at_k_list = list()
        recall_at_k_list = list()
        ndcg_at_k_list = list()
        mrr_at_k_list = list()
        artists, songs = self.get_sample(sample_size)

        for artist, song in tqdm(zip(artists, songs)):
            results = self.compute_results(ir_method, song, artist, N)

            if results is not None:
                for result in results.get("search_results"):  # type: ignore
                    results_set.add(result["id"])
                precision_at_k_list.append(results.get("precision"))
                recall_at_k_list.append(results.get("recall"))
                ndcg_at_k_list.append(results.get("ndcg"))
                mrr_at_k_list.append(results.get("mrr"))

        coverage = len(results_set) / sample_size
        return (
            np.round(coverage, tol),
            np.mean(precision_at_k_list).round(tol),
            np.mean(recall_at_k_list).round(tol),
            np.mean(ndcg_at_k_list).round(tol),
            np.mean(mrr_at_k_list).round(tol),
        )

    def compute_div_at_n(
        self,
        ir_method: IRMethod,
        sample_size: int,
        N: int,
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
        tag_set = set()
        precision_at_k_list = list()
        recall_at_k_list = list()
        ndcg_at_k_list = list()
        mrr_at_k_list = list()
        artists, songs = self.get_sample(sample_size)

        for artist, song in tqdm(zip(artists, songs)):
            results = self.compute_results(ir_method, song, artist, N)

            if results is not None:
                tag_set = self._update_tag_set(results.get("search_results"), tag_set)  # type: ignore
                precision_at_k_list.append(results.get("precision"))
                recall_at_k_list.append(results.get("recall"))
                ndcg_at_k_list.append(results.get("ndcg"))
                mrr_at_k_list.append(results.get("mrr"))

        diversity = len(tag_set) / sample_size
        return (
            np.round(diversity, tol),
            np.mean(precision_at_k_list).round(tol),
            np.mean(recall_at_k_list).round(tol),
            np.mean(ndcg_at_k_list).round(tol),
            np.mean(mrr_at_k_list).round(tol),
        )

    def _update_tag_set(self, results: List[Dict[str, Hashable | Any]], tag_set: Set):
        for result in results:
            genres = set(result["genre"])  # type: ignore
            tags = list(result["(tag, weight)"].keys())  # type: ignore
            tags_without_genre = [tag for tag in tags if tag not in genres]
            tag_set = tag_set.union(tags_without_genre)
        return tag_set

    def compute_avg_pop_at_n(
        self,
        ir_method: IRMethod,
        sample_size: int,
        N: int,
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
        total_popularity = 0.0
        precision_at_k_list = list()
        recall_at_k_list = list()
        ndcg_at_k_list = list()
        mrr_at_k_list = list()
        artists, songs = self.get_sample(sample_size)

        for artist, song in tqdm(zip(artists, songs)):
            results = self.compute_results(ir_method, song, artist, N)
            if results is not None:
                for result in results.get("search_results"):  # type: ignore
                    total_popularity += result["popularity"]  # type: ignore
                precision_at_k_list.append(results.get("precision"))
                recall_at_k_list.append(results.get("recall"))
                ndcg_at_k_list.append(results.get("ndcg"))
                mrr_at_k_list.append(results.get("mrr"))

        avg_pop = total_popularity / sample_size
        return (
            np.round(avg_pop, tol),
            np.mean(precision_at_k_list).round(tol),
            np.mean(recall_at_k_list).round(tol),
            np.mean(ndcg_at_k_list).round(tol),
            np.mean(mrr_at_k_list).round(tol),
        )

    def compute_optimized_div_at_n(
        self,
        ir_method: IRMethod,
        sample_size: int,
        N: int,
        tol: int = DISPLAY_TOL,
    ):
        cumulative_ndcg = 0.0
        artists, songs = self.get_sample(sample_size)
        pass
