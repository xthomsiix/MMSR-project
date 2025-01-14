import logging
from typing import Any, Dict, Hashable, List, Set

import numpy as np
import pandas as pd

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

    def compute_cov_at_n(
        self,
        ir_method: IRMethod,
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
        results_set = set()
        precision_at_k_list = list()
        recall_at_k_list = list()
        ndcg_at_k_list = list()
        mrr_at_k_list = list()
        total_songs = self.DEBUG_NR_SAMPLES or self.nr_samples

        for i in range(0, total_songs):
            artist = self.dataset_loader.id_artist_song_album["artist"].iloc[i]
            song = self.dataset_loader.id_artist_song_album["song"].iloc[i]

            if ir_method == IRMethod.EARLY_FUSION:
                results = self.compute_early_fusion_results(self.dataset_loader.tfidf, self.dataset_loader.bert, song, artist, k)
            elif ir_method == IRMethod.LATE_FUSION:
                tfidf_results = self.compute_results(IRMethod.TFIDF, song, artist, k)
                bert_results = self.compute_results(IRMethod.BERT, song, artist, k)
                if tfidf_results is not None and bert_results is not None:
                    results = self.compute_late_fusion_results(tfidf_results, bert_results, k)
            else:
                results = self.compute_results(ir_method, song, artist, k)

            if results is not None:
                for result in results.get("search_results"):  # type: ignore
                    results_set.add(result["id"])
                precision_at_k_list.append(results.get("precision"))
                recall_at_k_list.append(results.get("recall"))
                ndcg_at_k_list.append(results.get("ndcg"))
                mrr_at_k_list.append(results.get("mrr"))

        coverage = len(results_set) / total_songs
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
        total_songs = self.DEBUG_NR_SAMPLES or self.nr_samples
        tag_set = set()
        precision_at_k_list = list()
        recall_at_k_list = list()
        ndcg_at_k_list = list()
        mrr_at_k_list = list()

        for i in range(0, total_songs):
            artist = self.dataset_loader.id_artist_song_album["artist"].iloc[i]
            song = self.dataset_loader.id_artist_song_album["song"].iloc[i]

            results = self.compute_results(ir_method, song, artist, k)
            if results is not None:
                tag_set = self._update_tag_set(results.get("search_results"), tag_set)  # type: ignore
                precision_at_k_list.append(results.get("precision"))
                recall_at_k_list.append(results.get("recall"))
                ndcg_at_k_list.append(results.get("ndcg"))
                mrr_at_k_list.append(results.get("mrr"))

        diversity = len(tag_set) / total_songs
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

    def compute_early_fusion_results(self,tfidf: np.ndarray[Any, np.dtype[np.float64]],bert: np.ndarray[Any, np.dtype[np.float64]],song: str,
    artist: str,k: int,) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None] | None:
        results = self.mmrs.early_fusion(tfidf, bert, artist, song, k)
        return results

    def compute_late_fusion_results(self,
    tfidf_results: Dict[str, Any],
    bert_results: Dict[str, Any],
    k: int,
) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None] | None:
        results = self.mmrs.late_fusion(tfidf_results, bert_results, k)
        return results
    def compute_all_metrics(
        self,
        ir_method: IRMethod,
        song: str,
        artist: str,
        k: int,
        tol: int = DISPLAY_TOL,
    ):
        if ir_method == IRMethod.EARLY_FUSION:
            results = self.compute_early_fusion_results(self.dataset_loader.tfidf, self.dataset_loader.bert, song, artist, k)
        elif ir_method == IRMethod.LATE_FUSION:
            tfidf_results = self.compute_results(IRMethod.TFIDF, song, artist, k)
            bert_results = self.compute_results(IRMethod.BERT, song, artist, k)
            if tfidf_results is not None and bert_results is not None:
                results = self.compute_late_fusion_results(tfidf_results, bert_results, k)
            else:
                results = None
        else:
            results = self.compute_results(ir_method, song, artist, k)

        if results is not None:
            precision = results.get("precision")
            recall = results.get("recall")
            ndcg = results.get("ndcg")
            mrr = results.get("mrr")
            coverage = self.compute_cov_at_n(ir_method, k, tol)
            diversity = self.compute_div_at_n(ir_method, k, tol)
            avg_popularity = self.compute_avg_pop_at_n(ir_method, k, tol)
            return {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "mrr": mrr,
                "coverage": coverage,
                "diversity": diversity,
                "avg_popularity": avg_popularity,
            }
        return None
    def compute_avg_pop_at_n(
        self,
        ir_method: IRMethod,
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
        total_songs = self.DEBUG_NR_SAMPLES or self.nr_samples
        pop_list = list()
        precision_at_k_list = list()
        recall_at_k_list = list()
        ndcg_at_k_list = list()
        mrr_at_k_list = list()

        for i in range(0, total_songs):
            artist = self.dataset_loader.id_artist_song_album["artist"].iloc[i]
            song = self.dataset_loader.id_artist_song_album["song"].iloc[i]

            results = self.compute_results(ir_method, song, artist, k)
            if results is not None:
                for result in results.get("search_results"):  # type: ignore
                    pop_list.append(result["popularity"])  # type: ignore
                precision_at_k_list.append(results.get("precision"))
                recall_at_k_list.append(results.get("recall"))
                ndcg_at_k_list.append(results.get("ndcg"))
                mrr_at_k_list.append(results.get("mrr"))

        total_popularity = sum(pop_list)

        diversity = total_popularity / len(pop_list)
        return (
            np.round(diversity, tol),
            np.mean(precision_at_k_list).round(tol),
            np.mean(recall_at_k_list).round(tol),
            np.mean(ndcg_at_k_list).round(tol),
            np.mean(mrr_at_k_list).round(tol),
        )
