""""""

import logging
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


class MultiMediaRetrievalSystem:
    logger: logging.Logger = logging.getLogger(__name__)

    DISPLAY_TOL: int = 4
    FALLBACK_RESULTS: Dict[str, str | float | List[Dict[str, str]] | None] = {
        "search_results": [],
        "precision": None,
        "recall": None,
        "ndcg": None,
        "mrr": None,
        "message": "Query item not found",
    }

    def __init__(self):
        pass

    def autocomplete(self) -> Dict[str, List[str]]:
        """Autocomplete formatting for UI input fields for better testing and user experience."""

        # convert song data to a dict with artist as key and a list of song titles as value
        autocomplete_options: Dict[str, List[str]] = (  # type: ignore
            self.data.groupby("artist")["song"].apply(list).to_dict()  # type: ignore
        )
        self.logger.debug("Generated autocomplete options")
        return autocomplete_options

    def prepare_data(
        self,
        id_information_mmsr: pd.DataFrame,
        id_genres: pd.DataFrame,
        id_urls: pd.DataFrame,
    ) -> None:
        """Prepare the data for the search process.

        Args:
            id_information_mmsr (pd.DataFrame): The MMSR information data.
            id_genres (pd.DataFrame): The genre data.
            id_urls (pd.DataFrame): The URL data.
        """
        # merge with the genre data
        data = id_information_mmsr.merge(id_genres, on="id")
        # merge with the URL data
        self.data: pd.DataFrame = data.merge(id_urls, on="id")
        self.logger.debug("Prepared data for IR process")

    def retrieve_query_item(
        self,
        data: pd.DataFrame,
        artist: str | None,
        song_title: str | None,
    ) -> pd.DataFrame | None:
        """Retrieve the query item from the catalog data.

        Args:
            data (pd.DataFrame): The catalog data.
            artist (str): The artist of the query item.
            song_title (str): The song title of the query item.

        Returns:
            pd.DataFrame: The query item data.
        """
        query_item: pd.DataFrame = data[
            (data["artist"] == artist) & (data["song"] == song_title)
        ]
        if query_item.empty:
            self.logger.warning(f"Query item not found: {artist} - {song_title}")
            return None
        return query_item

    def baseline(
        self,
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[str, str]] | None]:
        """Create a simple baseline system that randomly (regardless of the query) selects N items
        from the catalog (excluding the query item); Make sure that the system produces new
        results for each query!

        Args:
            artist (str): The artist of the query item.
            song_title (str): The song title of the query item.
            N (int, optional): The number of items to return. Defaults to 10.

        Returns:
            Dict[str, float | List[Dict[str, str]]]: The search results and evaluation metrics.
        """
        self.logger.debug(
            f"Generating baseline search results for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # sample N random items
        query_result: pd.DataFrame = self.data.sample(n=N)

        # compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        # return only the relevant columns: artist, song, and url
        search_results = query_result[["artist", "song", "url"]].to_dict(  # type: ignore
            orient="records"
        )
        return {
            "search_results": search_results,  # type: ignore
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def tfidf(
        self,
        tfidf: np.ndarray[Any, np.dtype[np.float64]],
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[str, str]] | None]:
        """"""
        self.logger.debug(
            f"Generating tfidf search results based on lyrics for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # retrieve query item tfidf
        query_item_tfidf: np.ndarray[Any, np.dtype[np.float64]] = tfidf[
            self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        ]

        # calculate cosine similarity
        cosine_similarities: np.ndarray[Any, np.dtype[np.float64]] = np.dot(
            tfidf, query_item_tfidf.T
        ).flatten()

        # get the top N items
        modified_N = N + 1  # exclude the query item
        top_N_indices: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(
            cosine_similarities
        )[-modified_N:-1][::-1]

        # get the top N items
        query_result: pd.DataFrame = self.data.iloc[top_N_indices]

        # compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        # return only the relevant columns: artist, song, and url
        search_results = query_result[["artist", "song", "url"]].to_dict(  # type: ignore
            orient="records"
        )
        return {
            "search_results": search_results,  # type: ignore
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def _compute_precision_at_k(
        self,
        query_result: pd.DataFrame,
        query_item: pd.DataFrame,
        k: int,
        tol: int = DISPLAY_TOL,
    ) -> float:
        """Compute the Precision@K score.

        Both datafames have a "genre" column.

        Args:
            results (pd.DataFrame): The retrieval results.
            query_item (pd.DataFrame): The query item data.
            k (int): The number of items to consider.

        Returns:
            float: The Precision@K score.
        """
        top_k_genres: List[Set[str]] = (
            query_result.head(k)["genre"].apply(lambda x: set(x)).tolist()  # type: ignore
        )
        query_item_genres: Set[str] = set(query_item["genre"].tolist()[0])  # type: ignore

        relevant_items: int = sum(
            1 for genres in top_k_genres if query_item_genres.intersection(genres)
        )
        precision_at_k: float = relevant_items / k
        self.logger.debug(f"Precision@{k}: {precision_at_k}")
        return np.round(precision_at_k, tol)

    def _compute_recall_at_k(
        self,
        query_result: pd.DataFrame,
        query_item: pd.DataFrame,
        k: int,
        tol: int = DISPLAY_TOL,
    ):
        """Compute the Recall@K score.

        Args:
            query_result (pd.DataFrame): The retrieval results.
            query_item (pd.DataFrame): The query item data.
            k (int): The number of items to consider.

        Returns:
            float: The Recall@K score.
        """
        data_genres: List[Set[str]] = self.data["genre"].apply(lambda x: set(x)).tolist()  # type: ignore
        top_k_genres: Set[str] = (
            query_result.head(k)["genre"].apply(lambda x: set(x)).tolist()  # type: ignore
        )
        query_item_genres: Set[str] = set(query_item["genre"].tolist()[0])  # type: ignore

        total_relevant_items: int = sum(
            1 for genres in data_genres if query_item_genres.intersection(genres)
        )
        relevant_items: int = sum(
            1 for genres in top_k_genres if query_item_genres.intersection(genres)
        )
        recall_at_k: float = relevant_items / total_relevant_items
        self.logger.debug(f"Recall@{k}: {recall_at_k}")
        return np.round(recall_at_k, tol)

    def _compute_ndcg_at_k(
        self,
        query_result: pd.DataFrame,
        query_item: pd.DataFrame,
        k: int,
        tol: int = DISPLAY_TOL,
    ) -> float:
        """Compute the NDCG@K score.

        Args:
            query_result (pd.DataFrame): The retrieval results.
            query_item (pd.DataFrame): The query item data.
            k (int): The number of items to consider.

        Returns:
            float: The NDCG@K score.
        """
        top_k_genres: List[Set[str]] = (
            query_result.head(k)["genre"].apply(lambda x: set(x)).tolist()  # type: ignore
        )
        query_item_genres: Set[str] = set(query_item["genre"].tolist()[0])  # type: ignore

        # gain assignment
        gain: List[int] = [
            1 if query_item_genres.intersection(genres) else 0
            for genres in top_k_genres
        ]

        # calculate discounted cumulative gain DCG
        dcg: float = sum(
            gain[i] / np.log2(i + 1 + 1e-5) for i in range(k)
        )  # numerical stability

        # calculate ideal discounted cumulative gain IDCG
        ideal_gain: List[int] = [1] * len(gain)
        idcg: float = sum(
            ideal_gain[i] / np.log2(i + 1 + 1e-5) for i in range(k)
        )  # numerical stability

        # calculate normalized discounted cumulative gain NDCG
        ndcg: float = dcg / idcg
        self.logger.debug(f"NDCG@{k}: {ndcg}")
        return np.round(ndcg, tol)

    def _compute_mrr_at_k(
        self,
        query_result: pd.DataFrame,
        query_item: pd.DataFrame,
        k: int,
        tol: int = DISPLAY_TOL,
    ):
        """Compute the MRR@K score.

        Args:
            query_result (pd.DataFrame): The retrieval results.
            query_item (pd.DataFrame): The query item data.
            k (int): The number of items to consider.

        Returns:
            float: The MRR@K score.
        """
        top_k_genres: List[Set[str]] = (
            query_result.head(k)["genre"].apply(lambda x: set(x)).tolist()  # type: ignore
        )
        query_item_genres: Set[str] = set(query_item["genre"].tolist()[0])  # type: ignore

        # find the first relevant item
        for i, genres in enumerate(top_k_genres):
            if query_item_genres.intersection(genres):
                mrr_at_k: float = 1 / (i + 1)
                self.logger.debug(f"MRR@{k}: {mrr_at_k}")
                return np.round(mrr_at_k, tol)

        # if no relevant item is found, return zero
        return 0.0
