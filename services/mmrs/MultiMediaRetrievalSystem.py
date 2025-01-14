""""""

import logging
from typing import Any, Dict, Hashable, List, Set

import numpy as np
import pandas as pd


class MultiMediaRetrievalSystem:
    logger: logging.Logger = logging.getLogger(__name__)

    DISPLAY_TOL: int = 4
    FALLBACK_RESULTS: Dict[str, str | float | List[Dict[Hashable, Any]] | None] = {
        "search_results": [],
        "precision": None,
        "recall": None,
        "ndcg": None,
        "mrr": None,
        "message": "Query item not found",
    }
    RELEVANT_COLUMNS: List[str] = [
        "id",
        "artist",
        "song",
        "url",
        "popularity",
        "genre",
        "(tag, weight)",
    ]

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
        *datasets: pd.DataFrame,
    ):
        """Prepare the data for the search process.

        Args:
            id_information_mmsr (pd.DataFrame): The MMSR information data.
            id_genres (pd.DataFrame): The genre data.
            id_urls (pd.DataFrame): The URL data.
        """
        data = None
        for dataset in datasets:
            if data is None:
                data = dataset
            else:
                data = data.merge(dataset, on="id")
        self.logger.debug("Prepared data for IR process")
        self.data: pd.DataFrame = data  # type: ignore

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

    def get_formatted_search_results(
        self, query_results: pd.DataFrame
    ) -> List[Dict[Hashable, Any]]:
        """Format the search results for display.

        Args:
            query_results (pd.DataFrame): The search results.

        Returns:
            List[Dict[str, str]]: The formatted search results.
        """
        return query_results[self.RELEVANT_COLUMNS].to_dict(orient="records")

    def baseline(
        self,
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
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
        search_results = self.get_formatted_search_results(query_result)
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
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
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
        search_results = self.get_formatted_search_results(query_result)
        return {
            "search_results": search_results,  # type: ignore
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def bert(
        self,
        bert: np.ndarray[Any, np.dtype[np.float64]],
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
        """
        Performs a cosine-similarity-based search on BERT embeddings.
        """
        self.logger.debug(
            f"Generating BERT-based search results for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # retrieve query item bert
        query_item_bert: np.ndarray[Any, np.dtype[np.float64]] = bert[
            self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        ]

        cosine_similarities: np.ndarray[Any, np.dtype[np.float64]] = np.dot(
            bert, query_item_bert.T
        ).flatten()

        # Get the top N items
        modified_N = N + 1  # exclude the query item itself
        top_N_indices: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(
            cosine_similarities
        )[-modified_N:-1][::-1]

        # Retrieve those items from self.data
        query_result: pd.DataFrame = self.data.iloc[top_N_indices]

        # Compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        search_results = self.get_formatted_search_results(query_result)
        return {
            "search_results": search_results,  # type: ignore
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def blf_spectral(
        self,
        blf_spectral: np.ndarray[Any, np.dtype[np.float64]],
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
        """
        Performs a cosine-similarity–based search on blf_spectral embeddings.
        """
        self.logger.debug(
            f"Generating blf_spectral-based search results for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # retrieve query item blf_spectral
        query_item_blf_spectral: np.ndarray[Any, np.dtype[np.float64]] = blf_spectral[
            self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        ]

        cosine_similarities: np.ndarray[Any, np.dtype[np.float64]] = np.dot(
            blf_spectral, query_item_blf_spectral.T
        ).flatten()

        # Get the top N items
        modified_N = N + 1  # exclude the query item itself
        top_N_indices: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(
            cosine_similarities
        )[-modified_N:-1][::-1]

        # Retrieve those items from self.data
        query_result: pd.DataFrame = self.data.iloc[top_N_indices]

        # Compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        search_results = self.get_formatted_search_results(query_result)
        return {
            "search_results": search_results,  # type: ignore
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def music_nn(
        self,
        music_nn: np.ndarray[Any, np.dtype[np.float64]],
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
        """
        Performs a cosine-similarity–based search on music_nn embeddings.
        """
        self.logger.debug(
            f"Generating music_nn-based search results for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # retrieve query item music_nn
        query_item_music_nn: np.ndarray[Any, np.dtype[np.float64]] = music_nn[
            self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        ]

        cosine_similarities: np.ndarray[Any, np.dtype[np.float64]] = np.dot(
            music_nn, query_item_music_nn.T
        ).flatten()

        # Get the top N items
        modified_N = N + 1  # exclude the query item itself
        top_N_indices: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(
            cosine_similarities
        )[-modified_N:-1][::-1]

        # Retrieve those items from self.data
        query_result: pd.DataFrame = self.data.iloc[top_N_indices]

        # Compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        search_results = self.get_formatted_search_results(query_result)
        return {
            "search_results": search_results,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def resnet(
        self,
        resnet: np.ndarray[Any, np.dtype[np.float64]],
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
        """
        Performs a eucledian-distance–based search on resnet embeddings.
        """
        self.logger.debug(
            f"Generating resnet-based search results for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # retrieve query item resnet
        # query_item_resnet: np.ndarray[Any, np.dtype[np.float64]] = resnet[
        #     self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        # ]

        # # euclidean distance
        # distances = np.linalg.norm(resnet - query_item_resnet, axis=1)
        # top_indices = np.argsort(distances)  # ascending order
        # top_N_indices = top_indices[1 : N + 1]  # noqa: E203

        # retrieve query item music_nn
        query_item_resnet: np.ndarray[Any, np.dtype[np.float64]] = resnet[
            self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        ]

        cosine_similarities: np.ndarray[Any, np.dtype[np.float64]] = np.dot(
            resnet, query_item_resnet.T
        ).flatten()

        # Get the top N items
        modified_N = N + 1  # exclude the query item itself
        top_N_indices: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(
            cosine_similarities
        )[-modified_N:-1][::-1]

        query_result: pd.DataFrame = self.data.iloc[top_N_indices]

        # Compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        search_results = self.get_formatted_search_results(query_result)
        return {
            "search_results": search_results,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "mrr": mmr,
            "message": None,
        }

    def vgg19(
        self,
        vgg19: np.ndarray[Any, np.dtype[np.float64]],
        artist: str | None,
        song_title: str | None,
        N: int = 10,
    ) -> Dict[str, str | float | List[Dict[Hashable, Any]] | None]:
        """
        Performs a eucledian-distance–based search on vgg19 embeddings.
        """
        self.logger.debug(
            f"Generating vgg19-based search results for {artist} - {song_title}"
        )

        query_item = self.retrieve_query_item(self.data, artist, song_title)
        if query_item is None:
            return self.FALLBACK_RESULTS

        # # retrieve query item vgg19
        # query_item_vgg19: np.ndarray[Any, np.dtype[np.float64]] = vgg19[
        #     self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        # ]

        # # euclidean distance
        # distances = np.linalg.norm(vgg19 - query_item_vgg19, axis=1)
        # top_indices = np.argsort(distances)  # ascending order
        # top_N_indices = top_indices[1 : N + 1]  # noqa: E203
        query_item_vgg19: np.ndarray[Any, np.dtype[np.float64]] = vgg19[
            self.data.index[self.data["id"] == query_item["id"].values[0]]  # type: ignore
        ]

        cosine_similarities: np.ndarray[Any, np.dtype[np.float64]] = np.dot(
            vgg19, query_item_vgg19.T
        ).flatten()

        # Get the top N items
        modified_N = N + 1  # exclude the query item itself
        top_N_indices: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(
            cosine_similarities
        )[-modified_N:-1][::-1]

        query_result: pd.DataFrame = self.data.iloc[top_N_indices]

        # Compute metrics
        precision: float = self._compute_precision_at_k(query_result, query_item, N)
        recall: float = self._compute_recall_at_k(query_result, query_item, N)
        ndcg: float = self._compute_ndcg_at_k(query_result, query_item, N)
        mmr: float = self._compute_mrr_at_k(query_result, query_item, N)

        search_results = self.get_formatted_search_results(query_result)
        return {
            "search_results": search_results,
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

    def find_most_dissimilar(
        self,
        query: pd.DataFrame,
        query_results: pd.DataFrame,
    ):
        pass
