from pathlib import PurePath
from typing import NamedTuple, Union

import numpy as np

from CLTrainingFramework.utils.logging import get_logger


class MissingIndex(Exception):
    pass


class SearchResults(NamedTuple):
    scores: list[float]
    indices: list[int]


class BatchedSearchResults(NamedTuple):
    total_scores: list[list[float]]
    total_indices: list[list[int]]


class NearestExamplesResults(NamedTuple):
    scores: list[float]
    examples: dict


class BatchedNearestExamplesResults(NamedTuple):
    total_scores: list[list[float]]
    total_examples: list[dict]


class BaseIndex:
    """Base class for indexing"""

    def search(self, query, k: int = 10, **kwargs) -> SearchResults:
        """
        To implement.
        This method has to return the scores and the indices of the retrieved examples given a certain query.
        """
        raise NotImplementedError

    def search_batch(self, queries:Union[list[str],np.ndarray], k: int = 10, **kwargs) -> BatchedSearchResults:
        """Find the nearest examples indices to the query.

        Args:
            queries (`Union[List[str], np.ndarray]`): The queries as a list of strings if `column` is a text index or as a numpy array if `column` is a vector index.
            k (`int`): The number of examples to retrieve per query.

        Output:
            total_scores (`List[List[float]]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        total_scores, total_indices = [], []
        for query in queries:
            scores, indices = self.search(query, k)
            total_scores.append(scores)
            total_indices.append(indices)
        return BatchedSearchResults(total_scores, total_indices)

    def save(self, file: Union[str, PurePath]):
        """Serialize the index on disk"""
        raise NotImplementedError

    @classmethod
    def load(cls, file: Union[str, PurePath]) -> "BaseIndex":
        """Deserialize the index from disk"""
        raise NotImplementedError


logger = get_logger(__name__)
