from typing import Union, Optional, TYPE_CHECKING

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.index_enhancement.Faiss_plugin import FaissIndex
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.index_enhancement.base_plugin import MissingIndex, \
    SearchResults, BatchedSearchResults, NearestExamplesResults, \
    BatchedNearestExamplesResults, BaseIndex, logger

if TYPE_CHECKING:
    from .arrow_dataset import Dataset  # noqa: F401

    try:
        import faiss  # noqa: F401

    except ImportError:
        pass
from pathlib import PurePath

import numpy as np


class IndexablePlugin:
    """Add indexing features to `datasets.Dataset`"""

    def __init__(self):
        self._indexes: dict[str, BaseIndex] = {}

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def is_index_initialized(self, index_name: str) -> bool:
        return index_name in self._indexes

    def _check_index_is_initialized(self, index_name: str):
        if not self.is_index_initialized(index_name):
            raise MissingIndex(
                f"Index with index_name '{index_name}' not initialized yet. Please make sure that you call `add_faiss_index` first."
            )

    def list_indexes(self) -> list[str]:
        """List the `colindex_nameumns`/identifiers of all the attached indexes."""
        return list(self._indexes)

    def get_index(self, index_name: str) -> BaseIndex:
        """List the `index_name`/identifiers of all the attached indexes.

        Args:
            index_name (`str`): Index name.

        Returns:
            [`BaseIndex`]
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name]

    def add_faiss_index(
            self,
            column: str,
            index_name: Optional[str] = None,
            device: Optional[Union[int, list[int]]] = None,
            string_factory: Optional[str] = None,
            metric_type: Optional[int] = None,
            custom_index: Optional["faiss.Index"] = None,
            batch_size: int = 1000,
            train_size: Optional[int] = None,
            faiss_verbose: bool = False,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of the specified column.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index, see more below).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory

        Args:
            column (`str`): The column of the vectors to add to the index.
            index_name (Optional `str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
                By default it corresponds to `column`.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (Optional `str`): This is passed to the index factory of Faiss to create the index. Default index class is IndexFlatIP.
            metric_type (Optional `int`): Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (Optional `faiss.Index`): Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (Optional `int`): If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False): Enable the verbosity of the Faiss index.
        """
        index_name = index_name if index_name is not None else column
        faiss_index = FaissIndex(
            device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index
        )
        faiss_index.add_vectors(
            self, column=column, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose
        )
        self._indexes[index_name] = faiss_index

    def add_faiss_index_from_external_arrays(
            self,
            external_arrays: np.array,
            index_name: str,
            device: Optional[Union[int, list[int]]] = None,
            string_factory: Optional[str] = None,
            metric_type: Optional[int] = None,
            custom_index: Optional["faiss.Index"] = None,
            batch_size: int = 1000,
            train_size: Optional[int] = None,
            faiss_verbose: bool = False,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of `external_arrays`.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory

        Args:
            external_arrays (`np.array`): If you want to use arrays from outside the lib for the index, you can set `external_arrays`.
                It will use `external_arrays` to create the Faiss index instead of the arrays in the given `column`.
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (Optional `str`): This is passed to the index factory of Faiss to create the index. Default index class is IndexFlatIP.
            metric_type (Optional `int`): Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (Optional `faiss.Index`): Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (Optional `int`): If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False): Enable the verbosity of the Faiss index.
        """
        faiss_index = FaissIndex(
            device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index
        )
        faiss_index.add_vectors(
            external_arrays, column=None, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose
        )
        self._indexes[index_name] = faiss_index

    def save_faiss_index(self, index_name: str, file: Union[str, PurePath], storage_options: Optional[dict] = None):
        """Save a FaissIndex on disk.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk or remote URI (e.g. `"s3://my-bucket/index.faiss"`).
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.11.0"/>

        """
        index = self.get_index(index_name)
        if not isinstance(index, FaissIndex):
            raise ValueError(f"Index '{index_name}' is not a FaissIndex but a '{type(index)}'")
        index.save(file, storage_options=storage_options)
        logger.info(f"Saved FaissIndex {index_name} at {file}")

    def load_faiss_index(
            self,
            index_name: str,
            file: Union[str, PurePath],
            device: Optional[Union[int, list[int]]] = None,
            storage_options: Optional[dict] = None,
    ):
        """Load a FaissIndex from disk.

        If you want to do additional configurations, you can have access to the faiss index object by doing
        `.get_index(index_name).faiss_index` to make it fit your needs.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to
                call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk or remote URI (e.g. `"s3://my-bucket/index.faiss"`).
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.11.0"/>

        """
        index = FaissIndex.load(file, device=device, storage_options=storage_options)
        if index.faiss_index.ntotal != len(self):
            raise ValueError(
                f"Index size should match Dataset size, but Index '{index_name}' at {file} has {index.faiss_index.ntotal} elements while the dataset has {len(self)} examples."
            )
        self._indexes[index_name] = index
        logger.info(f"Loaded FaissIndex {index_name} from {file}")

    def drop_index(self, index_name: str):
        """Drop the index with the specified column.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
        """
        del self._indexes[index_name]

    def search(self, index_name: str, query: Union[str, np.array], k: int = 10, **kwargs) -> SearchResults:
        """Find the nearest examples indices in the dataset to the query.

        Args:
            index_name (`str`):
                The name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            `(scores, indices)`:
                A tuple of `(scores, indices)` where:
                - **scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples
                - **indices** (`List[List[int]]`): the indices of the retrieved examples
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name].search(query, k, **kwargs)

    def search_batch(
            self, index_name: str, queries: Union[list[str], np.array], k: int = 10, **kwargs
    ) -> BatchedSearchResults:
        """Find the nearest examples indices in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            `(total_scores, total_indices)`:
                A tuple of `(total_scores, total_indices)` where:
                - **total_scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples per query
                - **total_indices** (`List[List[int]]`): the indices of the retrieved examples per query
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name].search_batch(queries, k, **kwargs)

    def get_nearest_examples(
            self, index_name: str, query: Union[str, np.array], k: int = 10, **kwargs
    ) -> NearestExamplesResults:
        """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The index_name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            `(scores, examples)`:
                A tuple of `(scores, examples)` where:
                - **scores** (`List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples
                - **examples** (`dict`): the retrieved examples
        """
        self._check_index_is_initialized(index_name)
        scores, indices = self.search(index_name, query, k, **kwargs)
        top_indices = [i for i in indices if i >= 0]
        return NearestExamplesResults(scores[: len(top_indices)], self[top_indices])

    def get_nearest_examples_batch(
            self, index_name: str, queries: Union[list[str], np.array], k: int = 10, **kwargs
    ) -> BatchedNearestExamplesResults:
        """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            `(total_scores, total_examples)`:
                A tuple of `(total_scores, total_examples)` where:
                - **total_scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples per query
                - **total_examples** (`List[dict]`): the retrieved examples per query
        """
        self._check_index_is_initialized(index_name)
        total_scores, total_indices = self.search_batch(index_name, queries, k, **kwargs)
        total_scores = [
            scores_i[: len([i for i in indices_i if i >= 0])]
            for scores_i, indices_i in zip(total_scores, total_indices)
        ]
        total_samples = [self[[i for i in indices if i >= 0]] for indices in total_indices]
        return BatchedNearestExamplesResults(total_scores, total_samples)
