import importlib.util
from pathlib import PurePath
from typing import Optional, Union

import faiss
import fsspec
import numpy as np
from tqdm import tqdm

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.arrow_dataset import Dataset
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.index_enhancement.base_plugin import SearchResults, \
    BatchedSearchResults, BaseIndex, logger
from CLTrainingFramework.dataset.schema import Sequence

_has_faiss = importlib.util.find_spec("faiss") is not None


class FaissIndex(BaseIndex):
    """
    感谢huggingface，这个相当于faiss的plus版本

    """

    # TODO 这玩意可以再改一下，后面索引OOD数据或者别的在CL上无监督很有用
    def __init__(
            self,
            device: Optional[Union[int, list[int]]] = None,
            string_factory: Optional[str] = None,
            metric_type: Optional[int] = None,
            custom_index: Optional["faiss.Index"] = None,
    ):
        """
        Create a Dense index using Faiss. You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory
        """
        if string_factory is not None and custom_index is not None:
            raise ValueError("Please specify either `string_factory` or `custom_index` but not both.")
        if device is not None and custom_index is not None:
            raise ValueError(
                "Cannot pass both 'custom_index' and 'device'. "
                "Pass 'custom_index' already transferred to the target device instead."
            )
        self.device = device
        self.string_factory = string_factory
        self.metric_type = metric_type
        self.faiss_index = custom_index
        if not _has_faiss:
            raise ImportError(
                "You must install Faiss to use FaissIndex. To do so you can run `conda install -c pytorch faiss-cpu` or `conda install -c pytorch faiss-gpu`. "
                "A community supported package is also available on pypi: `pip install faiss-cpu` or `pip install faiss-gpu`. "
                "Note that pip may not have the latest version of FAISS, and thus, some of the latest features and bug fixes may not be available."
            )

    def add_vectors(
            self,
            vectors: Union[np.array, "Dataset"],
            column: Optional[str] = None,
            batch_size: int = 1000,
            train_size: Optional[int] = None,
            faiss_verbose: Optional[bool] = None,
    ):
        """
        Add vectors to the index.
        If the arrays are inside a certain column, you can specify it using the `column` argument.
        """
        import faiss  # noqa: F811

        if column and not isinstance(vectors.features[column], Sequence):
            raise ValueError(
                f"Wrong feature type for column '{column}'. Expected 1d array, got {vectors.features[column]}"
            )

        # Create index
        if self.faiss_index is None:
            size = len(vectors[0]) if column is None else len(vectors[0][column])
            if self.string_factory is not None:
                if self.metric_type is None:
                    index = faiss.index_factory(size, self.string_factory)
                else:
                    index = faiss.index_factory(size, self.string_factory, self.metric_type)
            else:
                if self.metric_type is None:
                    index = faiss.IndexFlat(size)
                else:
                    index = faiss.IndexFlat(size, self.metric_type)

            self.faiss_index = self._faiss_index_to_device(index, self.device)
            logger.info(f"Created faiss index of type {type(self.faiss_index)}")

        # Set verbosity level
        if faiss_verbose is not None:
            self.faiss_index.verbose = faiss_verbose
            if hasattr(self.faiss_index, "index") and self.faiss_index.index is not None:
                self.faiss_index.index.verbose = faiss_verbose
            if hasattr(self.faiss_index, "quantizer") and self.faiss_index.quantizer is not None:
                self.faiss_index.quantizer.verbose = faiss_verbose
            if hasattr(self.faiss_index, "clustering_index") and self.faiss_index.clustering_index is not None:
                self.faiss_index.clustering_index.verbose = faiss_verbose

        # Train
        if train_size is not None:
            train_vecs = vectors[:train_size] if column is None else vectors[:train_size][column]
            logger.info(f"Training the index with the first {len(train_vecs)} vectors")
            self.faiss_index.train(train_vecs)
        else:
            logger.info("Ignored the training step of the faiss index as `train_size` is None.")

        # Add vectors
        logger.info(f"Adding {len(vectors)} vectors to the faiss index")
        for i in tqdm(range(0, len(vectors), batch_size)):
            vecs = vectors[i: i + batch_size] if column is None else vectors[i: i + batch_size][column]
            self.faiss_index.add(vecs)

    @staticmethod
    def _faiss_index_to_device(index: "faiss.Index", device: Optional[Union[int, list[int]]] = None) -> "faiss.Index":
        """
        Sends a faiss index to a device.
        A device can either be a positive integer (GPU id), a negative integer (all GPUs),
            or a list of positive integers (select GPUs to use), or `None` for CPU.
        """

        # If device is not specified, then it runs on CPU.
        if device is None:
            return index

        import faiss  # noqa: F811

        # If the device id is given as an integer
        if isinstance(device, int):
            # Positive integers are directly mapped to GPU ids
            if device > -1:
                faiss_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(faiss_res, device, index)
            # And negative integers mean using all GPUs
            else:
                index = faiss.index_cpu_to_all_gpus(index)
        # Device ids given as a list mean mapping to those devices specified.
        elif isinstance(device, (list, tuple)):
            index = faiss.index_cpu_to_gpus_list(index, gpus=list(device))
        else:
            raise TypeError(
                f"The argument type: {type(device)} is not expected. "
                + "Please pass in either nothing, a positive int, a negative int, or a list of positive ints."
            )

        return index

    def search(self, query: np.array, k=10, **kwargs) -> SearchResults:
        """
        Find the nearest examples indices to the query.

        Args:
            query (`np.array`): The query as a numpy array.
            k (`int`): The number of examples to retrieve.

        Output:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        if len(query.shape) != 1 and (len(query.shape) != 2 or query.shape[0] != 1):
            raise ValueError("Shape of query is incorrect, it has to be either a 1D array or 2D (1, N)")

        queries = query.reshape(1, -1)
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self.faiss_index.search(queries, k, **kwargs)
        return SearchResults(scores[0], indices[0].astype(int))

    def search_batch(self, queries: np.array, k=10, **kwargs) -> BatchedSearchResults:
        """Find the nearest examples indices to the queries.

        Args:
            queries (`np.array`): The queries as a numpy array.
            k (`int`): The number of examples to retrieve.

        Output:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        if len(queries.shape) != 2:
            raise ValueError("Shape of query must be 2D")
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self.faiss_index.search(queries, k, **kwargs)
        return BatchedSearchResults(scores, indices.astype(int))

    def save(self, file: Union[str, PurePath], storage_options: Optional[dict] = None):
        """Serialize the FaissIndex on disk"""
        import faiss  # noqa: F811

        if self.device is not None and isinstance(self.device, (int, list, tuple)):
            index = faiss.index_gpu_to_cpu(self.faiss_index)
        else:
            index = self.faiss_index

        with fsspec.open(str(file), "wb", **(storage_options or {})) as f:
            faiss.write_index(index, faiss.BufferedIOWriter(faiss.PyCallbackIOWriter(f.write)))

    @classmethod
    def load(
            cls,
            file: Union[str, PurePath],
            device: Optional[Union[int, list[int]]] = None,
            storage_options: Optional[dict] = None,
    ) -> "FaissIndex":
        """Deserialize the FaissIndex from disk"""
        import faiss  # noqa: F811

        # Instances of FaissIndex is essentially just a wrapper for faiss indices.
        faiss_index = cls(device=device)
        with fsspec.open(str(file), "rb", **(storage_options or {})) as f:
            index = faiss.read_index(faiss.BufferedIOReader(faiss.PyCallbackIOReader(f.read)))
        faiss_index.faiss_index = faiss_index._faiss_index_to_device(index, faiss_index.device)
        return faiss_index
