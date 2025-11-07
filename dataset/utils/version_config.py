import importlib.metadata
import importlib.util
import logging
import os

from packaging import version

logger = logging.getLogger(__name__.split(".", 1)[0])  # to avoid circular import from framework's logging

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_FALSE_VALUES = {"0", "OFF", "NO", "FALSE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})
ENV_VARS_FALSE_AND_AUTO_VALUES = ENV_VARS_FALSE_VALUES.union({"AUTO"})
DILL_VERSION = version.parse(importlib.metadata.version("dill"))
FSSPEC_VERSION = version.parse(importlib.metadata.version("fsspec"))
PANDAS_VERSION = version.parse(importlib.metadata.version("pandas"))
PYARROW_VERSION = version.parse(importlib.metadata.version("pyarrow"))
HF_HUB_VERSION = version.parse(importlib.metadata.version("huggingface_hub"))

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()

TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
    if TORCH_AVAILABLE:
        try:
            TORCH_VERSION = version.parse(importlib.metadata.version("torch"))
            logger.info(f"PyTorch version {TORCH_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling PyTorch because USE_TF is set")

POLARS_VERSION = "N/A"
POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None

if POLARS_AVAILABLE:
    try:
        POLARS_VERSION = version.parse(importlib.metadata.version("polars"))
        logger.info(f"Polars version {POLARS_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

DUCKDB_VERSION = "N/A"
DUCKDB_AVAILABLE = importlib.util.find_spec("duckdb") is not None

if DUCKDB_AVAILABLE:
    try:
        DUCKDB_VERSION = version.parse(importlib.metadata.version("duckdb"))
        logger.info(f"Duckdb version {DUCKDB_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

TF_VERSION = "N/A"
TF_AVAILABLE = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
    if TF_AVAILABLE:
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for package in [
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        ]:
            try:
                TF_VERSION = version.parse(importlib.metadata.version(package))
            except importlib.metadata.PackageNotFoundError:
                continue
            else:
                break
        else:
            TF_AVAILABLE = False
    if TF_AVAILABLE:
        if TF_VERSION.major < 2:
            logger.info(f"TensorFlow found but with version {TF_VERSION}. `datasets` requires version 2 minimum.")
            TF_AVAILABLE = False
        else:
            logger.info(f"TensorFlow version {TF_VERSION} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")

JAX_VERSION = "N/A"
JAX_AVAILABLE = False

if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    JAX_AVAILABLE = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
    if JAX_AVAILABLE:
        try:
            JAX_VERSION = version.parse(importlib.metadata.version("jax"))
            logger.info(f"JAX version {JAX_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling JAX because USE_JAX is set to False")

# Optional tools for data loading
SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None

# Optional tools for feature decoding
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
IS_OPUS_SUPPORTED = importlib.util.find_spec("soundfile") is not None and version.parse(
    importlib.import_module("soundfile").__libsndfile_version__
) >= version.parse("1.0.31")
IS_MP3_SUPPORTED = importlib.util.find_spec("soundfile") is not None and version.parse(
    importlib.import_module("soundfile").__libsndfile_version__
) >= version.parse("1.1.0")
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
PDFPLUMBER_AVAILABLE = importlib.util.find_spec("pdfplumber") is not None

# Optional compression tools
RARFILE_AVAILABLE = importlib.util.find_spec("rarfile") is not None
ZSTANDARD_AVAILABLE = importlib.util.find_spec("zstandard") is not None
LZ4_AVAILABLE = importlib.util.find_spec("lz4") is not None
PY7ZR_AVAILABLE = importlib.util.find_spec("py7zr") is not None
