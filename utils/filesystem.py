import fsspec
from fsspec.implementations.local import LocalFileSystem


def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """
    Checks if `fs` is a remote filesystem.

    Args:
        fs (`fsspec.spec.AbstractFileSystem`):
            An abstract super-class for pythonic file-systems, e.g. `fsspec.filesystem(\'file\')` or `s3fs.S3FileSystem`.
    """
    return not isinstance(fs, LocalFileSystem)
