# mine
import os

from filelock import FileLock as FileLock_
from filelock import UnixFileLock
from filelock import __version__ as _filelock_version
from packaging import version


class FileLock(FileLock_):
    """
    一些操作系统对文件路径的长度限制要求不能超过255个字符（windows10这一块），这里通过hash算法缩短成一个合法长度的路径
    """

    MAX_FILENAME_LENGTH = 255

    def __init__(self, lock_file, *args, **kwargs):
        # The "mode" argument is required if we want to use the current umask in filelock >= 3.10
        # In previous it was already using the current umask.
        if "mode" not in kwargs and version.parse(_filelock_version) >= version.parse(
            "3.10.0"
        ):
            # 我草你能看到这里的代码？那我解释一下
            # os.umask有两个作用：返回旧的umask，并且设置新的umask
            umask = os.umask(0o666)
            os.umask(umask)
            # 文件的最大权限是0o666（rw-rw-rw-）,也就是110110110，我们通过这样得到系统自带的umask然后传入mode这个参数（类似chmod这个命令的作用）
            kwargs["mode"] = 0o666 & ~umask
        lock_file = self.hash_filename_if_too_long(lock_file)
        super().__init__(lock_file, *args, **kwargs)

    @classmethod
    def hash_filename_if_too_long(cls, path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        filename = os.path.basename(path)
        max_filename_length = cls.MAX_FILENAME_LENGTH
        if issubclass(cls, UnixFileLock):
            # 忘记从哪个库里抄过来的了，反正有些区别
            max_filename_length = min(
                max_filename_length, os.statvfs(os.path.dirname(path)).f_namemax
            )
        if len(filename) > max_filename_length:
            dirname = os.path.dirname(path)
            hashed_filename = str(hash(filename))
            new_filename = (
                filename[: max_filename_length - len(hashed_filename) - 8]
                + "..."
                + hashed_filename
                + ".lock"
            )
            return os.path.join(dirname, new_filename)
        else:
            return path
