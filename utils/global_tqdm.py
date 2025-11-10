from tqdm import  tqdm as old_tqdm
from CLTrainingFramework.utils import framework_config

ENABLE_TQDM = framework_config.ENABLE_TQDM

def are_tqdm_enabled():
    global ENABLE_TQDM
    return ENABLE_TQDM

def enable_tqdm():
    global ENABLE_TQDM
    ENABLE_TQDM = True

def disable_tqdm():
    global ENABLE_TQDM
    ENABLE_TQDM = False


class tqdm(old_tqdm):
    """
    Global tqdm copy from huggingface

    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        if not are_tqdm_enabled():
            kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/datasets/issues/6066"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise
