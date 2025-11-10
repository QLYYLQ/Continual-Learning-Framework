import inspect
import random
from functools import wraps
from typing import Callable, Optional, Any, TYPE_CHECKING

import numpy as np

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.fingerprint import Hasher
from CLTrainingFramework.utils import logging
from CLTrainingFramework.utils.naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
if TYPE_CHECKING:
    from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.arrow_dataset import Dataset

logger = logging.get_logger(__name__)
_CACHING_ENABLED = True

fingerprint_rng = random.Random()
fingerprint_warnings: dict[str, bool] = {}


def generate_random_fingerprint(nbits: int = 64) -> str:
    return f"{fingerprint_rng.getrandbits(nbits):0{nbits // 4}x}"


def update_fingerprint(fingerprint, transform, transform_args):
    global fingerprint_warnings
    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:  # noqa various errors might raise here from pickle or dill
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                logger.warning(
                    f"Transform {transform} couldn't be hashed properly, a random hash was used instead. "
                    "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                    "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                    "This warning is only showed once. Subsequent hashing failures won't be showed."
                )
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            else:
                logger.info(f"Transform {transform} couldn't be hashed properly, a random hash was used instead.")
        else:
            logger.info(
                f"Transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
            )

        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:  # noqa various errors might raise here from pickle or dill
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    logger.warning(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. "
                        "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                        "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                        "This warning is only showed once. Subsequent hashing failures won't be showed."
                    )
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
                else:
                    logger.info(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead."
                    )
            else:
                logger.info(
                    f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
                )
            return generate_random_fingerprint()
    return hasher.hexdigest()


def validate_fingerprint(fingerprint: str, max_length=64):
    """
    Make sure the fingerprint is a non-empty string that is not longer that max_length=64 by default,
    so that the fingerprint can be used to name cache files without issues.
    """
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Invalid fingerprint '{fingerprint}': it should be a non-empty string.")
    for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
        if invalid_char in fingerprint:
            raise ValueError(
                f"Invalid fingerprint. Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{fingerprint}'. "
                f"They could create issues when creating cache files."
            )
    if len(fingerprint) > max_length:
        raise ValueError(
            f"Invalid fingerprint. Maximum length is {max_length} but '{fingerprint}' has length {len(fingerprint)}."
            "It could create issues when creating cache files."
        )


def format_transform_for_fingerprint(func: Callable, version: Optional[str] = None) -> str:
    """
    Format a transform to the format that will be used to update the fingerprint.
    """
    transform = f"{func.__module__}.{func.__qualname__}"
    if version is not None:
        transform += f"@{version}"
    return transform


def format_kwargs_for_fingerprint(
        func: Callable,
        args: tuple,
        kwargs: dict[str, Any],
        use_kwargs: Optional[list[str]] = None,
        ignore_kwargs: Optional[list[str]] = None,
        randomized_function: bool = False,
) -> dict[str, Any]:
    """
    Format the kwargs of a transform to the format that will be used to update the fingerprint.
    """
    kwargs_for_fingerprint = kwargs.copy()
    if args:
        params = [p.name for p in inspect.signature(func).parameters.values() if p != p.VAR_KEYWORD]
        args = args[1:]  # assume the first argument is the dataset
        params = params[1:]
        kwargs_for_fingerprint.update(zip(params, args))
    else:
        del kwargs_for_fingerprint[
            next(iter(inspect.signature(func).parameters))
        ]  # assume the first key is the dataset

    # keep the right kwargs to be hashed to generate the fingerprint

    if use_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k in use_kwargs}
    if ignore_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k not in ignore_kwargs}
    if randomized_function:  # randomized functions have `seed` and `generator` parameters
        if kwargs_for_fingerprint.get("seed") is None and kwargs_for_fingerprint.get("generator") is None:
            _, seed, pos, *_ = np.random.get_state()
            seed = seed[pos] if pos < 624 else seed[0]
            kwargs_for_fingerprint["generator"] = np.random.default_rng(seed)

    # remove kwargs that are the default values

    default_values = {
        p.name: p.default for p in inspect.signature(func).parameters.values() if p.default != inspect._empty
    }
    for default_varname, default_value in default_values.items():
        if default_varname in kwargs_for_fingerprint and kwargs_for_fingerprint[default_varname] == default_value:
            kwargs_for_fingerprint.pop(default_varname)
    return kwargs_for_fingerprint


def fingerprint_transform(
        inplace: bool,
        use_kwargs: Optional[list[str]] = None,
        ignore_kwargs: Optional[list[str]] = None,
        fingerprint_names: Optional[list[str]] = None,
        randomized_function: bool = False,
        version: Optional[str] = None,
):
    """
    Wrapper for dataset transforms to update the dataset fingerprint using ``update_fingerprint``
    Args:
        inplace (:obj:`bool`):  If inplace is True, the fingerprint of the dataset is updated inplace.
            Otherwise, a parameter "new_fingerprint" is passed to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset.
        use_kwargs (:obj:`List[str]`, optional): optional white list of argument names to take into account
            to update the fingerprint to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset. By default, all the arguments are used.
        ignore_kwargs (:obj:`List[str]`, optional): optional black list of argument names to take into account
            to update the fingerprint. Note that ignore_kwargs prevails on use_kwargs.
        fingerprint_names (:obj:`List[str]`, optional, defaults to ["new_fingerprint"]):
            If the dataset transforms is not inplace and returns a DatasetDict, then it can require
            several fingerprints (one per dataset in the DatasetDict). By specifying fingerprint_names,
            one fingerprint named after each element of fingerprint_names is going to be passed.
        randomized_function (:obj:`bool`, defaults to False): If the dataset transform is random and has
            optional parameters "seed" and "generator", then you can set randomized_function to True.
            This way, even if users set "seed" and "generator" to None, then the fingerprint is
            going to be randomly generated depending on numpy current state. In this case, the
            generator is set to np.random.default_rng(np.random.get_state()[1][0]).
        version (:obj:`str`, optional): version of the transform. The version is taken into account when
            computing the fingerprint. If a datase transform changes (or at least if the output data
            that are cached changes), then one should increase the version. If the version stays the
            same, then old cached data could be reused that are not compatible with the new transform.
            It should be in the format "MAJOR.MINOR.PATCH".
    """

    if use_kwargs is not None and not isinstance(use_kwargs, list):
        raise ValueError(f"use_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if ignore_kwargs is not None and not isinstance(ignore_kwargs, list):
        raise ValueError(f"ignore_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if inplace and fingerprint_names:
        raise ValueError("fingerprint_names are only used when inplace is False")

    fingerprint_names = fingerprint_names if fingerprint_names is not None else ["new_fingerprint"]

    def _fingerprint(func):
        if not inplace and not all(name in func.__code__.co_varnames for name in fingerprint_names):
            raise ValueError(f"function {func} is missing parameters {fingerprint_names} in signature")

        if randomized_function:  # randomized function have seed and generator parameters
            if "seed" not in func.__code__.co_varnames:
                raise ValueError(f"'seed' must be in {func}'s signature")
            if "generator" not in func.__code__.co_varnames:
                raise ValueError(f"'generator' must be in {func}'s signature")
        # this call has to be outside the wrapper or since __qualname__ changes in multiprocessing
        transform = format_transform_for_fingerprint(func, version=version)

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(
                func,
                args,
                kwargs,
                use_kwargs=use_kwargs,
                ignore_kwargs=ignore_kwargs,
                randomized_function=randomized_function,
            )

            if args:
                dataset: Dataset = args[0]
                args = args[1:]
            else:
                dataset: Dataset = kwargs.pop(next(iter(inspect.signature(func).parameters)))

            # compute new_fingerprint and add it to the args of not in-place transforms
            if inplace:
                new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
            else:
                for fingerprint_name in fingerprint_names:  # transforms like `train_test_split` have several hashes
                    if kwargs.get(fingerprint_name) is None:
                        kwargs_for_fingerprint["fingerprint_name"] = fingerprint_name
                        kwargs[fingerprint_name] = update_fingerprint(
                            dataset._fingerprint, transform, kwargs_for_fingerprint
                        )
                    else:
                        validate_fingerprint(kwargs[fingerprint_name])

            # Call actual function

            out = func(dataset, *args, **kwargs)

            # Update fingerprint of in-place transforms + update in-place history of transforms

            if inplace:  # update after calling func so that the fingerprint doesn't change if the function fails
                dataset._fingerprint = new_fingerprint

            return out

        wrapper._decorator_name_ = "fingerprint"
        return wrapper

    return _fingerprint
