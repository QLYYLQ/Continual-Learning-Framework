from typing import Any, Mapping

import pandas
import pandas as pd

from CLTrainingFramework.dataset.schema.supported_schema_type import LargeSequence, Sequence
from CLTrainingFramework.dataset.schema.utils import _check_non_null_non_empty_recursive, write_image


def prepare_for_pa_cache(
        obj: Any, keep_dim: bool = True, only_check_first_element: bool = True
) -> Any:
    """
    Args:
        obj:Any
            nested struct object
        keep_dim:bool, default False
            whether to keep the full dimension
            arrow only support converting 1-dimensional array values
        only_check_first_element:bool, default True
            whether to only check the first element in list
    """
    return _column_object_to_python_object(
        obj, keep_dim=keep_dim, only_check_first_element=only_check_first_element
    )


def _column_object_to_python_object(
        obj: Any, keep_dim: bool = True, only_check_first_element: bool = True
) -> Any:
    import torch, PIL.Image
    import numpy as np

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj[()]
        elif keep_dim or obj.ndim == 1:
            return obj
        else:
            return (
                [
                    _column_object_to_python_object(
                        x, keep_dim, only_check_first_element
                    )
                    for x in obj
                ],
            )
    elif isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            # numpy does not support bfloat16, we need modify the type manually
            # https://stackoverflow.com/questions/78128662/converting-pytorch-bfloat16-tensors-to-numpy-throws-typeerror
            return _column_object_to_python_object(
                obj.detach().to(torch.float).cpu().numpy(),
                keep_dim,
                only_check_first_element,
            )
        return _column_object_to_python_object(
            obj.detach().cpu().numpy(), keep_dim, only_check_first_element
        )
    elif isinstance(obj, PIL.Image.Image):
        return write_image(obj)
    elif isinstance(obj, pd.Series):
        return (
            _column_object_to_python_object(obj.tolist(), keep_dim, only_check_first_element)
        )
    elif isinstance(obj, pandas.DataFrame):
        return (
            {
                k: _column_object_to_python_object(v, keep_dim, only_check_first_element)
                for k, v in obj.to_dict("series").items()
            }
        )
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    elif isinstance(obj, pd.Timedelta):
        return obj.to_pytimedelta()
    elif isinstance(obj, Mapping):
        output = {}
        for k, v in obj.items():
            _v = _column_object_to_python_object(v, keep_dim, only_check_first_element)
            output[k] = _v
        return output
    elif hasattr(obj, "__array__"):
        if np.isscalar(obj):
            return obj
        else:
            return (
                _column_object_to_python_object(obj.__array__(), keep_dim, only_check_first_element)
            )
    elif isinstance(obj, (list, tuple)):
        if len(obj) >= 1:
            for element in obj:
                if _check_non_null_non_empty_recursive(element):
                    break
                _element = _column_object_to_python_object(element, keep_dim, only_check_first_element)
                _modified = type(_element) == type(element) if not isinstance(element,
                                                                              Mapping) else _check_is_mapping_element_changed(
                    element, _element)
                if _modified or not only_check_first_element:
                    return(
                        [
                            _column_object_to_python_object(elmt, keep_dim, only_check_first_element)
                            for elmt in obj
                        ]
                    )
                else:
                    return obj
    else:
        return obj


def _check_is_mapping_element_changed(old_element:Mapping, new_element:Mapping):
    is_changed = False
    for _,old_v,_,new_v in zip(old_element.items(),new_element.items()):
        is_changed |= type(old_v) == type(new_v)
    return is_changed


def prepare_nested_sample_to_pa_cache(schema, obj, level=0):
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Get None but expect an dict object")
        return (
            {k: prepare_nested_sample_to_pa_cache(schema[k], obj[k], level + 1) for k in schema}
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return prepare_nested_sample_to_pa_cache(schema, obj.tolist())
        else:
            if len(obj) > 0:
                element = None
                for element in obj:
                    if _check_non_null_non_empty_recursive(element, sub_schema):
                        break
                if prepare_nested_sample_to_pa_cache(sub_schema, element, level + 1) != element:
                    return [prepare_nested_sample_to_pa_cache(sub_schema, o, level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeSequence):
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                sub_schema = schema.schema
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if (
                        prepare_nested_sample_to_pa_cache_with_level(sub_schema, first_elmt, level=level + 1)
                        != first_elmt
                ):
                    return [
                        prepare_nested_sample_to_pa_cache_with_level(sub_schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.schema, dict):
            # dict of list to fill
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                # obj is a list of dict
                for k in schema.schema:
                    list_dict[k] = [
                        prepare_nested_sample_to_pa_cache_with_level(
                            schema.schema[k], o.get(k), level=level + 1
                        )
                        for o in obj
                    ]
                return list_dict
            else:
                # obj is a single dict
                for k in schema.schema:
                    list_dict[k] = (
                        [
                            prepare_nested_sample_to_pa_cache_with_level(schema.schema[k], o, level=level + 1)
                            for o in obj[k]
                        ]
                        if k in obj
                        else None
                    )
                return list_dict
        # schema.feature is not a dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, schema.schema):
                        break
                # be careful when comparing tensors here
                if (
                        not (isinstance(first_elmt, list) or np.isscalar(first_elmt))
                        or prepare_nested_sample_to_pa_cache_with_level(
                    schema.schema, first_elmt, level=level + 1
                )
                        != first_elmt
                ):
                    return [
                        prepare_nested_sample_to_pa_cache_with_level(schema.schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)

    elif hasattr(schema, "sample_to_storage"):
        return schema.sample_to_pa_cache(obj) if obj is not None else None
    return obj


def prepare_nested_sample_to_pa_cache_with_level(schema, obj, level=0):
    """
    Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be encoded.
    If the first element needs to be encoded, then all the elements of the list will be encoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Got None but expected a dictionary instead")
        return (
            {
                k: prepare_nested_sample_to_pa_cache_with_level(schema[k], obj.get(k), level=level + 1)
                for k in schema
            }
            if obj is not None
            else None
        )

    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return prepare_nested_sample_to_pa_cache_with_level(schema, obj.tolist())
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if (
                        prepare_nested_sample_to_pa_cache_with_level(sub_schema, first_elmt, level=level + 1)
                        != first_elmt
                ):
                    return [
                        prepare_nested_sample_to_pa_cache_with_level(sub_schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    elif isinstance(schema, LargeSequence):
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                sub_schema = schema.schema
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if (
                        prepare_nested_sample_to_pa_cache_with_level(sub_schema, first_elmt, level=level + 1)
                        != first_elmt
                ):
                    return [
                        prepare_nested_sample_to_pa_cache_with_level(sub_schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.schema, dict):
            # dict of list to fill
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                # obj is a list of dict
                for k in schema.schema:
                    list_dict[k] = [
                        prepare_nested_sample_to_pa_cache_with_level(
                            schema.schema[k], o.get(k), level=level + 1
                        )
                        for o in obj
                    ]
                return list_dict
            else:
                # obj is a single dict
                for k in schema.schema:
                    list_dict[k] = (
                        [
                            prepare_nested_sample_to_pa_cache_with_level(schema.schema[k], o, level=level + 1)
                            for o in obj[k]
                        ]
                        if k in obj
                        else None
                    )
                return list_dict
        # schema.feature is not a dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, schema.schema):
                        break
                # be careful when comparing tensors here
                if (
                        not (isinstance(first_elmt, list) or np.isscalar(first_elmt))
                        or prepare_nested_sample_to_pa_cache_with_level(
                    schema.schema, first_elmt, level=level + 1
                )
                        != first_elmt
                ):
                    return [
                        prepare_nested_sample_to_pa_cache_with_level(schema.schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    # Object with special encoding:
    # ClassLabel will convert from string to int
    elif hasattr(schema, "sample_to_storage"):
        return schema.sample_to_pa_cache(obj) if obj is not None else None
    # Other object should be directly convertible to a native Arrow type (like Translation and Translation)

    return obj


if __name__ == "__main__":
    import numpy as np

    array1 = np.array([1, 2, 3])
    array2 = np.array([[4, 5, 6], [453, 4e5, 435]])
    array3 = np.array(123)
    a = _column_object_to_python_object(array1)
    b = _column_object_to_python_object(array2)
    c = _column_object_to_python_object(array3)
    print(a,type(a))
    print(b)
    print(c)
    import torch

    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([[4, 5, 6], [34, 643, 23]])
    tensor3 = torch.tensor(123)
    tensor4 = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    a = prepare_for_pa_cache(tensor1)
    b = prepare_for_pa_cache(tensor2)
    c = prepare_for_pa_cache(tensor3)
    d = prepare_for_pa_cache(tensor4)
    print(a,type(a))
    print(b)
    print(c)
    print(d)
