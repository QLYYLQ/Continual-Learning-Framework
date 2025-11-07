from typing import Optional, Union

from CLTrainingFramework.dataset.schema.supported_schema_type import LargeSequence, Sequence
from CLTrainingFramework.dataset.schema.utils import _check_non_null_non_empty_recursive
from CLTrainingFramework.dataset.utils.py_utils_mine import zip_dict


def load_from_storage_with_nested_sample(
    schema, obj, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None
):
    """
    unpack a nested example.
    This is used since some schema (in particular Audio and Image) have some logic during decoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be unpacked.
    If the first element needs to be unpacked, then all the elements of the list will be unpacked, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        return (
            {
                k: load_from_storage_with_nested_sample(sub_schema, sub_obj)
                for k, (sub_schema, sub_obj) in zip_dict(schema, obj)
            }
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                first_element = None
                for first_element in obj:
                    if _check_non_null_non_empty_recursive(first_element, sub_schema):
                        break
                assert first_element is not None
                if load_from_storage_with_nested_sample(sub_schema, first_element) != first_element:
                    return [load_from_storage_with_nested_sample(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeSequence):
        if obj is None:
            return None
        else:
            sub_schema = schema.schema
            if len(obj) > 0:
                for first_element in obj:
                    if _check_non_null_non_empty_recursive(first_element, sub_schema):
                        break
                if load_from_storage_with_nested_sample(sub_schema, first_element) != first_element:
                    return [load_from_storage_with_nested_sample(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.schema, dict):
            return {
                k: load_from_storage_with_nested_sample([schema.schema[k]], obj[k])
                for k in schema.schema
            }
        else:
            return load_from_storage_with_nested_sample([schema.schema], obj)
    # Object with special decoding:
    elif hasattr(schema, "sample_from_storage") and getattr(schema, "load_from_storage", True):
        # we pass the token to read and decode files from private repositories in streaming mode
        return (
            schema.sample_from_storage(obj, token_per_repo_id=token_per_repo_id)
            if obj is not None
            else None
        )
    return obj
