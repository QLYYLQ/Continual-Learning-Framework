import copy
import json
from dataclasses import fields
from functools import wraps
from typing import Union, Any, Optional, Callable

import pyarrow as pa

from CLTrainingFramework.dataset.arrow_utils import (
    arrow_type_to_framework_string_dtype,
    Array2DExtensionType,
    Array5DExtensionType,
    Array4DExtensionType,
    Array3DExtensionType,
    _ArrayXDExtensionType,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
)
from CLTrainingFramework.dataset.schema._type import SchemaType, _SCHEMA_TYPES
from CLTrainingFramework.dataset.schema.load_file import load_from_storage_with_nested_sample
from CLTrainingFramework.dataset.schema.supported_schema_type import Value, Sequence, LargeSequence
from CLTrainingFramework.dataset.schema.wirte_file import prepare_for_pa_cache, prepare_nested_sample_to_pa_cache
from CLTrainingFramework.dataset.utils.py_utils_mine import as_dict, zip_dict
from CLTrainingFramework.utils.naming import camelcase_to_snakecase, snakecase_to_camelcase

# Register the extension types for deserialization
pa.register_extension_type(Array2DExtensionType((1, 2), "int64"))
pa.register_extension_type(Array3DExtensionType((1, 2, 3), "int64"))
pa.register_extension_type(Array4DExtensionType((1, 2, 3, 4), "int64"))
pa.register_extension_type(Array5DExtensionType((1, 2, 3, 4, 5), "int64"))


def _from_dict_helper(obj: Any):
    """
    We use the '_schema' fields to get the dataclass name to load.

    Recursive helper for Schema.from_dict, and allows for a convenient constructor syntax
    to define schemas from deserialized JSON dictionaries. This function is used in particular when deserializing
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(obj, list):
        return [_from_dict_helper(value) for value in obj]
    # Otherwise we have a dict or a dataclass
    if "_schema" not in obj or isinstance(obj["_schema"], dict):
        return {key: _from_dict_helper(value) for key, value in obj.items()}
    obj = dict(obj)
    _type = obj.pop("_schema")
    class_type: Optional[SchemaType] = _SCHEMA_TYPES.get(_type, None) or globals().get(
        _type, None
    )

    if class_type is None:
        raise ValueError(
            f"Schema type '{_type}' not found. Available Schema types: {list(_SCHEMA_TYPES.keys())}"
        )

    if class_type == LargeSequence:
        feature = obj.pop("schema")
        return LargeSequence(schema=_from_dict_helper(feature), **obj)
    if class_type == Sequence:
        feature = obj.pop("schema")
        return Sequence(schema=_from_dict_helper(feature), **obj)

    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})


def schema_to_pyarrow(schema: SchemaType) -> pa.DataType:
    """
    dataset.schema -> pa.struct
    """
    if isinstance(schema, (Schema, dict)):
        # since python 3.7, dict has a deterministic order, and order matters in pa.struct
        return pa.struct({k: schema_to_pyarrow(schema[k]) for k in schema})
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError("Only one example of the inner type")
        value = schema_to_pyarrow(schema[0])
        return pa.list_(value)
    elif isinstance(schema,LargeSequence):
        value = schema_to_pyarrow(schema.schema)
        return pa.large_list(value)
    elif isinstance(schema,Sequence):
        value = schema_to_pyarrow(schema.schema)
        if isinstance(schema.schema,dict):
            data_type = pa.struct({f.name:pa.list_(f.type,schema.length) for f in value})
        else:
            data_type = pa.list_(value,schema.length)
        return data_type
    return schema()


def pyarrow_to_schema(pa_type: pa.DataType) -> SchemaType:
    """
    change pyarrow datatype to dataset.schema
    """
    if isinstance(pa_type, pa.StructType):
        return {field.name: pyarrow_to_schema(field.type) for field in pa_type}
    elif isinstance(pa_type, pa.FixedSizeListType):
        return Sequence(
            schema=pyarrow_to_schema(pa_type.value_type),
            length=pa_type.list_size,
        )
    elif isinstance(pa_type, pa.ListType):
        _schema = pyarrow_to_schema(pa_type.value_type)
        if isinstance(_schema, (dict, tuple, list)):
            return [_schema]
        return Sequence(schema=_schema)
    elif isinstance(pa_type, pa.LargeListType):
        _schema = pyarrow_to_schema(pa_type.value_type)
        return LargeSequence(schema=_schema)
    elif isinstance(pa_type, _ArrayXDExtensionType):
        array_feature = [None, None, Array2D, Array3D, Array4D, Array5D][pa_type.ndims]
        return array_feature(shape=pa_type.shape, dtype=pa_type.value_type)
    elif isinstance(pa_type, pa.DataType):
        return Value(dtype=arrow_type_to_framework_string_dtype(pa_type))
    else:
        raise ValueError(f"Cannot convert {pa_type} to a Schema type.")


def _keep_schema_dict_synced(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Schema" = args[0]
            args = args[1:]
        else:
            # the reason why Schema doesn't have self
            self: "Schema" = kwargs.pop("self")
        out = func(self, *args, **kwargs)
        assert hasattr(self, "_required_unpack_column")
        self._required_unpack_column = {k: require_unpacking(v) for k, v in self.items()}
        return out

    wrapper._decorator_name_ = "_keep_schema_dict_synced"
    return wrapper


def require_unpacking(schema: SchemaType, forced_decode: bool = False):
    if isinstance(schema, dict):
        return any(require_unpacking(f) for f in schema.values())
    elif isinstance(schema, (list, tuple)):
        return require_unpacking(schema[0], forced_decode=forced_decode)
    elif isinstance(schema, (LargeSequence, Sequence)):
        return require_unpacking(schema.schema, forced_decode=forced_decode)
    else:
        return hasattr(schema, "unpack_example") and (
            getattr(schema, "unpack", True) if not forced_decode else forced_decode
        )


class Schema(dict):
    """
    Schema has many field type:
        - [`Value`] feature specifies a single data type value, e.g. `int64` or `string`.
        - [`ClassLabel`] feature specifies a predefined set of classes which can have labels associated to them and will be stored as integers in the dataset.
        - Python `dict` specifies a composite feature containing a mapping of sub-fields to sub-features. It's possible to have nested fields of nested fields in an arbitrary manner.
        - Python `list`, [`LargeList`] or [`Sequence`] specifies a composite feature containing a sequence of sub-features, all of the same feature type.

        <Tip>
            A [`Sequence`] with an internal dictionary feature will be automatically converted into a dictionary of lists. This behavior is implemented to have a compatibility layer with the TensorFlow Datasets library but may be un-wanted in some cases. If you don't want this behavior, you can use a Python `list` or a [`LargeList`] instead of the [`Sequence`].
        </Tip>

        - [`Array2D`], [`Array3D`], [`Array4D`] or [`Array5D`] feature for multidimensional arrays.
        - [`Audio`] feature to store the absolute path to an audio file or a dictionary with the relative path to an audio file ("path" key) and its bytes content ("bytes" key). This feature extracts the audio data.
        - [`Image`] feature to store the absolute path to an image file, an `np.ndarray` object, a `PIL.Image.Image` object or a dictionary with the relative path to an image file ("path" key) and its bytes content ("bytes" key). This feature extracts the image data.
        - [`Translation`] or [`TranslationVariableLanguages`] feature specific to Machine Translation.
    """

    def __init__(self, *args, **kwargs):
        # if not args:
        #     raise TypeError("Schema requires at least one argument")
        # # Schema(data_dict, foo="bar"), then we have __init__(<object>, data_dict, foo="bar"), it means that we can have
        # # Schema(...,foo="xxx", self=...) without TypeError for some meta programming
        # self, *args = args
        super(Schema, self).__init__(*args, **kwargs)
        self._required_unpack_column: dict[str, bool] = {
            k: require_unpacking(v) for k, v in self.items()
        }
        assert hasattr(self, "_required_unpack_column")

    # rewire dict method with dict_synced
    __delitem__ = _keep_schema_dict_synced(dict.__delitem__)
    __setitem__ = _keep_schema_dict_synced(dict.__setitem__)
    update = _keep_schema_dict_synced(dict.update)
    setdefault = _keep_schema_dict_synced(dict.setdefault)
    pop = _keep_schema_dict_synced(dict.pop)
    popitem = _keep_schema_dict_synced(dict.popitem)
    clear = _keep_schema_dict_synced(dict.clear)

    def __reduce__(self):
        # for pickle: a = pickle.dumps(schema);b=pickle.loads(a)
        return Schema, (dict(self),)

    @property
    def type(self):
        return schema_to_pyarrow(self)

    def to_arrow_schema(self) -> pa.schema:
        metadata = {"info": {"schema": self.to_dict()}}
        return pa.schema(self.type).with_metadata({"CLTrainingFramework": json.dumps(metadata)})

    @classmethod
    def from_arrow_schema(cls, arrow_schema: pa.Schema) -> "Schema":
        """
        create Schema from pyarrow schema
        Also allow user download datasets from huggingface, this method can load those datasets can create a Schema object.
        """
        mine_schema = Schema()
        # for huggingface dataset
        if (
                arrow_schema.metadata is not None
                and b"huggingface" in arrow_schema.metadata
        ):
            metadata = json.loads(arrow_schema.metadata[b"huggingface"].decode())
            if (
                    "info" in metadata
                    and "features" in metadata["info"]
                    and metadata["info"]["features"] is not None
            ):
                mine_schema = Schema.from_dict(metadata["info"]["features"])
        elif (
                arrow_schema.metadata is not None
                and b"CLTrainingFramework" in arrow_schema.metadata
        ):
            metadata = json.loads(arrow_schema.metadata[b"CLTrainingFramework"].decode())
            if (
                    "info" in metadata
                    and "schema" in metadata["info"]
                    and metadata["info"]["schema"] is not None
            ):
                mine_schema = Schema.from_dict(metadata["info"]["schema"])
        metadata_schema = mine_schema.to_arrow_schema()
        obj = {
            field.name: (
                mine_schema[field.name]
                if field.name in mine_schema
                   and metadata_schema.field(field.name) == field
                else pyarrow_to_schema(field.type)
            )
            for field in arrow_schema
        }
        return cls(**obj)

    def to_dict(self) -> dict:
        # lower name, Sequence -> sequence, can be seen from https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
        return as_dict(self)

    @classmethod
    def from_dict(cls, dic) -> "Schema":
        obj = _from_dict_helper(dic)
        return cls(**obj)

    def to_yaml_list(self) -> list:
        yaml_data = self.to_dict()

        def simplify(_schema: dict) -> dict:
            if not isinstance(_schema, dict):
                raise TypeError(f"Expected a dict but got a {type(_schema)}: {_schema}")

            for list_type in ["large_sequence", "list", "sequence"]:
                #
                # list_type:                ->              list_type: int32
                #   dtype: int32            ->
                #
                if isinstance(_schema.get(list_type), dict) and list(
                        _schema[list_type]
                ) == ["dtype"]:
                    _schema[list_type] = _schema[list_type]["dtype"]

                #
                # list_type:                ->              list_type:
                #   struct:                 ->              - name: foo
                #   - name: foo             ->                dtype: int32
                #     dtype: int32          ->
                #
                if isinstance(_schema.get(list_type), dict) and list(
                        _schema[list_type]
                ) == ["struct"]:
                    _schema[list_type] = _schema[list_type]["struct"]

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #   - negative              ->                  '0': negative
            #   - positive              ->                  '1': positive
            #
            if isinstance(_schema.get("class_label"), dict) and isinstance(
                    _schema["class_label"].get("names"), list
            ):
                # server-side requirement: keys must be strings
                _schema["class_label"]["names"] = {
                    str(label_id): label_name
                    for label_id, label_name in enumerate(
                        _schema["class_label"]["names"]
                    )
                }
            return _schema

        def to_yaml_inner(obj: Union[dict, list]) -> dict:
            if isinstance(obj, dict):
                _type = obj.pop("_schema", None)
                if _type == "LargeSequence":
                    _schema = obj.pop("schema")
                    return simplify({"large_list": to_yaml_inner(_schema), **obj})
                elif _type == "Sequence":
                    _schema = obj.pop("schema")
                    return simplify({"sequence": to_yaml_inner(_schema), **obj})
                elif _type == "Value":
                    return obj
                elif _type and not obj:
                    return {"dtype": camelcase_to_snakecase(_type)}
                elif _type:
                    return {"dtype": simplify({camelcase_to_snakecase(_type): obj})}
                else:
                    return {
                        "struct": [
                            {"name": name, **to_yaml_inner(_feature)}
                            for name, _feature in obj.items()
                        ]
                    }
            elif isinstance(obj, list):
                return simplify({"list": simplify(to_yaml_inner(obj[0]))})
            elif isinstance(obj, tuple):
                return to_yaml_inner(list(obj))
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        def to_yaml_types(obj: Union[dict, list]):
            if isinstance(obj, dict):
                return {k: to_yaml_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_yaml_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return to_yaml_types(list(obj))
            else:
                return obj

        # 进入to_yaml_types的一定先是一个list，list中有其他元素
        return to_yaml_types(to_yaml_inner(yaml_data)["struct"])

    @classmethod
    def from_yaml_list(cls, yaml_data: list) -> "Schema":
        yaml_data = copy.deepcopy(yaml_data)

        # we convert the list obtained from YAML data into the dict representation that is used for JSON dump

        def unsimplify(_schema: dict) -> dict:
            if not isinstance(_schema, dict):
                raise TypeError(f"Expected a dict but got a {type(_schema)}: {_schema}")

            for list_type in ["large_sequence", "list", "sequence"]:
                #
                # list_type: int32          ->              list_type:
                #                           ->                dtype: int32
                #
                if isinstance(_schema.get(list_type), str):
                    _schema[list_type] = {"dtype": _schema[list_type]}

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #     '0': negative              ->               - negative
            #     '1': positive              ->               - positive
            #
            if isinstance(_schema.get("class_label"), dict) and isinstance(
                    _schema["class_label"].get("names"), dict
            ):
                label_ids = sorted(_schema["class_label"]["names"], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(
                        range(int(label_ids[-1]) + 1)
                ):
                    raise ValueError(
                        f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing."
                    )
                _schema["class_label"]["names"] = [
                    _schema["class_label"]["names"][label_id] for label_id in label_ids
                ]
            return _schema

        def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
            if isinstance(obj, dict):
                if not obj:
                    return {}
                _type = next(iter(obj))
                if _type == "large_sequence":
                    _schema = unsimplify(obj).pop(_type)
                    return {
                        "schema": from_yaml_inner(_schema),
                        **obj,
                        "_schema": "LargeList",
                    }
                if _type == "sequence":
                    _schema = unsimplify(obj).pop(_type)
                    return {
                        "schema": from_yaml_inner(_schema),
                        **obj,
                        "_schema": "Sequence",
                    }
                if _type == "list":
                    return [from_yaml_inner(unsimplify(obj)[_type])]
                if _type == "struct":
                    return from_yaml_inner(obj["struct"])
                elif _type == "dtype":
                    if isinstance(obj["dtype"], str):
                        # e.g. int32, float64, string, audio, image
                        try:
                            Value(obj["dtype"])
                            return {**obj, "_schema": "Value"}
                        except ValueError:
                            # e.g. Audio, Image, ArrayXD
                            return {"_schema": snakecase_to_camelcase(obj["dtype"])}
                    else:
                        return from_yaml_inner(obj["dtype"])
                else:
                    return {
                        "_schema": snakecase_to_camelcase(_type),
                        **unsimplify(obj)[_type],
                    }
            elif isinstance(obj, list):
                names = [_feature.pop("name") for _feature in obj]
                return {
                    name: from_yaml_inner(_feature)
                    for name, _feature in zip(names, obj)
                }
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        # 这里第一次进入的时候type是list，YAML解析保证list的顺序是不会变的
        return cls.from_dict(from_yaml_inner(yaml_data))

    def copy(self) -> "Schema":
        return copy.deepcopy(self)

    def sample_from_storage(
            self,
            sample: dict,
            token_pre_repo_id: Optional[dict[str, Union[str, bool, None]]] = None,
    ):
        """
        load from storage file, transform them into high level python object
        """
        return {
            k: load_from_storage_with_nested_sample(f, v, token_pre_repo_id)
            if self._required_unpack_column[k]
            else v
            for k, (f, v) in zip_dict(
                {key: value for key, value in self.items() if key in sample}, sample
            )
        }

    def column_from_storage(self, column: list, column_name: str):
        return (
            [
                load_from_storage_with_nested_sample(self[column_name], v) if v is not None else None
                for v in column_name
            ]
            if self._required_unpack_column[column_name]
            else column
        )

    def batch_from_storage(
            self,
            batch: dict,
            token_per_repo_id: Optional[dict[str, Union[str, bool, None]]],
    ):
        loaded_batch = {}
        for k, v in batch.items():
            loaded_batch[k] = (
                [
                    load_from_storage_with_nested_sample(self[k], value, token_per_repo_id)
                    if value is not None
                    else None
                    for value in v
                ]
                if self._required_unpack_column[k]
                else v
            )
        return loaded_batch

    def sample_to_pa_cache(self, sample):
        sample = prepare_for_pa_cache(sample)
        return prepare_nested_sample_to_pa_cache(self, sample)

    def column_to_pa_cache(self, column, column_name: str):
        column = prepare_for_pa_cache(column)
        return [prepare_nested_sample_to_pa_cache(self[column_name], obj, level=1) for obj in column]

    def batch_to_pa_cache(self, batch: dict):
        storage_batch = {}
        if set(batch) != set(self):
            raise ValueError(f"Column mismatch between batch {set(batch)} and schema {set(self)}")
        for k, v in batch.items():
            column = prepare_for_pa_cache(v)
            storage_batch[k] = [prepare_nested_sample_to_pa_cache(self, obj, level=1) for obj in column]
        return storage_batch

    def reorder_fields_as(self, other: "Schema") -> "Schema":
        """
        Reorder Schema fields to match the field order of other [`Schema`].

        The order of the fields is important since it matters for the underlying arrow data.
        Re-ordering the fields allows to make the underlying arrow data type match.

        Args:
            other ([`Schema`]):
                The other [`Schema`] to align with.

        Returns:
            [`Schema`]

        """

        def recursive_reorder(source, target, stack=""):
            stack_position = " at " + stack[1:] if stack else ""
            if isinstance(target, Sequence):
                target = target.schema
                if isinstance(target, dict):
                    target = {k: [v] for k, v in target.items()}
                else:
                    target = [target]
            if isinstance(source, Sequence):
                sequence_kwargs = vars(source).copy()
                source = sequence_kwargs.pop("schema")
                if isinstance(source, dict):
                    source = {k: [v] for k, v in source.items()}
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence(
                        {k: v[0] for k, v in reordered.items()}, **sequence_kwargs
                    )
                else:
                    source = [source]
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence(reordered[0], **sequence_kwargs)
            elif isinstance(source, dict):
                if not isinstance(target, dict):
                    raise ValueError(
                        f"Type mismatch: between {source} and {target}" + stack_position
                    )
                if sorted(source) != sorted(target):
                    message = (
                            f"Keys mismatch: between {source} (source) and {target} (target).\n"
                            f"{source.keys() - target.keys()} are missing from target "
                            f"and {target.keys() - source.keys()} are missing from source"
                            + stack_position
                    )
                    raise ValueError(message)
                return {
                    key: recursive_reorder(source[key], target[key], stack + f".{key}")
                    for key in target
                }
            elif isinstance(source, list):
                if not isinstance(target, list):
                    raise ValueError(
                        f"Type mismatch: between {source} and {target}" + stack_position
                    )
                if len(source) != len(target):
                    raise ValueError(
                        f"Length mismatch: between {source} and {target}"
                        + stack_position
                    )
                return [
                    recursive_reorder(source[i], target[i], stack + ".<list>")
                    for i in range(len(target))
                ]
            elif isinstance(source, LargeSequence):
                if not isinstance(target, LargeSequence):
                    raise ValueError(
                        f"Type mismatch: between {source} and {target}" + stack_position
                    )
                return LargeSequence(
                    recursive_reorder(source.schema, target.schema, stack)
                )
            else:
                return source

        return Schema(recursive_reorder(self, other))

    def flatten(self, max_depth=16) -> "Schema":
        for depth in range(1, max_depth):
            no_change = True
            flattened = self.copy()
            for column_name, subfeature in self.items():
                if isinstance(subfeature, dict):
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.items()})
                    del flattened[column_name]
                elif isinstance(subfeature, Sequence) and isinstance(
                        subfeature.schema, dict
                ):
                    no_change = False
                    flattened.update({
                        f"{column_name}.{k}": Sequence(v)
                        if not isinstance(v, dict)
                        else [v]
                        for k, v in subfeature.schema.items()
                    })
                    del flattened[column_name]
                elif (
                        hasattr(subfeature, "flatten")
                        and subfeature.flatten() != subfeature
                ):
                    no_change = False
                    flattened.update({
                        f"{column_name}.{k}": v
                        for k, v in subfeature.flatten().items()
                    })
                    del flattened[column_name]
            self = flattened
            if no_change:
                break
        return self


def map_nested_schema(schema: "SchemaType", func: Callable[[SchemaType], Optional[SchemaType]]) -> "SchemaType":
    if isinstance(schema, Schema):
        out = func(Schema({k: map_nested_schema(v, func) for k, v in schema.items()}))
    elif isinstance(schema, (list, tuple)):
        out = func([map_nested_schema(schema[0], func)])
    elif isinstance(schema, LargeSequence):
        out = func(LargeSequence(map_nested_schema(schema.schema, func)))
    elif isinstance(schema, dict):
        out = func({k: map_nested_schema(v, func) for k, v in schema.items()})
    elif isinstance(schema, Sequence):
        out = func(Sequence(map_nested_schema(schema.schema, func), length=schema.length))
    else:
        out = func(schema)
    return schema if out is None else out


def check_embed_storage_for_schema(schema:SchemaType)->bool:
    if isinstance(schema,dict):
        return any(check_embed_storage_for_schema(v) for v in schema.values())
    elif isinstance(schema,(list,tuple)):
        return check_embed_storage_for_schema(schema[0])
    elif isinstance(schema,( LargeSequence ,Sequence)):
        return check_embed_storage_for_schema(schema.schema)
    else:
        return hasattr(schema,"embed_local_file_to_pa_cache")
