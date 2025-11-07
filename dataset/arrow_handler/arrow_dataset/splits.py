# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Splits related API."""

import dataclasses
from dataclasses import dataclass

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_splits import SplitInfo, NamedSplit, \
    SplitReadInstruction


# 2 requirements:
# 1. datasets.percent be sliceable
# 2. datasets.percent be documented
#
# Instances are not documented, so we want datasets.percent to be a class, but to
# have it be sliceable, we need this metaclass.
class PercentSliceMeta(type):
    def __getitem__(cls, slice_value):
        if not isinstance(slice_value, slice):
            raise ValueError(f"datasets.percent should only be called with slice, not {slice_value}")
        return slice_value


class PercentSlice(metaclass=PercentSliceMeta):
    # pylint: disable=line-too-long
    """Syntactic sugar for defining slice subsplits: `datasets.percent[75:-5]`.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.
    """

    # pylint: enable=line-too-long
    pass


percent = PercentSlice  # pylint: disable=invalid-name


class NamedSplitAll(NamedSplit):
    """Split corresponding to the union of all defined dataset splits."""

    def __init__(self):
        super().__init__("all")

    def __repr__(self):
        return "NamedSplitAll()"

    def get_read_instruction(self, split_dict):
        # Merge all dataset split together
        read_instructions = [SplitReadInstruction(s) for s in split_dict.values()]
        return sum(read_instructions, SplitReadInstruction())


class Split:
    # pylint: disable=line-too-long
    """`Enum` for dataset splits.

    Datasets are typically split into different subsets to be used at various
    stages of training and evaluation.

    - `TRAIN`: the training data.
    - `VALIDATION`: the validation data. If present, this is typically used as
      evaluation data while iterating on a model (e.g. changing hyperparameters,
      model architecture, etc.).
    - `TEST`: the testing data. This is the data to report metrics on. Typically
      you do not want to use this during model iteration as you may overfit to it.
    - `ALL`: the union of all defined dataset splits.

    All splits, including compositions inherit from `datasets.SplitBase`.


    """

    # pylint: enable=line-too-long
    TRAIN = NamedSplit("train")
    TEST = NamedSplit("test")
    VALIDATION = NamedSplit("validation")
    ALL = NamedSplitAll()

    def __new__(cls, name):
        """Create a custom split with datasets.Split('custom_name')."""
        return NamedSplitAll() if name == "all" else NamedSplit(name)


# Similar to SplitInfo, but contain an additional slice info


@dataclass
class SplitGenerator:
    """Defines the split information for the generator.

    This should be used as returned value of
    `GeneratorBasedBuilder._split_generators`.
    See `GeneratorBasedBuilder._split_generators` for more info and example
    of usage.

    Args:
        name (`str`):
            Name of the `Split` for which the generator will
            create the examples.
        **gen_kwargs (additional keyword arguments):
            Keyword arguments to forward to the `DatasetBuilder._generate_examples` method
            of the builder.

    """

    name: str
    gen_kwargs: dict = dataclasses.field(default_factory=dict)
    split_info: SplitInfo = dataclasses.field(init=False)

    def __post_init__(self):
        self.name = str(self.name)  # Make sure we convert NamedSplits in strings
        NamedSplit(self.name)  # check that it's a valid split name
        self.split_info = SplitInfo(name=self.name)
