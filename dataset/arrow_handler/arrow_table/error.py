# Thank you huggingface

from functools import partial

from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import short_str


class CastError(ValueError):
    """When it's not possible to cast an Arrow table to a specific schema or set of features"""

    def __init__(self, *args, table_column_names: list[str], requested_column_names: list[str]) -> None:
        super().__init__(*args)
        self.table_column_names = table_column_names
        self.requested_column_names = requested_column_names

    def __reduce__(self):
        # Fix unpickling: TypeError: __init__() missing 2 required keyword-only arguments: 'table_column_names' and 'requested_column_names'
        return partial(
            CastError, table_column_names=self.table_column_names, requested_column_names=self.requested_column_names
        ), ()

    def details(self):
        new_columns = set(self.table_column_names) - set(self.requested_column_names)
        missing_columns = set(self.requested_column_names) - set(self.table_column_names)
        if new_columns and missing_columns:
            return f"there are {len(new_columns)} new columns ({short_str(new_columns)}) and {len(missing_columns)} missing columns ({short_str(missing_columns)})."
        elif new_columns:
            return f"there are {len(new_columns)} new columns ({short_str(new_columns)})"
        else:
            return f"there are {len(missing_columns)} missing columns ({short_str(missing_columns)})"
