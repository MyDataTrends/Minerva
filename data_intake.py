"""
Compatibility shim for data_intake module.

This module re-exports from Data_Intake to support snake_case imports.
The canonical location is Data_Intake/ but this shim allows:
    from data_intake import ...
"""
import warnings

warnings.warn(
    "The 'data_intake' module name is preferred. "
    "The 'Data_Intake' directory is kept for backward compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

from Data_Intake.datalake_ingestion import *
from Data_Intake import storage_helpers
