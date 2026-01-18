"""
Compatibility shim for integration module.

This module re-exports from Integration to support snake_case imports.
The canonical location is Integration/ but this shim allows:
    from integration import ...
"""
# Note: No deprecation warning here because 'integration' IS the preferred name
# The Integration/ directory should eventually be renamed

from Integration.semantic_merge import *
from Integration.semantic_integration import *
from Integration.data_integration import *
