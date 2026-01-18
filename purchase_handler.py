"""
Compatibility shim for purchase_handler module.

This module re-exports from Purchase_handler to support snake_case imports.
The canonical location is Purchase_handler/ but this shim allows:
    from purchase_handler import ...
"""
# Note: No deprecation warning here because 'purchase_handler' IS the preferred name

from Purchase_handler.purchase_handler import *
