"""Compatibility for the abagen 0.1.3 extraction environment."""

import numpy as np
import pandas as pd
from abagen import matching as abagen_matching


def install_abagen_runtime_compatibility() -> dict[str, bool]:
    """Provide removed pandas APIs still called by abagen 0.1.3.

    The adapters preserve the legacy return semantics and are confined to this
    process. They do not modify site-packages or any input data.
    """

    installed = {
        "DataFrame.set_axis_inplace_adapter": False,
        "DataFrame.append_adapter": False,
        "numpy.row_stack_alias": False,
        "AtlasTree._assign_sample_scalar_adapter": False,
    }
    original_set_axis = pd.DataFrame.set_axis
    if "inplace" not in original_set_axis.__code__.co_varnames:

        def legacy_set_axis(self, labels, *, axis=0, inplace=None, copy=None):
            result = original_set_axis(self, labels, axis=axis, copy=copy)
            if inplace:
                self._mgr = result._mgr
                return None
            return result

        pd.DataFrame.set_axis = legacy_set_axis
        installed["DataFrame.set_axis_inplace_adapter"] = True

    if not hasattr(pd.DataFrame, "append"):

        def legacy_append(
            self, other, ignore_index=False, verify_integrity=False, sort=False
        ):
            if isinstance(other, dict):
                if not ignore_index:
                    raise TypeError("Can only append a dict if ignore_index=True")
                other = pd.DataFrame([other])
            elif isinstance(other, pd.Series):
                other = other.to_frame().T
            return pd.concat(
                [self, other],
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
                sort=sort,
            )

        pd.DataFrame.append = legacy_append
        installed["DataFrame.append_adapter"] = True
    if not hasattr(np, "row_stack"):
        np.row_stack = np.vstack
        installed["numpy.row_stack_alias"] = True
    original_assign_sample = abagen_matching.AtlasTree._assign_sample

    def scalar_assign_sample(self, possible, sample):
        result = original_assign_sample(self, possible, sample)
        array = np.asarray(result)
        return array.item() if array.size == 1 else result

    abagen_matching.AtlasTree._assign_sample = scalar_assign_sample
    installed["AtlasTree._assign_sample_scalar_adapter"] = True
    return installed
