import warnings

from .calibration_curves import intcal20, shcal20, marine20
from .calibrate import calibrate_14C_age

# Monkey-patch the line away from warnings, as it is rather irritating.
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, cat, fname, lineno, line=None: \
    formatwarning_orig(msg, cat, fname, lineno, line='')


__all__ = [
    'intcal20',
    'shcal20',
    'marine20',
    'calibrate_14C_age',
]

packagename = "hamstrpy"
