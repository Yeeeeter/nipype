# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Base interface for nipy"""

from ..base import LibraryBaseInterface
from ...utils.misc import package_check

# Originally set in model, preprocess and utils
# Set here to be imported, in case anybody depends on its presence
# Remove in 2.0
have_bct = True
try:
    package_check("bct")
except:
    have_bct = False

have_numpy = True
try:
    package_check("numpy")
except:
    have_numpy = False

have_scipy = True
try:
    package_check("scipy")
except:
    have_scipy = False

class NipyBaseInterface(LibraryBaseInterface):
    imports = ("bct", "numpy", "scipy")
    _pkg = "bct"