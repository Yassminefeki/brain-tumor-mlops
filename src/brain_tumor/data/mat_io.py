"""Reading the Figshare ``.mat`` files.

The files are MATLAB v7.3 (HDF5) and each stores a struct ``cjdata`` with:

* ``image``       int16 MRI slice (512x512, a few 256x256)
* ``label``       1 = meningioma, 2 = glioma, 3 = pituitary tumour
* ``PID``         patient identifier, stored as character codes
* ``tumorBorder`` coordinates of points on the tumour border
* ``tumorMask``   binary tumour mask, same size as the image

HDF5 stores MATLAB arrays column-major, so arrays read with h5py are transposed;
:func:`read_mat` transposes them back to the standard viewing orientation.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class MatRecord:
    image: np.ndarray  # (H, W) original intensities
    label_raw: int  # 1, 2 or 3 as stored in the file
    patient_id: str
    mask: np.ndarray | None  # (H, W) uint8 in {0, 1}

    @property
    def label(self) -> int:
        """Zero-based class index (0 = meningioma, 1 = glioma, 2 = pituitary)."""
        return self.label_raw - 1


def decode_pid(codes: np.ndarray) -> str:
    """Patient IDs are stored as an array of character codes."""
    return "".join(chr(int(c)) for c in np.asarray(codes).flatten() if int(c) != 0).strip()


def read_mat(source: str | os.PathLike | bytes, load_mask: bool = True) -> MatRecord:
    """Read one ``.mat`` file from a path or raw bytes."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
    with h5py.File(handle, "r") as f:
        if "cjdata" not in f:
            raise ValueError("Not a Figshare brain-tumour file: missing 'cjdata' struct")
        d = f["cjdata"]
        image = np.array(d["image"]).T
        label_raw = int(np.array(d["label"]).item())
        patient_id = decode_pid(np.array(d["PID"]))
        mask = None
        if load_mask and "tumorMask" in d:
            mask = (np.array(d["tumorMask"]).T > 0).astype(np.uint8)
    if label_raw not in (1, 2, 3):
        raise ValueError(f"Unexpected label {label_raw}")
    return MatRecord(image=image, label_raw=label_raw, patient_id=patient_id, mask=mask)
