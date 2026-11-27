# Dataset card — Figshare brain tumour dataset

| | |
|---|---|
| **Source** | [figshare.com/articles/dataset/brain_tumor_dataset/1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| **Author** | Jun Cheng |
| **Version used** | Version 8 (2024-12-21) |
| **Modality** | T1-weighted contrast-enhanced MRI, 2-D slices |
| **Size** | 3,064 slices from 233 patients |
| **Format** | MATLAB v7.3 (`.mat`, HDF5) — 4 zip archives of 766 slices + `cvind.mat` |
| **Collection** | Nanfang Hospital (Guangzhou) and Tianjin Medical University General Hospital, China, 2005–2010 |

## Composition

| Class | Label in file | Index here | Slices | Patients |
|---|---|---|---|---|
| Meningioma | 1 | 0 | 708 (23.1 %) | 82 |
| Glioma | 2 | 1 | 1,426 (46.5 %) | 91 |
| Pituitary tumour | 3 | 2 | 930 (30.4 %) | 60 |
| **Total** | | | **3,064** | **233** |

Each file holds the image (int16, mostly 512 × 512, 15 slices 256 × 256, intensities up to
~12,700), the label, the patient ID, the tumour border points and a binary tumour mask.
Slices were acquired in axial, sagittal and coronal planes.

## Known characteristics and limitations

- **Several slices per patient** (~13 on average): a random image split leaks patients.
  This project splits by patient.
- **Class imbalance**: glioma is twice as frequent as meningioma; macro-F1 and class weights
  are used.
- **Single sequence and two centres**: no T2/FLAIR, limited scanner diversity — models may
  not generalise to other hospitals or protocols.
- **Selected slices**: slices containing large tumour sections were chosen; there is no
  "no tumour" class, so models cannot detect the absence of a tumour.
- **Mixed planes**: the same network must handle axial, sagittal and coronal views.

## How this project uses it

The data is **not redistributed** in this repository: `dvc repro` downloads it from
Figshare (checksums verified) or `dvc pull` fetches the team's copy from the private DVC
remote. Masks are used only to check Grad-CAM, never as training input.

## Citation

Please cite both papers when using the dataset:

```bibtex
@article{cheng2015enhanced,
  title   = {Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition},
  author  = {Cheng, Jun and others},
  journal = {PLoS ONE},
  volume  = {10},
  number  = {10},
  pages   = {e0140381},
  year    = {2015}
}

@article{cheng2016retrieval,
  title   = {Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation},
  author  = {Cheng, Jun and others},
  journal = {PLoS ONE},
  volume  = {11},
  number  = {6},
  pages   = {e0157112},
  year    = {2016}
}
```

The licence of the data is the one stated on the Figshare page; check it before any reuse.
