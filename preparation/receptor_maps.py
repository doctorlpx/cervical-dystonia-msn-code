"""Project the 25 source images and rebuild the 19 target distributions for each atlas."""

from pathlib import Path
from study_settings import RECEPTOR_SOURCES
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.interpolate import interpn


def standardize(v):
    return (v - v.mean(axis=0)) / v.std(axis=0, ddof=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        type=Path,
        help="Directory with the 25 reference images named in study_settings.py",
    )
    parser.add_argument(
        "registration",
        type=Path,
        help="neuromaps MNI152-to-fsaverage registration-fusion coordinates",
    )
    parser.add_argument("reference", type=Path, help="Prepared reference directory containing atlas labels")
    args = parser.parse_args()
    ref = args.reference
    sources = RECEPTOR_SOURCES
    xyz = np.concatenate(
        [
            np.loadtxt(
                args.registration
                / f"tpl-MNI152_space-fsaverage_den-164k_hemi-{h}_regfusion.txt"
            )
            for h in ["L", "R"]
        ]
    )
    assert xyz.shape == (327684, 3)
    parcels = {}
    tables = {}
    for atlas in ["schaefer400", "schaefer600"]:
        with np.load(ref / atlas / "vertex_labels.npz") as f:
            parcels[atlas] = f["labels"]
        names = (
            (ref / atlas / "regions.txt").read_text(encoding="utf-8-sig").splitlines()
        )
        tables[atlas] = pd.DataFrame(index=names)
    for items in sources.values():
        for filename, n in items:
            img = nib.load(args.images / filename)
            data = np.squeeze(np.asarray(img.dataobj, dtype=float))
            assert data.ndim == 3
            vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)
            vals = interpn(
                tuple(np.arange(k) for k in data.shape),
                data,
                vox,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            for atlas, labels in parcels.items():
                means = []
                for i in range(len(tables[atlas])):
                    v = vals[labels == i]
                    finite = np.isfinite(v)
                    if finite.mean() < 0.8:
                        raise ValueError(
                            f"Insufficient coverage for {filename}, {atlas}, parcel {i}."
                        )
                    means.append(v[finite].mean())
                tables[atlas][filename] = np.asarray(means)
    for atlas, df in tables.items():
        targets = pd.DataFrame(index=df.index)
        for target, items in sources.items():
            weights = np.array([w for name, w in items])
            values = standardize(df[[name for name, w in items]].to_numpy())
            targets[target] = standardize(values @ weights / weights.sum())
        targets.to_csv(ref / atlas / "receptor_targets.csv")


if __name__ == "__main__":
    main()
