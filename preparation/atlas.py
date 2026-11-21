"""Build parcel labels, seven-network assignments and spatial-null indices locally."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.spatial import cKDTree


def geometry(cat, seven, n):
    """Match surface labels and sphere coordinates to CAT12's volume lookup order."""
    basename = f"Schaefer2018_{n}Parcels_17Networks_order"
    lookup = cat / "templates_MNI152NLin2009cAsym" / (basename + ".csv")
    raw_names = pd.read_csv(lookup, sep=";").ROIname.tolist()
    if len(raw_names) != n or len(set(raw_names)) != n:
        raise ValueError(f"Invalid atlas lookup: {lookup}")

    names, labels, centers, hemispheres, systems = [], [], [], [], []
    for side, hemi in enumerate(("lh", "rh")):
        ids, _, entries = nib.freesurfer.read_annot(cat / "atlases_surfaces" / f"{hemi}.{basename}.annot")
        entries = [entry.decode() for entry in entries]
        ids7, _, entries7 = nib.freesurfer.read_annot(seven / f"{hemi}.Schaefer2018_{n}Parcels_7Networks_order.annot")
        sphere = nib.load(cat / "templates_surfaces" / f"{hemi}.sphere.freesurfer.gii").agg_data("pointset")
        if len(ids) != len(sphere) or len(ids7) != len(ids):
            raise ValueError("Annotations and sphere must share fsaverage vertex order.")
        tree = cKDTree(sphere)
        mapped = np.full(len(ids), -1, dtype=np.int16)
        for index in range(side * n // 2, (side + 1) * n // 2):
            name = raw_names[index]
            if not name.startswith(f"17Networks_{hemi.upper()}_"):
                raise ValueError(f"Unexpected hemisphere for {name}")
            mask = ids == entries.index(name)
            if not mask.any():
                raise ValueError(f"Missing parcel: {name}")
            mapped[mask] = index
            center = tree.query(sphere[mask].mean(axis=0))[1]
            centers.append(sphere[center])
            values, counts = np.unique(ids7[mask], return_counts=True)
            winner = counts.argmax()
            if values[winner] < 0 or counts[winner] / mask.sum() <= 0.99999:
                raise ValueError(f"The two annotations disagree for {name}")
            systems.append(entries7[values[winner]].decode().split("_")[2])
            names.append(f"{hemi}_{name}")
            hemispheres.append(side)
        labels.append(mapped)
    return names, np.concatenate(labels), systems, np.stack(centers), np.array(hemispheres)


def rotations(coords, hemispheres, left_rotations=5000, bilateral_rotations=10000):
    from neuromaps.nulls.spins import gen_spinsamples

    left = hemispheres == 0
    def bank(points, sides, count, method, seed):
        indices = gen_spinsamples(points, sides, n_rotate=count, method=method,
                                  seed=seed, check_duplicates=True)
        return indices.astype(np.int16)

    return dict(
        vasa=bank(coords[left], hemispheres[left], left_rotations, "vasa", 1234),
        alternate=bank(coords[left], hemispheres[left], left_rotations, "original", 20261018),
        pet=bank(coords, hemispheres, bilateral_rotations, "vasa", 20261018),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cat12", type=Path, help="CAT12 root with surface and volume templates")
    parser.add_argument("seven", type=Path, help="Official Schaefer fsaverage 7-network annotation directory")
    parser.add_argument("reference", type=Path, help="Destination, usually prepared/reference")
    parser.add_argument("--atlas", choices=["both", "400", "600"], default="both")
    args = parser.parse_args()
    for n in (400, 600) if args.atlas == "both" else (int(args.atlas),):
        names, labels, systems, centers, sides = geometry(args.cat12, args.seven, n)
        directory = args.reference / f"schaefer{n}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "regions.txt").write_text("\n".join(names) + "\n", encoding="utf8")
        pd.DataFrame({"region": names, "network7": systems}).to_csv(directory / "networks.csv", index=False)
        np.savez_compressed(directory / "vertex_labels.npz", labels=labels)
        savemat(directory / "matlab_labels.mat", {"labels": labels}, do_compression=True)
        np.savez_compressed(directory / "rotations.npz", **rotations(centers, sides))

if __name__ == "__main__":
    main()

