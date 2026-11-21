"""Build the six-donor Schaefer expression matrices from the Allen normalized microarray archives."""

import argparse
from pathlib import Path
import warnings
from study_settings import DESIGN
import numpy as np
import pandas as pd
import abagen
from abagen_compat import install_abagen_runtime_compatibility


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "donor_root",
        type=Path,
        help="Directory containing six normalized_microarray_donor<ID> folders",
    )
    ap.add_argument(
        "cat_atlases", type=Path, help="CAT12 templates_MNI152NLin2009cAsym directory"
    )
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--atlas", choices=["schaefer400", "schaefer600", "both"], default="both"
    )
    ap.add_argument("--reference", type=Path, help="Default: OUTPUT/reference")
    arg = ap.parse_args()
    install_abagen_runtime_compatibility()
    ref = arg.reference or arg.output / "reference"
    donors = DESIGN["donors"]
    raw_names = dict(
        microarray="MicroarrayExpression.csv",
        ontology="Ontology.csv",
        pacall="PACall.csv",
        probes="Probes.csv",
        annotation="SampleAnnot.csv",
    )
    files = {
        d: {
            k: str(arg.donor_root / f"normalized_microarray_donor{d}" / v)
            for k, v in raw_names.items()
        }
        for d in donors
    }
    for d in files.values():
        for path in d.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    def local_files(*args, **kwargs):
        return {
            d: files[d].copy()
            for d in abagen.datasets.check_donors(kwargs.get("donors", donors))
        }

    abagen.datasets.fetch_microarray = local_files
    abagen.io.use_parq = False
    for n in [400, 600]:
        atlas = f"schaefer{n}"
        if arg.atlas not in ["both", atlas]:
            continue
        dest = arg.output / "expression" / atlas
        dest.mkdir(parents=True, exist_ok=True)
        labels = (
            (ref / atlas / "regions.txt").read_text(encoding="utf-8-sig").splitlines()
        )
        volume = arg.cat_atlases / f"Schaefer2018_{n}Parcels_17Networks_order.nii"
        lookup = pd.read_csv(volume.with_suffix(".csv"), sep=";")
        assert lookup.ROIname.tolist() == [s[3:] for s in labels]
        info = pd.DataFrame(
            dict(
                id=np.arange(1, n + 1),
                label=labels,
                hemisphere=["L"] * (n // 2) + ["R"] * (n // 2),
                structure="cortex",
            )
        )
        expr, counts, report = abagen.get_expression_data(
            str(volume),
            atlas_info=info,
            donors=donors,
            ibf_threshold=0.5,
            probe_selection="diff_stability",
            donor_probes="aggregate",
            lr_mirror=None,
            missing=None,
            tolerance=2,
            sample_norm="srs",
            gene_norm="srs",
            norm_matched=True,
            norm_structures=False,
            region_agg="donors",
            agg_metric="mean",
            corrected_mni=True,
            reannotated=True,
            return_counts=True,
            return_report=True,
            return_donors=True,
            n_proc=1,
            verbose=1,
            data_dir=str(arg.output / "abagen_cache"),
        )
        if not isinstance(expr, dict):
            expr = dict(zip(donors, expr))
        by_id = info.set_index("id").label.to_dict()
        left = labels[: n // 2]
        frames = []
        for d, frame in expr.items():
            frame.index = [by_id[int(k)] for k in frame.index]
            frame = frame.reindex(left)
            frames.append(frame.to_numpy(float))
            frame.to_csv(dest / f"donor_{d}.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            aggregate = np.nanmean(frames, axis=0)
        pd.DataFrame(aggregate, index=left, columns=frame.columns).to_csv(
            dest / "expression.csv"
        )
        counts.index = [by_id[int(k)] for k in counts.index]
        counts.reindex(left).to_csv(dest / "sample_counts.csv")
        (dest / "abagen_report.txt").write_text(report, encoding="utf8")


if __name__ == "__main__":
    main()
