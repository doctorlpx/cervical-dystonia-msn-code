"""Run the analyses underlying the cervical dystonia manuscript."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Prepared inputs, as described in README.md.",
    )
    parser.add_argument(
        "--reference", type=Path,
        help="Prepared atlas and molecular references (default: INPUTS/reference).",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="An empty directory for this run."
    )
    parser.add_argument(
        "--stage",
        choices=["full", "group-maps", "receptors"],
        default="full",
        help="group-maps uses supplied group contrasts and omits participant-based and bootstrap analyses. receptors runs only the receptor analyses from supplied group contrasts.",
    )
    parser.add_argument(
        "--branch",
        choices=["all", "S400_full60", "S600_full60", "S400_age44"],
        default="all",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    reference = args.reference or args.inputs / "reference"
    if not args.inputs.is_dir() or not reference.is_dir():
        parser.error("Supply the prepared inputs and reference directories.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Please choose an empty output directory.")
    if args.threads < 1:
        parser.error("Set --threads to a positive integer.")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MSN_REFERENCE_DIR"] = str(reference.resolve())
    os.environ["MSN_INPUT_DIR"] = str(args.inputs.resolve())
    os.environ["MSN_OUTPUT_DIR"] = str(args.output.resolve())
    for key in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = str(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
    import common, imaging, transcriptomics, receptors, gene_sets
    from shutil import copyfile

    for b in common.CFG["branches"]:
        if args.branch not in ["all", b["id"]]:
            continue
        print(b["id"], flush=True)
        if args.stage == "full":
            imaging.run(b)
        else:
            copyfile(
                common.INPUT / "group_maps" / (b["id"] + ".csv"),
                common.out("branches/" + b["id"] + "/REGIONAL_GLM.csv"),
            )
        receptors.run(b)
        if args.stage != "receptors":
            transcriptomics.run(b)
            if b["full_downstream"] and args.stage == "full":
                transcriptomics.stability(b)
            gene_sets.run(b, regional_bootstraps=args.stage == "full")
    if args.stage == "full" and args.branch == "all":
        import sensitivity

        sensitivity.run()
    common.dump(
        "run.json", dict(branch=args.branch, stage=args.stage, threads=args.threads)
    )


if __name__ == "__main__":
    main()
