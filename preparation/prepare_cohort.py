"""Read the public participant file and the CAT12 reports for the submitted imaging cohort."""

from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
from study_settings import COHORT_SELECTION
import re
import xml.etree.ElementTree as ET
import pandas as pd


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("participants", type=Path)
    parser.add_argument("cat_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = COHORT_SELECTION
    frame = pd.read_csv(args.participants, sep="\t")
    frame["group"] = frame.group.str.strip()
    frame = frame[frame.group.isin(["CD", "HC"])].copy()
    selected = ~frame.participant_id.map(lambda s: sha256(s.encode()).hexdigest()).isin(
        spec["excluded_public_id_digests"]
    )
    frame = frame[selected].sort_values("participant_id").reset_index(drop=True)
    frame["sex"] = frame.sex.str.strip().str.upper()
    assert (
        sha256(("\n".join(frame.participant_id) + "\n").encode()).hexdigest()
        == spec["ordered_sample_sha256"]
    )
    assert frame.group.value_counts().to_dict() == {"CD": 24, "HC": 36}
    for i, row in frame.iterrows():
        folder = args.cat_root / row.participant_id / "anat"
        root = ET.parse(folder / "cat_T1.xml").getroot()
        entries = [
            (el.text or "").strip() for el in root.findall(".//subjectmeasures/vol_TIV")
        ]
        tiv = next(
            float(value)
            for value in entries
            if re.fullmatch(r"[+\-]?[\d.]+(?:[eE][+\-]?\d+)?", value)
        )
        report = (folder / "catlog_T1.txt").read_text(encoding="utf8", errors="replace")
        iqr = re.search(r"Image Quality Rating \(IQR\):\s*([0-9.]+)%", report)
        if iqr is None:
            raise ValueError(f"Missing IQR in {folder}")
        frame.loc[i, ["TIV", "IQR_percent"]] = [tiv, float(iqr.group(1))]
    args.output.mkdir(exist_ok=True, parents=True)
    frame[
        [
            "participant_id",
            "group",
            "sex",
            "age",
            "TIV",
            "IQR_percent",
            "Burke-Fahn_scale",
        ]
    ].to_csv(args.output / "cohort.csv", index=False)


if __name__ == "__main__":
    main()
