"""Summarize enrichment terms and network edges from local Metascape exports."""

import argparse
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("positive", type=Path)
    p.add_argument("negative", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    terms = []
    for side, path in [("negative", a.negative), ("positive", a.positive)]:
        with ZipFile(path) as archive:
            with archive.open("Enrichment_GO/_FINAL_GO.csv") as f:
                df = pd.read_csv(f)
            df["platform_q"] = 10.0 ** df["Log(q-value)"]
            selected = df[
                df.FirstInGroupByLogP.eq(1) & df.platform_q.lt(0.05)
            ].sort_values(["LogP", "GO"])
            if side == "negative":
                selected = selected.head(12)
            terms.append(selected.assign(direction=side))
            if side == "negative":
                graph = ET.fromstring(archive.read("Enrichment_GO/GONetwork.xgmml"))
                groups = set(selected.GROUP_ID)
    table = pd.concat(terms, ignore_index=True)
    table[
        [
            "direction",
            "GO",
            "Description",
            "platform_q",
            "#GeneInGOAndHitList",
            "GROUP_ID",
        ]
    ].to_csv(a.output / "terms.csv", index=False)
    nodes = []
    edges = []
    for n in graph.findall("{*}node"):
        fields = {t.get("name"): t.get("value") for t in n.findall("{*}att")}
        if int(fields["GROUP_ID"]) in groups:
            nodes.append((n.get("id"), n.get("label"), int(fields["GROUP_ID"])))
    ids = {row[0] for row in nodes}
    for e in graph.findall("{*}edge"):
        s, t = e.get("source"), e.get("target")
        if s != t and s in ids and t in ids:
            score = next(
                float(att.get("value"))
                for att in e.findall("{*}att")
                if att.get("name") == "SCORE"
            )
            if score > 0.3:
                edges.append((s, t, score))
    pd.DataFrame(nodes, columns=["node_id", "term_id", "GROUP_ID"]).to_csv(
        a.output / "nodes.csv", index=False
    )
    pd.DataFrame(edges, columns=["source", "target", "kappa"]).to_csv(
        a.output / "edges.csv", index=False
    )


if __name__ == "__main__":
    main()
