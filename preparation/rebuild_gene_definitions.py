"""Build the HGNC mapping and testable gene-set memberships."""

from argparse import ArgumentParser
from pathlib import Path
from study_settings import DESIGN
import pandas as pd


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument(
        "source",
        type=Path,
        help="HGNC complete set, five named GMT releases and celltype_genes_seidlitz2020.csv",
    )
    p.add_argument("expression", type=Path, help="Six-donor expression.csv")
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    approved = pd.read_csv(
        a.source / "hgnc_complete_set.txt", sep="\t", dtype=str
    ).fillna("")
    approved = approved[approved.status.eq("Approved")]
    symbols = set(approved.symbol.str.upper())
    other = {}
    for row in approved.itertuples():
        for raw in (row.alias_symbol, row.prev_symbol):
            for alias in raw.upper().split("|"):
                if alias and alias not in symbols:
                    other.setdefault(alias, set()).add(row.symbol.upper())

    def convert(name):
        s = name.strip().upper()
        if s in symbols:
            return s, "approved"
        if s in other and len(other[s]) == 1:
            return next(iter(other[s])), "unambiguous_alias"
        return s, "unmapped_or_ambiguous_retained_exact"

    expr = pd.read_csv(a.expression, index_col=0).dropna(how="all")
    expr.columns = expr.columns.str.strip().str.upper()
    if expr.columns.duplicated().any():
        expr = expr.T.groupby(level=0, sort=False).mean().T
    expr = expr.dropna(axis=1)
    features = expr.columns[expr.std(ddof=1) > 0]
    mapping = pd.DataFrame(
        [(f, *convert(f)) for f in features],
        columns=["measured_gene", "canonical_gene", "mapping_status"],
    )
    universe = set(mapping.canonical_gene)
    rows = []

    def add(lib, term, genes, lower, upper):
        found = sorted({convert(g)[0] for g in genes} & universe)
        if lower <= len(found) <= upper:
            rows.append([lib, term, ";".join(found), len(found)])

    for lib in DESIGN["libraries"]:
        for line in (a.source / (lib + ".gmt")).read_text(encoding="utf8").splitlines():
            term, description, *genes = line.split("\t")
            add(lib, term, genes, 10, 500)
    markers = pd.read_csv(a.source / "celltype_genes_seidlitz2020.csv")
    for group, d in markers.groupby("class"):
        add("Cell_types_Seidlitz2020", group, d.gene, 5, 100000)
    for group, genes in DESIGN["disease_sets"].items():
        add("Dystonia_disease_genes", group, genes, 3, 100000)
    sets = (
        pd.DataFrame(
            rows, columns=["collection", "term", "gene_members", "matched_genes"]
        )
        .sort_values(["collection", "term"])
        .reset_index(drop=True)
    )
    for file, frame in [("gene_names.csv.gz", mapping), ("gene_sets.csv.gz", sets)]:
        frame.to_csv(a.output / file, index=False)


if __name__ == "__main__":
    main()
