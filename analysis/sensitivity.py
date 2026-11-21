"""Compare effect maps and gene weights across the three fitted analyses."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CFG, REF, ROOT, cohort, csv


def paired_labels(regional):
    """Summarize contralateral parcels sharing a corrected parcel's atlas label."""
    rows = []
    for label in regional.index[regional.q_ols < 0.05]:
        side, _, _, family = label.rsplit("_", 1)[0].split("_", 3)
        other = {"rh": "lh", "lh": "rh"}[side]
        prefix = f"{other}_17Networks_{other.upper()}_{family}_"
        matched = regional.loc[regional.index.str.startswith(prefix)]
        rows.append(dict(parcel=label, family=family, n=len(matched),
                         negative=int((matched.beta < 0).sum()),
                         positive=int((matched.beta > 0).sum()), minimum_q=matched.q_ols.min()))
    return pd.DataFrame(rows, columns=["parcel", "family", "n", "negative", "positive", "minimum_q"])


def run():
    branches = CFG["branches"]
    primary, alternate, age = branches

    def table(branch, name, index):
        frame = pd.read_csv(ROOT / "branches" / branch["id"] / (name + ".csv"))
        return frame.set_index(index, verify_integrity=True)

    def compare(name, first, second):
        return dict(comparison=name, n=len(first), rho=spearmanr(first, second).statistic)

    regional = table(primary, "REGIONAL_GLM", "region")
    csv("sensitivity/paired_labels.csv", paired_labels(regional))
    primary_map = regional.t_ols
    alternate_map = table(alternate, "REGIONAL_GLM", "region").t_ols
    age_map = table(age, "REGIONAL_GLM", "region").t_ols.loc[primary_map.index]
    labels = []
    for branch in (primary, alternate):
        with np.load(REF / branch["atlas"] / "vertex_labels.npz") as data:
            labels.append(data["labels"])
    first, second = labels
    cortex = (first >= 0) & (second >= 0)
    rows = [
        compare("atlas_t", primary_map.to_numpy()[first[cortex]],
                alternate_map.to_numpy()[second[cortex]]),
        compare("age_t", primary_map, age_map),
    ]
    weights = table(primary, "CANONICAL_GENE_WEIGHTS", "gene").mean_PLS_weight
    for branch in (alternate, age):
        other = table(branch, "CANONICAL_GENE_WEIGHTS", "gene").mean_PLS_weight
        rows.append(compare(branch["id"] + "_weight", weights, other.loc[weights.index]))
    csv("sensitivity/concordance.csv", pd.DataFrame(rows))

    demographics = []
    for branch in (primary, age):
        for group, participants in cohort(branch).groupby("group"):
            for measure in ["age", "TIV", "IQR_percent", "Burke-Fahn_scale"]:
                values = participants[measure].dropna()
                if measure == "Burke-Fahn_scale" and group != "CD":
                    values = values.iloc[:0]
                demographics.append(dict(branch=branch["id"], group=group, measure=measure,
                                         n=len(participants), mean=values.mean(), sd=values.std(),
                                         minimum=values.min(), maximum=values.max(),
                                         n_male=int(participants.sex.eq("M").sum()),
                                         n_female=int(participants.sex.eq("F").sum())))
    csv("sensitivity/demographics.csv", pd.DataFrame(demographics))

    # Summarize the existing within-branch tests without refitting a pooled model.
    for name, keys, q_column, effect in [
        ("PET_PRIMARY", ["target"], "q_within_targets", "rho"),
        ("CANDIDATE_GENES", ["gene"], "q_BH12", None),
        ("SPATIAL_MEAN_R_ALL", ["collection", "term"], "q_within_collection", "mean_gene_r"),
    ]:
        frames = [table(branch, name, keys) for branch in branches]
        if not all(frame.index.sort_values().equals(frames[0].index.sort_values()) for frame in frames):
            raise ValueError(f"Input labels differ across branches for {name}")
        q = pd.concat([frame[q_column] for frame in frames], axis=1)
        stable = q.lt(0.05).all(axis=1)
        if effect is not None:
            effects = pd.concat([frame[effect] for frame in frames], axis=1)
            stable &= effects.lt(0).all(axis=1)
        result = stable.rename("meets_criterion_in_each_branch").reset_index()
        csv("sensitivity/" + name.lower() + ".csv", result)

