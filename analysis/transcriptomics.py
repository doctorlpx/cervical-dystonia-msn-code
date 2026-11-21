"""Fit and orient PLS1, rotate the target, omit each donor and bootstrap the two full samples."""

import warnings
from common import (
    CFG,
    CONTRAST,
    DONORS,
    ROOT,
    csv,
    design,
    expression,
    exprdir,
    load_pls,
    nodal,
    np,
    out,
    pd,
    progress,
    spatial_test,
    spins,
    z,
)


def run(b):
    base = "branches/" + b["id"] + "/"
    names, covered, X, genes, target, rotations, weights = load_pls(b)
    left = names[: len(target)]
    pd.read_csv(exprdir(b["atlas"]) / "sample_counts.csv", index_col=0).reindex(
        left
    ).to_csv(out(base + "AHBA_DONOR_SAMPLE_COUNTS.csv"))
    csv(base + "PLS_WEIGHTS.csv", pd.DataFrame({"gene": genes, "PLS1_weight": weights}))
    csv(
        base + "PLS_SCORES.csv",
        pd.DataFrame(
            {
                "region": np.array(left)[covered],
                "imaging_t": target[covered],
                "PLS1_score": X @ weights,
            }
        ),
    )
    group_maps = pd.read_csv(ROOT / base / "REGIONAL_GLM.csv")
    kernel = X @ X.T
    rows = []
    for metric in (["t_ols", "beta"] if b["full_downstream"] else ["t_ols"]):
        target_map = group_maps[metric].to_numpy()[: len(left)]
        result, _ = spatial_test(kernel, target_map, covered, rotations)
        row = dict(
            metric=metric,
            n_regions=len(covered),
            genes=len(genes),
            r=result["r"],
            p_spin=result["p_spin"],
        )
        if b["full_downstream"] and metric == "t_ols":
            alternative, _ = spatial_test(
                kernel, target_map, covered, spins(b["atlas"], "alternate")
            )
            row["alternate_NN_rotation_p"] = alternative["p_spin"]
        rows.append(row)
    csv(base + "PLS_RESULTS.csv", pd.DataFrame(rows))
    if not b["full_downstream"]:
        return
    donors = []
    for d in DONORS:
        f = pd.read_csv(exprdir(b["atlas"]) / ("donor_" + d + ".csv"), index_col=0)
        f.columns = f.columns.str.strip().str.upper()
        if f.columns.duplicated().any():
            f = f.T.groupby(level=0, sort=False).mean().T
        donors.append(f.reindex(index=left, columns=genes).to_numpy())
    stack = np.stack(donors)
    folds = []
    for i, d in enumerate(DONORS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reduced = np.nanmean(np.delete(stack, i, axis=0), axis=0)
        locations, values, features = expression(
            pd.DataFrame(reduced, index=left, columns=genes), left
        )
        result, _ = spatial_test(values @ values.T, target, locations, rotations)
        folds.append(
            dict(
                excluded_donor=d,
                n_regions=len(locations),
                genes=len(features),
                r=result["r"],
                p_spin=result["p_spin"],
            )
        )
    csv(base + "PLS_LODO.csv", pd.DataFrame(folds))


def stability(b):
    base = "branches/" + b["id"] + "/"
    names, covered, X, features, target, _, observed = load_pls(b)
    n = len(covered)
    rng = np.random.default_rng(CFG["seed"])
    counts = rng.multinomial(n, np.repeat(1 / n, n), size=2000).T
    y = z(target[covered])
    covariance = (
        X.T @ (counts * y[:, None]) - (X.T @ counts) * (y @ counts / n)[None, :]
    )
    regional = covariance / np.linalg.norm(covariance, axis=0)
    regional[:, (observed @ regional) < 0] *= -1
    # The submitted gene-wise SE used float32 bootstrap weights, aggregated
    # across aliases in float64 before the sample standard deviation.
    np.savez_compressed(
        out(base + "regional_bootstrap_weights.npz"), weights=regional.astype("float32")
    )
    del regional, covariance
    cohort, ordered, values = nodal(b)
    assert ordered == names
    full_design = design(cohort)
    groups = [np.flatnonzero(cohort.group.to_numpy() == g) for g in ["HC", "CD"]]
    rng = np.random.default_rng(CFG["seed"] + 1)
    correlations = []
    for i in range(1000):
        while True:
            draw = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            d = full_design[draw]
            if np.linalg.matrix_rank(d) == 7:
                break
        v = values[draw, : len(target)]
        inv = np.linalg.inv(d.T @ d)
        coef = inv @ d.T @ v
        residuals = v - d @ coef
        t = (CONTRAST @ coef) / np.sqrt(
            (residuals**2).sum(0) / (len(d) - 7) * (CONTRAST @ inv @ CONTRAST)
        )
        w = X.T @ z(t[covered])
        w /= np.linalg.norm(w)
        if observed @ w < 0:
            w = -w
        correlations.append(float(np.corrcoef(X @ w, t[covered])[0, 1]))
    csv(
        base + "SUBJECT_BOOTSTRAP_DIAGNOSTICS.csv",
        pd.DataFrame(dict(replicate=np.arange(1, 1001), PLS_score_r=correlations)),
    )
