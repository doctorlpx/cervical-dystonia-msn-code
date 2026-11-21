from common import *


def run(b):
    base = "branches/" + b["id"]

    names = read_regions(b["atlas"])
    T = pd.read_csv(REF / b["atlas"] / "receptor_targets.csv", index_col=0)
    assert T.index.tolist() == names and T.shape == (b["regions"], 19)
    sp = spins(b["atlas"], "pet")
    P = T.to_numpy(float)
    U, sv, vt = np.linalg.svd(P, full_matrices=False)
    pc = U[:, 0] * sv[0]
    m = pd.read_csv(ROOT / base / "REGIONAL_GLM.csv")
    assert m.region.tolist() == names
    rows = []
    for metric in (["t_ols", "beta"] if b["full_downstream"] else ["t_ols"]):
        for adjustment in (
            ["none", "PC1_secondary"]
            if b["full_downstream"] and metric == "t_ols"
            else ["none"]
        ):
            r, p, null = pet_assoc(
                m[metric].to_numpy(), P, sp, None if adjustment == "none" else pc
            )
            q = bh(p)
            for i, target in enumerate(T.columns):
                rows.append(
                    dict(
                        branch=b["id"],
                        metric=metric,
                        adjustment=adjustment,
                        target=target,
                        n_regions=len(names),
                        rho=float(r[i]),
                        p_spin=float(p[i]),
                        q_within_targets=float(q[i]),
                        n_spins=10000,
                        role=(
                            "PRIMARY"
                            if metric == "t_ols" and adjustment == "none"
                            else "SECONDARY"
                        ),
                        external_healthy_map_not_patient_PET=True,
                    )
                )
            np.save(out(base + "/PET_" + metric + "_" + adjustment + "_NULL.npy"), null)
            # Independent observed and first-null estimator replay including reranking PC1 residuals.
            yy = stats.rankdata(m[metric].to_numpy())
            xx = stats.rankdata(P, axis=0)
            if adjustment == "none":
                check = np.array(
                    [
                        stats.spearmanr(m[metric], P[:, i]).statistic
                        for i in range(P.shape[1])
                    ]
                )
                checknull = np.array(
                    [
                        stats.pearsonr(yy[sp[:, 0]], xx[:, i]).statistic
                        for i in range(P.shape[1])
                    ]
                )
            else:
                C = np.column_stack([np.ones(len(pc)), stats.rankdata(pc)])
                M = np.eye(len(pc)) - C @ np.linalg.pinv(C)
                check = np.array(
                    [
                        stats.pearsonr(M @ yy, M @ xx[:, i]).statistic
                        for i in range(P.shape[1])
                    ]
                )
                checknull = np.array(
                    [
                        stats.pearsonr(M @ yy[sp[:, 0]], M @ xx[:, i]).statistic
                        for i in range(P.shape[1])
                    ]
                )
            assert np.allclose(check, r, atol=1e-12) and np.allclose(
                checknull, null[0], atol=1e-12
            )
    f = pd.DataFrame(rows)
    primary = f[f.role == "PRIMARY"]
    csv(base + "/PET_ALL.csv", f)
    csv(base + "/PET_PRIMARY.csv", primary)
    csv(base + "/PET_PRIMARY_HITS.csv", primary[primary.q_within_targets < 0.05])
