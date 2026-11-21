from common import *


def freeman_lane(Y, X, B, seed):
    X0 = np.delete(X, 1, axis=1)
    hat0 = X0 @ np.linalg.pinv(X0)
    fitted = hat0 @ Y
    res = Y - fitted
    pinv = np.linalg.pinv(X)
    inv = np.linalg.inv(X.T @ X)
    df = len(X) - X.shape[1]
    scale = CONTRAST @ inv @ CONTRAST
    obs = fit(Y, X)["t_ols"]
    observed = float(np.sum(obs**2))
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for start in range(0, B, 64):
        k = min(64, B - start)
        ix = np.stack([rng.permutation(len(X)) for _ in range(k)])
        yy = fitted[None, :, :] + res[ix]
        ymat = yy.transpose(1, 0, 2).reshape(len(X), -1)
        coef = pinv @ ymat
        ee = ymat - X @ coef
        tt = (CONTRAST @ coef) / np.sqrt(np.sum(ee**2, axis=0) / df * scale)
        null[start : start + k] = np.sum(tt.reshape(k, -1) ** 2, axis=1)
    assert np.isfinite(null).all()
    return (
        dict(
            statistic="sum(t^2)",
            observed=observed,
            p_permutation=float((np.sum(null >= observed - 1e-12) + 1) / (B + 1)),
            n_permutations=B,
            role="SECONDARY_WHOLE_MAP",
            scheme="Freedman-Lane: reduced-model residual rows jointly permuted across all parcels; full model refit",
            assumption="exchangeable reduced-model errors; nuisance adjustment, not a guarantee of identical residual variance",
        ),
        null,
    )


def clinical(b, c, S, labels):
    cd = (c.group == "CD").to_numpy()
    cc = c.loc[cd].reset_index(drop=True)
    y = stats.rankdata(cc["Burke-Fahn_scale"].to_numpy(float))
    assert len(cc) == 24 and np.isfinite(y).all()
    age = cc.age.to_numpy(float)
    age -= age.mean()
    C = np.column_stack(
        [
            np.ones(24),
            age,
            age**2,
            (cc.sex == "M").astype(float),
            z(cc.TIV),
            z(cc.IQR_percent),
        ]
    )
    assert np.linalg.matrix_rank(C) == 6
    M = np.eye(24) - C @ np.linalg.pinv(C)
    ey = M @ y
    fitted = y - ey
    xx = stats.rankdata(S[cd], axis=0)
    ex = M @ xx
    ex /= np.linalg.norm(ex, axis=0)
    obs = (ey / np.linalg.norm(ey)) @ ex
    rng = np.random.default_rng(CFG["permutation_seed"] + 700)
    null = np.empty((10000, len(labels)))
    for start in range(0, 10000, 250):
        ix = np.stack([rng.permutation(24) for _ in range(250)], axis=1)
        eperm = M @ (fitted[:, None] + ey[ix])
        eperm /= np.linalg.norm(eperm, axis=0)
        null[start : start + 250] = eperm.T @ ex
    p = (np.sum(abs(null) >= abs(obs) - 1e-12, axis=0) + 1) / 10001
    f = pd.DataFrame(
        dict(
            summary=labels,
            partial_rank_rho=obs,
            p_permutation=p,
            q_BH8=bh(p),
            n_CD=24,
            role="EXPLORATORY",
            ranked="BFM and MSN summaries; age/age_squared/sex/TIV/IQR as raw covariates",
            BFM_unique_values=int(cc["Burke-Fahn_scale"].nunique()),
        )
    )
    csv("branches/" + b["id"] + "/CLINICAL_BFM.csv", f)
    np.save(out("branches/" + b["id"] + "/CLINICAL_BFM_NULL.npy"), null)


def run(b):
    base = "branches/" + b["id"]

    c, names, Y = nodal(b)
    X = design(c)
    fitres = fit(Y, X)
    fields = {k: v for k, v in fitres.items() if isinstance(v, np.ndarray)}
    f = pd.DataFrame(dict(region=names, **fields))
    f["q_ols"] = bh(f.p_ols)
    f["q_hc3"] = bh(f.p_hc3)
    f["CD_mean"] = Y[(c.group == "CD").to_numpy()].mean(0)
    f["HC_mean"] = Y[(c.group == "HC").to_numpy()].mean(0)
    f["df"] = len(c) - 7
    csv(base + "/REGIONAL_GLM.csv", f)
    csv(base + "/REGIONAL_FDR_HITS.csv", f[f.q_ols < 0.05].sort_values("q_ols"))
    # Independent OLS solve verifies the primary group coefficient / t statistic.
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ coef
    testt = coef[1] / np.sqrt(
        (e**2).sum(0) / (len(c) - 7) * np.linalg.inv(X.T @ X)[1, 1]
    )
    assert np.allclose(coef[1], f.beta, rtol=1e-9, atol=1e-12) and np.allclose(
        testt, f.t_ols, atol=1e-10
    )
    networks = pd.read_csv(REF / b["atlas"] / "networks.csv")
    assert networks.region.tolist() == names
    S = np.column_stack(
        [Y.mean(1)]
        + [Y[:, networks.network7.to_numpy() == sys].mean(1) for sys in SYSTEMS]
    )
    slabels = ["Global"] + SYSTEMS
    sr = fit(S, X)
    sf = pd.DataFrame(
        dict(
            summary=slabels,
            **{k: v for k, v in sr.items() if isinstance(v, np.ndarray)}
        )
    )
    sf["q_ols_BH8"] = bh(sf.p_ols)
    sf["q_hc3_BH8"] = bh(sf.p_hc3)
    csv(base + "/GLOBAL_YEO7_GLM.csv", sf)
    csv(
        base + "/GLOBAL_YEO7_SUBJECT_VALUES.csv",
        pd.DataFrame(S, columns=slabels).assign(
            participant_id=c.participant_id, group=c.group
        ),
    )
    if b["full_downstream"]:
        clinical(b, c, S, slabels)
    progress(b["id"], "WHOLE_MAP_PERMUTATIONS", permutations=10000)
    wm, null = freeman_lane(Y, X, 10000, CFG["permutation_seed"])
    np.save(out(base + "/WHOLE_MAP_NULL.npy"), null)
    dump(base + "/WHOLE_MAP_PERMUTATION.json", wm)
