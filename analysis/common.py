"""Shared estimators. Input and output roots are specified by the command-line driver."""

from pathlib import Path
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats

from preparation.study_settings import DESIGN as CFG

REF = Path(os.environ["MSN_REFERENCE_DIR"])
INPUT = Path(os.environ["MSN_INPUT_DIR"])
ROOT = Path(os.environ["MSN_OUTPUT_DIR"])
CONTRAST = np.array([0, 1, 0, 0, 0, 0, 0.0])
DONORS = CFG["donors"]
SYSTEMS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def progress(stage, status, **kwargs):
    print(stage, status, flush=True)


def cohort(b):
    c = pd.read_csv(INPUT / "cohort.csv")
    if (
        not c.participant_id.is_unique
        or not c.group.isin(["CD", "HC"]).all()
        or not c.sex.isin(["M", "F"]).all()
    ):
        raise ValueError(
            "Check participant identifiers, group and sex coding against the input specification."
        )
    if not b["full_downstream"]:
        c = c[c.age.between(35, 69)].reset_index(drop=True)
    if c.group.value_counts().to_dict() != {"CD": b["CD"], "HC": b["HC"]}:
        raise ValueError("Unexpected cohort size for " + b["id"])
    return c


def exprdir(atlas):
    return INPUT / "expression" / atlas


def spins(atlas, kind="vasa"):
    with np.load(REF / atlas / "rotations.npz") as f:
        a = f[kind]
    half = int(atlas[8:]) // 2
    n = 2 * half if kind == "pet" else half
    assert a.shape == (n, 10000 if kind == "pet" else 5000)
    assert (a >= 0).all() and (a < n).all()
    if kind != "alternate":
        assert (np.sort(a, axis=0) == np.arange(n)[:, None]).all()
    if kind == "pet":
        assert ((a < half) == (np.arange(n)[:, None] < half)).all()
    return a


def read_regions(atlas):
    return (REF / atlas / "regions.txt").read_text(encoding="utf-8-sig").splitlines()


def out(rel):
    p = (ROOT / rel).resolve()
    assert p.is_relative_to(ROOT.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump(rel, v):
    out(rel).write_text(
        json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def csv(rel, f):
    f.to_csv(out(rel), index=False, encoding="utf-8-sig")


def bh(p):
    p = np.asarray(p, float)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    ix = np.argsort(p)
    n = len(p)
    q = np.minimum.accumulate((p[ix] * n / np.arange(1, n + 1))[::-1])[::-1]
    ans = np.empty(n)
    ans[ix] = np.minimum(q, 1)
    return ans


def z(x):
    x = np.asarray(x, float)
    s = x.std(axis=0, ddof=1)
    assert np.all(s > 0)
    return (x - x.mean(axis=0)) / s


def design(c):
    age = c.age.to_numpy(float)
    age = age - age.mean()
    X = np.column_stack(
        [
            np.ones(len(c)),
            (c.group == "CD").astype(float),
            age,
            age * age,
            (c.sex == "M").astype(float),
            z(c.TIV),
            z(c.IQR_percent),
        ]
    )
    assert X.shape[1] == np.linalg.matrix_rank(X) == 7
    return X


def fit(Y, X):
    inv = np.linalg.inv(X.T @ X)
    betas = inv @ X.T @ Y
    e = Y - X @ betas
    df = len(X) - X.shape[1]
    beta = CONTRAST @ betas
    v = CONTRAST @ inv @ CONTRAST
    se = np.sqrt((e * e).sum(0) / df * v)
    t = beta / se
    h = np.einsum("ij,ij->i", X, X @ inv)
    influence = X @ inv @ CONTRAST
    se3 = np.sqrt(np.sum((e * influence[:, None] / (1 - h[:, None])) ** 2, axis=0))
    return dict(
        beta=beta,
        se=se,
        ci95_lower=beta - stats.t.ppf(0.975, df) * se,
        ci95_upper=beta + stats.t.ppf(0.975, df) * se,
        t_ols=t,
        p_ols=2 * stats.t.sf(abs(t), df),
        se_hc3=se3,
        t_hc3=beta / se3,
        p_hc3=2 * stats.t.sf(abs(beta / se3), df),
        leverage_max=float(h.max()),
        df=df,
    )


def nodal(b):
    c = cohort(b)
    f = pd.read_csv(INPUT / (b["atlas"] + "_nodal60.csv")).set_index("Subject")
    names = f.columns.tolist()
    assert names == read_regions(b["atlas"])
    Y = f.loc[c.participant_id].to_numpy(float)
    assert Y.shape == (len(c), b["regions"]) and np.isfinite(Y).all()
    return c, names, Y


def expression(f, left):
    f = f.copy()
    f.index = f.index.astype(str)
    f.columns = f.columns.astype(str).str.strip().str.upper()
    if f.columns.duplicated().any():
        f = f.T.groupby(level=0, sort=False).mean().T
    f = f.reindex(left)
    cv = np.flatnonzero(f.notna().any(axis=1))
    f = f.iloc[cv].dropna(axis=1)
    f = f.loc[:, f.std(axis=0, ddof=1) > 0]
    X = z(f.to_numpy())
    assert np.isfinite(X).all()
    return cv, X, f.columns.tolist()


def correlations(K, Y):
    y = z(Y)
    s = K @ y
    s -= s.mean(axis=0)
    return abs(
        np.sum(s * y, axis=0) / np.sqrt(np.sum(s * s, axis=0) * np.sum(y * y, axis=0))
    )


def spatial_test(K, y, cv, sp):
    r = float(correlations(K, y[cv]))
    null = np.empty(sp.shape[1])
    for start in range(0, len(null), 256):
        ys = y[sp[:, start : start + 256]][cv]
        null[start : start + ys.shape[1]] = correlations(K, ys)
    assert np.isfinite(null).all()
    return (
        dict(
            r=r,
            p_spin=float((np.sum(null >= r - 1e-12) + 1) / (len(null) + 1)),
            null_mean=float(null.mean()),
            null_p95=float(np.quantile(null, 0.95)),
            excess_over_null_mean=float(r - null.mean()),
            n_spins=len(null),
        ),
        null,
    )


def pet_assoc(y, P, sp, pc=None):
    y = stats.rankdata(y)
    x = stats.rankdata(P, axis=0)
    yn = y[sp]
    if pc is not None:
        C = np.column_stack([np.ones(len(y)), stats.rankdata(pc)])
        proj = C @ np.linalg.pinv(C)
        y = y - proj @ y
        x = x - proj @ x
        yn = yn - proj @ yn
    else:
        y = y - y.mean()
        x = x - x.mean(0)
        yn = yn - yn.mean(0)
    y /= np.linalg.norm(y)
    x /= np.linalg.norm(x, axis=0)
    yn /= np.linalg.norm(yn, axis=0)
    obs = y @ x
    null = yn.T @ x
    p = (np.sum(abs(null) >= abs(obs) - 1e-12, axis=0) + 1) / (len(null) + 1)
    return obs, p, null


def load_pls(b):
    names = read_regions(b["atlas"])
    left = names[: len(names) // 2]
    cv, X, genes = expression(
        pd.read_csv(exprdir(b["atlas"]) / "expression.csv", index_col=0), left
    )
    m = pd.read_csv(ROOT / "branches" / b["id"] / "REGIONAL_GLM.csv")
    assert m.region.tolist() == names
    y = m.t_ols.to_numpy()[: len(left)]
    sp = spins(b["atlas"])
    w = X.T @ z(y[cv])
    w /= np.linalg.norm(w)
    return names, cv, X, genes, y, sp, w
