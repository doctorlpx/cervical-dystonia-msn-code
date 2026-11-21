"""Gene-level and fixed-set inference, with each complete multiple-testing family retained."""

from common import *
from scipy.sparse import csr_matrix


def membership_matrices(features):
    mapping = (
        pd.read_csv(REF / "gene_names.csv.gz").set_index("measured_gene").loc[features]
    )
    canonical = sorted(mapping.canonical_gene.unique())
    idx = pd.Index(canonical).get_indexer(mapping.canonical_gene)
    symbol = csr_matrix(
        (1 / np.bincount(idx)[idx], (idx, np.arange(len(idx)))),
        shape=(len(canonical), len(idx)),
    )
    sets = pd.read_csv(REF / "gene_sets.csv.gz")
    members = [
        pd.Index(canonical).get_indexer(term.split(";")) for term in sets.gene_members
    ]
    assert all((group >= 0).all() for group in members)
    row = np.repeat(np.arange(len(sets)), [len(group) for group in members])
    col = np.concatenate(members)
    weights = 1 / np.repeat(sets.matched_genes, sets.matched_genes).to_numpy(
        dtype=float
    )
    matrix = csr_matrix((weights, (row, col)), shape=(len(sets), len(canonical)))
    return canonical, symbol, sets, matrix


def fixed_spatial(profile, y, cv, spin):
    n = len(cv)
    obs = profile @ z(y[cv]) / (n - 1)
    high = np.zeros(len(obs), int)
    low = high.copy()
    avg = np.zeros(len(obs))
    for start in range(0, spin.shape[1], 128):
        batch = profile @ z(y[spin[:, start : start + 128]][cv]) / (n - 1)
        high += (batch >= obs[:, None] - 1e-12).sum(1)
        low += (batch <= obs[:, None] + 1e-12).sum(1)
        avg += batch.sum(1)
    p = np.minimum(1, 2 * (np.minimum(high, low) + 1) / (spin.shape[1] + 1))
    return obs, p, avg / spin.shape[1]


def conventional(sets, matrix, wc, selected, cap):
    rows = []
    cardinal = matrix.copy()
    cardinal.data[:] = 1
    for sign, positive in [("positive", True), ("negative", False)]:
        mask = selected & ((wc > 0) if positive else (wc < 0))
        if cap is not None and not positive:
            mask = mask & cap
        counts = np.rint(cardinal @ mask.astype(float)).astype(int)
        p = stats.hypergeom.sf(counts - 1, len(wc), sets.matched_genes, int(mask.sum()))
        rows.append(
            sets[["collection", "term"]].assign(
                direction=sign,
                overlap=counts,
                query_size=int(mask.sum()),
                p_hypergeometric=p,
            )
        )
    ora = pd.concat(rows, ignore_index=True)
    ora["q_within_collection_both_lists"] = ora.groupby(
        "collection"
    ).p_hypergeometric.transform(bh)
    return ora


def run(b, regional_bootstraps=True):
    dest = "branches/" + b["id"] + "/"
    names, cv, X, features, y, spin, w = load_pls(b)
    canonical, C, sets, A = membership_matrices(features)
    raw = C @ X.T
    wc = np.asarray(C @ w).ravel()
    gene = pd.DataFrame(dict(gene=canonical, mean_PLS_weight=wc))
    obs, p, avg = fixed_spatial(A @ raw, y, cv, spin)
    spatial = sets[["collection", "term", "matched_genes"]].assign(
        mean_gene_r=obs, null_mean=avg, p_spatial_two_sided=p
    )
    spatial["q_within_collection"] = spatial.groupby(
        "collection"
    ).p_spatial_two_sided.transform(bh)
    csv(dest + "SPATIAL_MEAN_R_ALL.csv", spatial)
    candidates = CFG["disease_sets"]["GeneReviews_isolated_dystonia_2025"]
    indices = pd.Index(canonical).get_indexer(candidates)
    assert (indices >= 0).all()
    obs, p, _ = fixed_spatial(raw[indices], y, cv, spin)
    candidate = pd.DataFrame(dict(gene=candidates, r=obs, p_spin=p, q_BH12=bh(p)))
    if b["full_downstream"] and regional_bootstraps:
        with np.load(ROOT / dest / "regional_bootstrap_weights.npz") as f:
            se = (C @ f["weights"].astype(float)).std(1, ddof=1)
        ratio = wc / se
        q = bh(2 * stats.norm.sf(abs(ratio)))
        selected = q < 0.01
        gene = gene.assign(
            bootstrap_ratio=ratio,
            q_gene_BH=q,
            selected_q01=selected,
            direction=np.where(wc > 0, "positive", "negative"),
        )
        candidate = candidate.merge(
            gene[["gene", "selected_q01", "direction"]], on="gene"
        )
        csv(
            dest + "CONVENTIONAL_ORA_ALL.csv", conventional(sets, A, wc, selected, None)
        )
        if b["id"] == "S400_full60":
            ranked = (
                gene[gene.selected_q01]
                .assign(abs_bsr=lambda d: d.bootstrap_ratio.abs())
                .sort_values(["abs_bsr", "gene"], ascending=[False, True])
            )
            positive = ranked.loc[ranked.direction.eq("positive"), "gene"]
            negative = ranked.loc[ranked.direction.eq("negative"), "gene"].iloc[:3000]
            capped = pd.Index(canonical).isin(negative)
            csv(dest + "CAPPED_ORA.csv", conventional(sets, A, wc, selected, capped))
            for tag, values in [
                ("positive", positive),
                ("negative", negative),
                ("background", canonical),
            ]:
                out("metascape_inputs/" + tag + ".txt").write_text(
                    "\n".join(values) + "\n", encoding="utf8"
                )
    csv(dest + "CANONICAL_GENE_WEIGHTS.csv", gene)
    csv(dest + "CANDIDATE_GENES.csv", candidate)
