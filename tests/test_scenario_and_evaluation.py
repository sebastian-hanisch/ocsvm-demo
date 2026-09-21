"""Szenario (identisch zu den Vorgänger-Demos, dazu der Korrelationsbruch), Kennzahlen, Analyse und Schwellen für vier Detektoren, Sweeps und Tabellen, Urteil."""

import numpy as np
import pytest

import ocsvm_algorithm as svm
import ocsvm_constants as C
import ocsvm_ee_algorithm as alg
import ocsvm_evaluation as ev
import ocsvm_isolation_forest as isf
import ocsvm_lof as lof

# Zeilensummen der ersten acht Zeilen der PCA-Demo (dieselben normalen Zeilen wie in den Vorgänger-Demos): permutationsinvariant, eingefroren
PCA_ROW_SUMS = [39967.27507413389, 42083.657538741994, 40172.02304566072, 44766.540605465496, 45034.44402326185, 55757.45917619934, 50727.55911151846, 43923.05872324106]


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_normal_rows_equal_the_pca_and_the_predecessor_rows():
    ds = ev.make_dataset(contamination=1)
    assert not ds.anomaly[:8].any() and ds.X.shape == (300, 12)
    assert np.allclose(ds.X[:8].sum(axis=1), PCA_ROW_SUMS, rtol=1e-12)


def test_default_dataset_is_frozen_at_the_predecessor_values():
    ds = ev.make_dataset()
    assert float(ds.X.sum()) == pytest.approx(13396385.215115668, rel=1e-12) and int(ds.anomaly.sum()) == 30 and ds.X[0, 0] == pytest.approx(19701.54827513291, rel=1e-12)


def test_noise_features_are_appended_and_change_nothing_else():
    base = ev.make_dataset()
    for n_noise in (1, 10, 40):
        ds = ev.make_dataset(n_noise=n_noise)
        assert ds.X.shape == (300, 12 + n_noise) and np.array_equal(ds.X[:, :12], base.X) and np.array_equal(ds.anomaly, base.anomaly)
        assert len(ds.names) == 12 + n_noise and ds.names[-1] == f"Rauschmerkmal {n_noise}" and ds.n_noise == n_noise
    assert np.array_equal(ev.make_dataset(p=30, n_noise=5).X[:, :30], ev.make_dataset(p=30).X)


@pytest.mark.parametrize("n,pct,expected", [(300, 10, 30), (300, 1, 3), (20, 1, 1), (30, 10, 3), (600, 45, 270)])
def test_contamination_is_exact_with_at_least_one_anomaly(n, pct, expected):
    assert int(ev.make_dataset(n=n, contamination=pct).anomaly.sum()) == expected


def test_gap_needs_two_modes_and_kinds_have_their_geometry():
    assert ev.make_dataset(kind="gap").kind == "scattered"
    gap = ev.make_dataset(n_modes=2, kind="gap")
    assert np.abs(gap.z[gap.anomaly]).max() < 1.2
    clu = ev.make_dataset(kind="cluster", contamination=20)
    assert np.linalg.norm(clu.z[clu.anomaly].mean(axis=0) - 6.0 * np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])) < 0.15


def test_decorrelated_anomalies_keep_the_marginals_and_lose_the_dependence():
    base = ev.make_dataset()
    ds = ev.make_dataset(kind="decorrelated", contamination=20)
    assert ds.kind == "decorrelated" and int(ds.anomaly.sum()) == 60 and ds.X.shape == base.X.shape
    normal = ds.X[~ds.anomaly]
    anom = ds.X[ds.anomaly]
    for j in range(ds.p):
        assert set(np.round(anom[:, j], 9)) <= set(np.round(normal[:, j], 9))                          # jede Spalte stammt aus den Normalen
    assert np.array_equal(normal, ev.make_dataset(kind="scattered", contamination=20).X[~ev.make_dataset(kind="scattered", contamination=20).anomaly])
    ds = ev.make_dataset(kind="decorrelated", contamination=40, n=600)
    c_norm = np.corrcoef(ds.X[~ds.anomaly].T)
    c_anom = np.corrcoef(ds.X[ds.anomaly].T)
    off = ~np.eye(ds.p, dtype=bool)
    assert np.abs(c_norm[off]).mean() > 0.4 and np.abs(c_anom[off]).mean() < 0.15                       # Abhängigkeit zerstört
    for j in range(ds.p):                                                                                 # gleiche Randverteilung: Quantile der Anomalien im Bereich der Normalen
        q_n, q_a = np.quantile(ds.X[~ds.anomaly][:, j], [0.05, 0.5, 0.95]), np.quantile(ds.X[ds.anomaly][:, j], [0.05, 0.5, 0.95])
        assert np.all(np.abs(q_n - q_a) < 0.35 * (q_n[2] - q_n[0]))


def test_decorrelated_is_drawn_last_so_other_kinds_and_defaults_stay_bit_identical():
    a = ev.make_dataset(seed=11, kind="scattered")
    b = ev.make_dataset(seed=11, kind="decorrelated")
    assert np.array_equal(a.X[~a.anomaly], b.X[~b.anomaly]) and np.array_equal(a.anomaly, b.anomaly)
    assert not np.array_equal(a.X[a.anomaly], b.X[b.anomaly])


def test_dataset_is_deterministic():
    assert np.array_equal(ev.make_dataset(seed=3, n_noise=4, kind="decorrelated").X, ev.make_dataset(seed=3, n_noise=4, kind="decorrelated").X)


# --- Kennzahlen: Handinstanzen ------------------------------------------------------------------------------------------------------------


def test_roc_auc_and_average_precision_hand_instances():
    assert ev.roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.5
    assert ev.roc_auc(np.array([1, 4, 2, 3.0]), np.array([0, 1, 1, 0], bool)) == pytest.approx(0.75)
    assert np.isnan(ev.roc_auc(np.array([1.0, 2.0]), np.array([False, False])))
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    assert ev.roc_auc(s, y) == pytest.approx(np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]))
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 0, 1, 0], bool)) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_curve_and_flag_metrics():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(300), rng.random(300) < 0.2
    fpr, tpr = ev.roc_curve(s, y)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and np.trapezoid(tpr, fpr) == pytest.approx(ev.roc_auc(s, y), abs=1e-9)
    m = ev.flag_metrics(np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 0, 0], bool))
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["false_alarm"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.4)


def test_lof_k_is_twenty_but_at_most_half_the_tours():
    assert [ev.lof_k(n) for n in (8, 20, 30, 100, 300, 600)] == [4, 10, 15, 20, 20, 20]


# --- Analyse und Schwellen ----------------------------------------------------------------------------------------------------------------


def _params(**kw):
    p = {**ev.DEFAULT_DATA, **kw}
    return (p["n"], p["p"], p["n_noise"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], 7)


def test_analysis_fields_and_consistency():
    a = ev.analyse_for(_params())
    assert set(a.scores) == set(ev.DETECTORS) == set(a.flags) == set(a.values) == set(a.oracle_f1) == {"ocsvm", "lof", "iforest", "robust"}
    assert a.p_total == 12 and a.forest_if.psi == 256 and a.robust.h == 156 and a.fit.converged and a.fit.nu == 0.1
    Z = ev.standardise(a.ds.X)
    assert np.allclose(a.fit.X, Z) and a.fit.gamma == pytest.approx(svm.gamma_scale(Z)) and a.fit.gamma == pytest.approx(1.0 / 12)                       # gamma_0 = 1 / p
    assert np.allclose(a.values["ocsvm"], -svm.decision_function(a.fit, Z)) and np.allclose(a.values["lof"], lof.fit_lof(Z, 20).lof)
    assert np.allclose(a.values["iforest"], isf.score_from_paths(isf.path_lengths(a.forest_if, a.ds.X), a.forest_if.psi))
    for d in ev.DETECTORS:
        assert a.scores[d]["n_flagged"] == int(a.flags[d].sum()) and 0 <= a.oracle_f1[d] <= 1
    assert a.scores["ocsvm"]["threshold"] == C.FLAG_EPS and (a.flags["ocsvm"] == (a.values["ocsvm"] > C.FLAG_EPS)).all()
    assert a.scores["lof"]["threshold"] == 1.5 and a.scores["iforest"]["threshold"] == 0.5 and a.scores["robust"]["threshold"] == pytest.approx(alg.threshold(a.robust, 0.975))
    assert a.scores["ocsvm"]["n_support"] == a.fit.n_support and a.scores["ocsvm"]["converged"] and 0 <= a.ecod_auc <= 1 and a.seconds["ocsvm"] > 0


def test_nu_bounds_hold_in_the_analysis():
    for nu in (0.05, 0.1, 0.3):
        a = ev.analyse_for(_params(), ev.Settings(nu=nu))
        assert a.scores["ocsvm"]["n_flagged"] / a.ds.n <= nu + 1e-9 <= a.fit.n_support / a.ds.n + 2e-9


def test_share_threshold_flags_exactly_the_assumed_share_for_all_four_detectors():
    for share in (2, 10, 40):
        a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=share))
        for d in ev.DETECTORS:
            assert int(a.flags[d].sum()) == max(1, round(share / 100 * 300))
            assert a.flags[d][np.argsort(-a.values[d])[:3]].all()
    a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=10))
    assert all(a.scores[d]["f1"] == pytest.approx(a.oracle_f1[d]) for d in ev.DETECTORS)                    # 10 % = wahrer Anteil


def test_nu_and_gamma_move_only_the_one_class_svm_and_the_chi2_quantile_only_the_robust_detector():
    base = ev.analyse_for(_params())
    for st in (ev.Settings(nu=0.3), ev.Settings(gamma_factor=0.1)):
        b = ev.analyse_for(_params(), st)
        assert b.scores["ocsvm"] != base.scores["ocsvm"] and all(b.scores[d] == base.scores[d] for d in ("lof", "iforest", "robust"))
    q = ev.analyse_for(_params(), ev.Settings(quantile=0.999))
    assert q.scores["ocsvm"] == base.scores["ocsvm"] and q.scores["robust"]["false_alarm"] <= base.scores["robust"]["false_alarm"] and q.scores["robust"]["auc"] == base.scores["robust"]["auc"]


def test_standardisation_matters_for_the_one_class_svm_and_the_lof_only():
    a = ev.analyse_for(_params())
    b = ev.analyse_for(_params(), ev.Settings(standardize=False))
    assert np.array_equal(a.values["iforest"], b.values["iforest"]) and a.values["robust"].tolist() == b.values["robust"].tolist()
    assert not np.allclose(a.values["ocsvm"], b.values["ocsvm"]) and not np.allclose(a.values["lof"], b.values["lof"]) and b.scores["ocsvm"]["auc"] < a.scores["ocsvm"]["auc"] - 0.05


def test_forest_seed_is_separate_from_the_data_seed_and_the_one_class_svm_and_lof_are_deterministic():
    a, b = ev.analyse_for(_params()), ev.analyse_for(_params(), ev.Settings(start=5))
    assert np.array_equal(a.ds.X, b.ds.X) and a.values["iforest"].tolist() != b.values["iforest"].tolist()
    assert np.array_equal(a.values["ocsvm"], b.values["ocsvm"]) and np.array_equal(a.values["lof"], b.values["lof"])


def test_noise_features_are_used_and_few_tours_many_features_work():
    b = ev.analyse_for(_params(n_noise=10))
    assert b.ds.X.shape[1] == 22 and b.p_total == 22 and b.fit.gamma == pytest.approx(1.0 / 22)
    c = ev.analyse_for(_params(n=20, p=30))
    assert 0 <= c.scores["ocsvm"]["auc"] <= 1 and c.forest_if.psi == 20 and c.fit.converged


def test_sweep_rows_labels_and_the_setting_sweeps():
    rows = ev.sweep("contamination", values=(5, 10))
    assert [r["x"] for r in rows] == [5, 10]
    for r in rows:
        for d in ev.DETECTORS:
            assert r[f"{d}_auc_min"] <= r[f"{d}_auc"] <= r[f"{d}_auc_max"] and r[f"{d}_auc_std"] >= 0 and f"{d}_oracle_f1" in r and f"{d}_seconds" in r
        assert "ecod_auc" in r and "ocsvm_support_share" in r and "ocsvm_bound_share" in r
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    g = ev.sweep("gamma_factor", values=(0.1, 8.0))
    assert g[0]["ocsvm_auc"] != g[1]["ocsvm_auc"] and g[0]["lof_auc"] == g[1]["lof_auc"] and g[0]["robust_auc"] == g[1]["robust_auc"] and g[0]["ocsvm_f1"] > g[1]["ocsvm_f1"]
    nu = ev.sweep("nu", values=(0.05, 0.3))
    assert nu[0]["ocsvm_support_share"] < nu[1]["ocsvm_support_share"] and nu[0]["iforest_auc"] == nu[1]["iforest_auc"] and nu[0]["ocsvm_recall"] < nu[1]["ocsvm_recall"]
    n = ev.sweep("n", values=(20,))
    assert n[0]["lof_auc"] > 0.9                                                                        # k = 20 würde bei n = 20 klemmen; der Vergleich begrenzt k auf n / 2


def test_settings_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.nu, s.gamma_factor, s.threshold_kind, s.quantile, s.share) == (C.DEFAULT_NU, C.DEFAULT_GAMMA_FACTOR, C.DEFAULT_THRESHOLD_KIND, C.DEFAULT_QUANTILE, C.DEFAULT_SHARE) and s.standardize is True
    assert set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS) and C.SWEEP_SEEDS == tuple(range(100000, 100005)) and set(ev.SWEEP_VALUES["gamma_factor"]) == set(C.GAMMA_FACTORS)


def test_score_maps_show_the_learned_boundary():
    ds = ev.make_dataset(p=2)
    X2 = ds.X[~ds.anomaly]
    xs, ys, F, S, sv = ev.score_maps(X2, grid=40)
    assert F.shape == S.shape == (40, 40) and len(xs) == len(ys) == 40 and sv.shape == (len(X2),) and sv.any()
    center = (np.argmin(np.abs(ys - X2[:, 1].mean())), np.argmin(np.abs(xs - X2[:, 0].mean())))
    assert F[center] > 0 and F[0, 0] < 0 and F[-1, -1] < 0 and S[0, 0] > S[center] + 0.1                 # innen positiv, weit außen negativ


# --- Experimente: Form der Tabellen ----------------------------------------------------------------------------------------------------------


def test_scenario_grid_nu_gap_standardise_cost_and_threshold_tables_have_their_documented_shape():
    assert len(ev.SCENARIOS) == 8
    sc = ev.scenario_table(scenarios=ev.SCENARIOS[:2])
    assert [r["scenario"] for r in sc] == [s[0] for s in ev.SCENARIOS[:2]] and all(f"{d}_auc" in sc[0] for d in ev.DETECTORS) and "ecod_auc" in sc[0]
    g = ev.grid_table(nus=(0.1, 0.2), gammas=(0.1, 1.0))
    assert [(c["nu"], c["gamma_factor"]) for c in g] == [(0.1, 0.1), (0.1, 1.0), (0.2, 0.1), (0.2, 1.0)] and all({"auc", "f1", "false_alarm", "support_share"} <= set(c) for c in g)
    nt = ev.nu_table(nus=(0.05, 0.2))
    assert [r["nu"] for r in nt] == [0.05, 0.2] and {"pure_flagged", "pure_support", "std_flagged", "std_support", "std_false_alarm"} <= set(nt[0])
    for r in nt:                                                                                        # die Schranken der Theorie
        assert r["pure_flagged"] <= r["nu"] + 1e-9 <= r["pure_support"] + 2e-9 and r["std_flagged"] <= r["nu"] + 1e-9 <= r["std_support"] + 2e-9
    gp = ev.gap_table(gammas=(0.1, 1.0))
    assert [(r["n_modes"], r["nu"]) for r in gp] == [(2, 0.1), (2, 0.3), (3, 0.1), (3, 0.3)] and all(sorted(r["by_gamma"]) == [0.1, 1.0] for r in gp)
    st_ = ev.standardise_table()
    assert [r["standardize"] for r in st_] == [True, False] and st_[0]["ocsvm_auc"] > st_[1]["ocsvm_auc"]
    ct = ev.cost_table()["times"]
    assert [(t["n"], t["p"]) for t in ct] == [(100, 12), (300, 12), (600, 12), (100, 52), (300, 52), (600, 52)] and ct[2]["kernel_mb"] == pytest.approx(600 * 600 * 8 / 1e6)
    tt = ev.threshold_table(ev.Settings())
    assert [r["x"] for r in tt["cutoff"]] == list(ev.SWEEP_VALUES["nu"]) and [r["factor"] for r in tt["wrong_share"]] == [0.5, 1.0, 2.0] and [r["x"] for r in tt["wrong_share"]] == [5, 10, 20]


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _verdict(**kw):
    settings = ev.Settings(**{k: kw.pop(k) for k in ("nu", "gamma_factor", "threshold_kind", "share") if k in kw})
    return ev.verdict(ev.analyse_for(_params(**kw), settings))


def test_verdict_codes_for_the_presets_and_edge_cases():
    assert _verdict()[1] in ("comparable", "others_win")
    assert _verdict(gamma_factor=0.1)[:2] == ("success", "comparable")
    assert _verdict(gamma_factor=8.0)[:2] == ("warning", "others_win")
    assert _verdict(n_noise=40, gamma_factor=0.1)[:2] == ("success", "ocsvm_wins")
    assert _verdict(n_modes=3, kind="gap")[:2] == ("warning", "gap_rank")
    assert _verdict(contamination=45)[:2] == ("warning", "nu_low")
    assert _verdict(contamination=45, nu=0.5)[1] in ("comparable", "ocsvm_wins")
    assert _verdict(kind="cluster", contamination=30, nu=0.3, gamma_factor=0.1)[:2] == ("success", "ocsvm_wins")


def test_verdict_data_carries_the_numbers_the_messages_use():
    kind, code, data = _verdict(contamination=45)
    assert code == "nu_low"
    for key in ("ocsvm_auc", "lof_auc", "iforest_auc", "robust_auc", "best_other_auc", "best_other_f1", "ocsvm_recall", "ocsvm_false_alarm", "ocsvm_f1", "ocsvm_n_flagged", "lof_f1", "iforest_f1", "robust_f1", "oracle_ocsvm",
                "contamination", "n_anomalies", "n", "p", "kind", "nu", "gamma_factor", "ecod_auc"):
        assert key in data
    assert data["p"] == 12 and data["n_anomalies"] == 135 and data["contamination"] == pytest.approx(45.0) and data["nu"] == 0.1


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["ocsvm"] < 1.5 and a.seconds["lof"] < 1.5 and a.seconds["iforest"] < 4.0 and a.seconds["robust"] < 10.0
