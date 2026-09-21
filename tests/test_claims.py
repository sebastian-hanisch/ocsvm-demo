"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo die One-Class SVM gegen LOF, Isolation Forest, die Wurzel oder ECOD verliert, steht das hier ebenso als Test wie dort, wo sie gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import ocsvm_constants as C
import ocsvm_evaluation as ev

TOL = 0.02
STANDARD = ev.Settings()
WIDE = ev.Settings(gamma_factor=0.1)


def S(**kw):
    return ev.Settings(**kw)


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev._analyse_seed(s, settings, dict(items)) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    if det == "ecod":
        return float(np.mean([a.ecod_auc for a in runs(settings, **kw)]))
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(det, settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1[det] for a in runs(settings, **kw)]))


def support_share(settings=STANDARD, **kw):
    return float(np.mean([a.fit.n_support / a.ds.n for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@lru_cache(maxsize=None)
def grid():
    return {(c["nu"], c["gamma_factor"]): c for c in ev.grid_table()}


@lru_cache(maxsize=None)
def nu_tab():
    return {r["nu"]: r for r in ev.nu_table()}


@lru_cache(maxsize=None)
def gap_tab():
    return {(r["n_modes"], r["nu"]): r["by_gamma"] for r in ev.gap_table()}


# --- Seitenleiste: Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil -------------------------------------------------


@pytest.mark.parametrize("n,auc,f1,lof_f1,if_f1", [(20, 0.76, 0.00, 0.96, 0.60), (30, 0.88, 0.00, 0.94, 0.67), (50, 0.86, 0.00, 0.88, 0.78), (100, 0.92, 0.10, 0.87, 0.90), (200, 0.94, 0.64, 0.91, 0.96),
                                                   (400, 0.95, 0.72, 0.94, 0.96), (600, 0.96, 0.76, 0.90, 0.95)])
def test_tours_sweep_the_threshold_flags_nothing_at_small_n(n, auc, f1, lof_f1, if_f1):
    near(m("ocsvm", "auc", n=n), auc, 0.012)
    near(m("ocsvm", "f1", n=n), f1, 0.015)
    near(m("lof", "f1", n=n), lof_f1)
    near(m("iforest", "f1", n=n), if_f1)


@pytest.mark.parametrize("p,auc,f1", [(2, 0.90, 0.70), (5, 0.96, 0.67), (8, 0.95, 0.66), (12, 0.95, 0.67), (20, 0.95, 0.68), (30, 0.93, 0.68)])
def test_features_sweep(p, auc, f1):
    near(m("ocsvm", "auc", p=p), auc, 0.012)
    near(m("ocsvm", "f1", p=p), f1, 0.015)


@pytest.mark.parametrize("nn,auc,recall,f1,lof_recall", [(0, 0.95, 0.52, 0.67, 1.00), (5, 0.95, 0.30, 0.45, 0.93), (10, 0.94, 0.29, 0.45, 0.71), (20, 0.93, 0.26, 0.41, 0.18), (30, 0.91, 0.25, 0.40, 0.03), (40, 0.91, 0.25, 0.39, 0.00)])
def test_noise_features_with_the_default_gamma(nn, auc, recall, f1, lof_recall):
    near(m("ocsvm", "auc", n_noise=nn), auc, 0.012)
    near(m("ocsvm", "recall", n_noise=nn), recall)
    near(m("ocsvm", "f1", n_noise=nn), f1)
    near(m("lof", "recall", n_noise=nn), lof_recall)


@pytest.mark.parametrize("nn", [0, 5, 10, 20, 30, 40])
def test_noise_features_with_a_broad_kernel_stay_at_the_top(nn):
    assert 0.99 - 0.006 <= m("ocsvm", "auc", WIDE, n_noise=nn) <= 1.0
    assert 0.915 <= m("ocsvm", "f1", WIDE, n_noise=nn) <= 0.96


def test_noise_features_hurt_lof_and_robust_but_not_the_broad_kernel():
    near(m("robust", "auc", n_noise=0), 1.00, 0.01)
    near(m("robust", "auc", n_noise=40), 0.88, 0.015)
    near(m("lof", "f1", WIDE, n_noise=40), 0.00, 0.02)
    near(m("iforest", "f1", WIDE, n_noise=40), 0.56)
    near(m("iforest", "recall", n_noise=40), 0.39)
    near(m("robust", "f1", n_noise=40), 0.35)


@pytest.mark.parametrize("modes,auc,f1,robust_auc", [(1, 0.95, 0.67, 1.00), (2, 0.95, 0.76, 0.95), (3, 0.91, 0.77, 0.85)])
def test_modes_sweep(modes, auc, f1, robust_auc):
    near(m("ocsvm", "auc", n_modes=modes), auc, 0.012)
    near(m("ocsvm", "f1", n_modes=modes), f1)
    near(m("robust", "auc", n_modes=modes), robust_auc, 0.015)


@pytest.mark.parametrize("curv,auc,f1,lof_f1", [(0.0, 0.95, 0.67, 0.94), (0.25, 0.99, 0.78, 0.90), (0.5, 0.99, 0.84, 0.83), (0.75, 0.98, 0.85, 0.78), (1.0, 0.98, 0.85, 0.74)])
def test_curvature_helps_the_learned_boundary_and_hurts_lof_and_the_ellipse(curv, auc, f1, lof_f1):
    near(m("ocsvm", "auc", curvature=curv), auc, 0.012)
    near(m("ocsvm", "f1", curvature=curv), f1)
    near(m("lof", "f1", curvature=curv), lof_f1)


def test_curvature_kills_the_robust_ellipse_at_the_threshold():
    near(m("robust", "f1", curvature=0.0), 0.84)
    near(m("robust", "f1", curvature=1.0), 0.38)


@pytest.mark.parametrize("noise,auc", [(0.0, 0.92), (0.25, 0.95), (0.5, 0.97), (0.75, 0.97), (1.0, 0.96)])
def test_measurement_noise_auc(noise, auc):
    near(m("ocsvm", "auc", noise=noise), auc, 0.012)


def test_measurement_noise_lowers_the_threshold_metrics():
    near(m("ocsvm", "f1", noise=0.0), 0.70)
    near(m("ocsvm", "f1", noise=1.0), 0.53)
    near(m("ocsvm", "recall", noise=0.0), 0.57)
    near(m("ocsvm", "recall", noise=1.0), 0.37)


@pytest.mark.parametrize("c,auc,lof_auc", [(2, 1.00, 1.00), (5, 1.00, 1.00), (10, 0.95, 1.00), (20, 0.77, 0.99), (30, 0.56, 0.90), (40, 0.37, 0.72), (45, 0.32, 0.64)])
def test_contamination_above_nu_pushes_the_auc_below_chance(c, auc, lof_auc):
    near(m("ocsvm", "auc", contamination=c), auc, 0.012 if c <= 10 else 0.02)
    near(m("lof", "auc", contamination=c), lof_auc)


def test_contamination_others_are_insensitive():
    near(m("iforest", "auc", contamination=45), 1.00, 0.01)
    near(m("ecod", "auc", contamination=45), 1.00, 0.01)
    near(m("robust", "auc", contamination=2), 1.00, 0.01)
    near(m("robust", "auc", contamination=45), 0.87, 0.015)


@pytest.mark.parametrize("kw,auc,lof,iforest,robust,ecod", [(dict(kind="cluster"), 0.71, 0.35, 0.95, 1.00, 0.98), (dict(kind="cluster", contamination=30), 0.74, 0.47, 0.70, 0.49, 0.84),
                                                            (dict(n_modes=2, kind="gap"), 0.64, 0.53, 0.54, 0.40, 0.04), (dict(n_modes=3, kind="gap"), 0.82, 0.36, 0.27, 0.38, 0.01),
                                                            (dict(kind="decorrelated"), 0.88, 0.99, 0.80, 1.00, 0.56)])
def test_kind_help_numbers(kw, auc, lof, iforest, robust, ecod):
    near(m("ocsvm", "auc", **kw), auc, 0.02)
    near(m("lof", "auc", **kw), lof)
    near(m("iforest", "auc", **kw), iforest)
    near(m("robust", "auc", **kw), robust)
    near(m("ecod", "auc", **kw), ecod, 0.03)


def test_the_gap_others_range():
    values = [m(d, "auc", n_modes=nm, kind="gap") for d in ("lof", "iforest", "robust") for nm in (2, 3)]
    assert 0.26 <= min(values) and max(values) <= 0.55


# --- Regler der One-Class SVM ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nu,auc,recall,fa,f1", [(0.01, 0.89, 0.00, 0.000, 0.00), (0.02, 0.89, 0.00, 0.000, 0.00), (0.05, 0.90, 0.01, 0.000, 0.03), (0.10, 0.95, 0.52, 0.002, 0.67), (0.20, 1.00, 1.00, 0.09, 0.71),
                                                  (0.30, 1.00, 1.00, 0.21, 0.51), (0.50, 1.00, 1.00, 0.44, 0.34)])
def test_nu_help_numbers(nu, auc, recall, fa, f1):
    s = S(nu=nu)
    near(m("ocsvm", "auc", s), auc, 0.012)
    near(m("ocsvm", "recall", s), recall, 0.015)
    near(m("ocsvm", "false_alarm", s), fa, 0.01)
    near(m("ocsvm", "f1", s), f1, 0.015)


@pytest.mark.parametrize("g,f1,auc", [(0.02, 0.95, 1.00), (0.05, 0.96, 1.00), (0.1, 0.95, 1.00), (0.25, 0.90, 0.96), (0.5, 0.81, 0.94), (1.0, 0.67, 0.95), (2.0, 0.44, 0.95), (4.0, 0.03, 0.90), (8.0, 0.00, 0.83),
                                      (16.0, 0.00, 0.68), (32.0, 0.00, 0.60)])
def test_gamma_help_numbers(g, f1, auc):
    s = S(gamma_factor=g)
    near(m("ocsvm", "f1", s), f1, 0.015)
    near(m("ocsvm", "auc", s), auc, 0.012)


@pytest.mark.parametrize("q,f1", [(0.9, 0.67), (0.95, 0.77), (0.975, 0.84), (0.99, 0.89), (0.999, 0.90)])
def test_chi2_quantile_help_numbers(q, f1):
    near(m("robust", "f1", S(quantile=q)), f1)


@pytest.mark.parametrize("share,svm_f1,lof_if,robust", [(2, 0.33, 0.33, 0.33), (5, 0.64, 0.67, 0.67), (10, 0.67, 0.97, 0.90), (20, 0.63, 0.67, 0.66), (40, 0.38, 0.40, 0.40)])
def test_assumed_share_costs_all_detectors_about_the_same(share, svm_f1, lof_if, robust):
    s = S(threshold_kind="share", share=share)
    near(m("ocsvm", "f1", s), svm_f1)
    near(m("lof", "f1", s), lof_if)
    near(m("iforest", "f1", s), lof_if)
    near(m("robust", "f1", s), robust)


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_standard_preset_numbers():
    near(m("ocsvm", "auc"), 0.95, 0.012)
    near(m("ocsvm", "f1"), 0.67)
    near(m("ocsvm", "recall"), 0.52)
    near(m("ocsvm", "false_alarm"), 0.0, 0.01)
    near(m("lof", "f1"), 0.94)
    near(m("iforest", "f1"), 0.96)
    near(m("robust", "f1"), 0.84)
    for d in ("lof", "iforest", "robust"):
        near(m(d, "auc"), 1.00, 0.01)


def test_broad_kernel_and_memorising_presets():
    near(m("ocsvm", "auc", WIDE), 1.00, 0.01)
    near(m("ocsvm", "f1", WIDE), 0.95)
    near(m("ocsvm", "recall", WIDE), 0.95)
    assert m("ocsvm", "false_alarm", WIDE) < 0.01
    g8 = S(gamma_factor=8.0)
    near(m("ocsvm", "auc", g8), 0.83, 0.012)
    near(m("ocsvm", "f1", g8), 0.00, 0.01)
    near(m("ocsvm", "recall", g8), 0.00, 0.01)
    near(support_share(g8), 0.43, 0.02)
    near(oracle("ocsvm", g8), 0.20, 0.03)


def test_noise_curved_gap_and_many_anomalies_presets():
    near(m("ocsvm", "f1", WIDE, n_noise=40), 0.92)
    near(m("ocsvm", "recall", WIDE, n_noise=40), 0.90)
    assert m("ocsvm", "false_alarm", WIDE, n_noise=40) < 0.02
    near(m("ocsvm", "f1", curvature=1.0), 0.85)
    near(m("iforest", "f1", curvature=1.0), 0.95)
    near(m("ocsvm", "recall", n_modes=3, kind="gap"), 0.01, 0.02)
    near(m("ocsvm", "f1", n_modes=3, kind="gap"), 0.01, 0.02)
    for g in (0.02, 0.1, 0.25):
        near(m("ocsvm", "auc", S(gamma_factor=g), n_modes=3, kind="gap"), 0.00, 0.02)
    near(m("ocsvm", "auc", S(nu=0.5), contamination=45), 0.99, 0.015)
    near(m("ocsvm", "f1", S(nu=0.5), contamination=45), 0.93)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario,svm_auc,lof,iforest,robust,ecod", [("Standardfall", 0.95, 1.00, 1.00, 1.00, 1.00), ("gekrümmter Normalbereich", 0.98, 1.00, 1.00, 1.00, 1.00), ("dichte Gruppe 30 %", 0.74, 0.47, 0.70, 0.49, 0.84),
                                                                     ("Lücke, 2 Betriebsarten", 0.64, 0.53, 0.54, 0.40, 0.04), ("Lücke, 3 Betriebsarten", 0.82, 0.36, 0.27, 0.38, 0.01),
                                                                     ("Korrelationsbruch", 0.88, 0.99, 0.80, 1.00, 0.56), ("45 % verstreut", 0.32, 0.64, 1.00, 0.87, 1.00), ("40 Rauschmerkmale", 0.91, 0.99, 0.99, 0.88, 0.97)])
def test_scenario_table_auc(scenario, svm_auc, lof, iforest, robust, ecod):
    kw = dict(dict(ev.SCENARIOS)[scenario])
    near(m("ocsvm", "auc", **kw), svm_auc, 0.02)
    near(m("lof", "auc", **kw), lof, 0.02)
    near(m("iforest", "auc", **kw), iforest, 0.02)
    near(m("robust", "auc", **kw), robust, 0.02)
    near(m("ecod", "auc", **kw), ecod, 0.02)


def test_scenario_f1_statements():
    near(oracle("ocsvm"), 0.67)
    near(oracle("lof"), 0.97)
    near(oracle("iforest"), 0.97)
    near(m("lof", "f1", curvature=1.0), 0.74)
    near(m("robust", "f1", curvature=1.0), 0.38)


def test_grid_statements():
    g = grid()
    for gamma in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        near(g[(0.2, gamma)]["auc"], 1.00, 0.01)
    for gamma in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        near(g[(0.3, gamma)]["auc"], 1.00, 0.01)
    for gamma in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        assert 0.66 <= g[(0.2, gamma)]["f1"] <= 0.72 and 0.49 <= g[(0.3, gamma)]["f1"] <= 0.52
    for gamma in (0.02, 0.05, 0.1, 0.25):
        assert 0.89 <= g[(0.1, gamma)]["f1"] <= 0.97
    near(g[(0.2, 8.0)]["f1"], 0.87, 0.02)
    near(g[(0.3, 16.0)]["f1"], 0.87, 0.02)
    assert sorted(k for k, c in g.items() if c["f1"] >= 0.85) == [(0.1, 0.02), (0.1, 0.05), (0.1, 0.1), (0.1, 0.25), (0.2, 8.0), (0.3, 16.0)]
    for gamma, auc in zip((0.02, 0.05, 0.1, 0.25, 0.5, 1.0), (1.00, 1.00, 0.98, 0.40, 0.60, 0.89)):
        near(g[(0.02, gamma)]["auc"], auc, 0.03)


@pytest.mark.parametrize("nu,pure_flagged,pure_support,std_flagged,std_support,std_fa", [(0.02, 0.000, 0.123, 0.000, 0.155, 0.000), (0.05, 0.019, 0.130, 0.001, 0.155, 0.000), (0.10, 0.063, 0.157, 0.054, 0.167, 0.002),
                                                                                         (0.20, 0.169, 0.233, 0.183, 0.213, 0.093), (0.30, 0.279, 0.326, 0.289, 0.308, 0.210), (0.50, 0.487, 0.513, 0.495, 0.506, 0.439)])
def test_nu_calibration_table_and_the_bounds_of_the_theory(nu, pure_flagged, pure_support, std_flagged, std_support, std_fa):
    r = nu_tab()[nu]
    near(r["pure_flagged"], pure_flagged, 0.01)
    near(r["pure_support"], pure_support, 0.01)
    near(r["std_flagged"], std_flagged, 0.01)
    near(r["std_support"], std_support, 0.01)
    near(r["std_false_alarm"], std_fa, 0.01)
    assert r["pure_flagged"] <= nu + 1e-9 <= r["pure_support"] + 2e-9 and r["std_flagged"] <= nu + 1e-9 <= r["std_support"] + 2e-9
    if nu == 0.2:
        near(r["pure_false_alarm"], 0.16, 0.01)


def test_gap_over_gamma():
    g = gap_tab()
    for gamma in (0.02, 0.05, 0.1, 0.25):
        near(g[(2, 0.1)][gamma], 0.02, 0.02)
        near(g[(3, 0.1)][gamma], 0.00, 0.02)
    near(g[(2, 0.1)][1.0], 0.64, 0.02)
    near(g[(3, 0.1)][1.0], 0.82, 0.02)
    near(g[(2, 0.3)][1.0], 0.46, 0.02)
    near(g[(3, 0.3)][1.0], 0.57, 0.02)
    assert g[(3, 0.1)][32.0] < 0.55 and g[(2, 0.1)][32.0] < 0.55


def test_standardisation_table():
    rows = {r["standardize"]: r for r in ev.standardise_table()}
    near(rows[True]["ocsvm_auc"], 0.95, 0.012)
    near(rows[False]["ocsvm_auc"], 0.79, 0.02)
    near(rows[True]["ocsvm_f1"], 0.67)
    near(rows[False]["ocsvm_f1"], 0.68)
    near(rows[True]["lof_auc"], 1.00, 0.01)
    near(rows[False]["lof_auc"], 0.87, 0.02)
    near(rows[True]["lof_f1"], 0.94)
    near(rows[False]["lof_f1"], 0.68)


def test_correlation_break_limits():
    near(m("ocsvm", "f1", kind="decorrelated"), 0.30)
    tuned = S(nu=0.3, gamma_factor=2.0)
    near(m("ocsvm", "auc", tuned, kind="decorrelated"), 0.98, 0.02)
    near(m("ocsvm", "f1", tuned, kind="decorrelated"), 0.57)


def test_pure_normal_share_flagged_at_nu_02():
    near(float(np.mean([(a.values["ocsvm"] > C.FLAG_EPS).mean() for a in runs(S(nu=0.2), contamination=1)])), 0.169, 0.01)


def test_cost_table():
    times = {(t["n"], t["p"]): t for t in ev.cost_table()["times"]}
    for t in times.values():
        assert t["ocsvm"] < 0.05 and t["iforest"] > 3 * t["ocsvm"]                                        # höchstens einige zehn ms; der Wald deutlich mehr
    assert times[(600, 12)]["ocsvm"] < 3 * times[(600, 12)]["lof"] + 0.01 and times[(600, 12)]["lof"] < 3 * times[(600, 12)]["ocsvm"] + 0.01           # etwa gleich teuer wie der LOF
    for n, mb in ((100, 0.08), (300, 0.72), (600, 2.88)):
        near(times[(n, 12)]["kernel_mb"], mb, 0.01)
    for n, it in ((100, 109), (300, 330), (600, 589)):
        assert 0.6 * it <= times[(n, 12)]["iterations"] <= 1.4 * it
    near(10_000 * 10_000 * 8 / 1e6, 800.0, 0.5)


def test_wrong_share_table_and_nu_cutoff_table():
    tt = ev.threshold_table(STANDARD)
    by_factor = {r["factor"]: r for r in tt["wrong_share"]}
    near(by_factor[0.5]["ocsvm_f1"], 0.64)
    near(by_factor[1.0]["ocsvm_f1"], 0.67)
    near(by_factor[2.0]["ocsvm_f1"], 0.63)
    near(by_factor[0.5]["lof_f1"], 0.67)
    near(by_factor[1.0]["lof_f1"], 0.97)
    for r, f1, recall in zip(tt["cutoff"], (0.00, 0.00, 0.03, 0.67, 0.71, 0.51, 0.34), (0.00, 0.00, 0.01, 0.52, 1.00, 1.00, 1.00)):
        near(r["ocsvm_f1"], f1, 0.015)
        near(r["ocsvm_recall"], recall, 0.015)


# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------


def test_limits_table_numbers():
    near(m("ocsvm", "auc", contamination=45), 0.32, 0.02)
    near(m("ocsvm", "auc", contamination=30), 0.56, 0.02)
    near(m("ocsvm", "auc", S(nu=0.5), contamination=45), 0.99, 0.015)
    near(m("ocsvm", "auc", S(nu=0.2)), 1.00, 0.01)
    near(m("ocsvm", "auc", ev.Settings(standardize=False)), 0.79, 0.02)
    near(m("ocsvm", "f1", WIDE), 0.95)
    near(m("ocsvm", "recall", n_modes=2, kind="gap"), 0.03, 0.02)
    near(m("ocsvm", "recall", n_modes=3, kind="gap"), 0.01, 0.02)
    for n in (20, 30, 50):
        near(m("ocsvm", "f1", n=n), 0.0, 0.01)


def test_dense_group_with_nu_at_the_share_helps_the_threshold_not_the_ranking():
    s = S(nu=0.3, gamma_factor=0.1)
    kw = dict(kind="cluster", contamination=30)
    near(m("ocsvm", "f1", s, **kw), 0.46)
    near(m("iforest", "f1", s, **kw), 0.24)
    near(m("robust", "f1", s, **kw), 0.07)
    near(m("lof", "f1", s, **kw), 0.00, 0.02)
    near(m("ocsvm", "auc", s, **kw), 0.75, 0.02)
    near(m("ocsvm", "auc", **kw), 0.74, 0.02)


def test_isolation_forest_stays_high_over_the_curvature_range():
    for curv in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert 0.94 <= m("iforest", "f1", curvature=curv) <= 0.985
