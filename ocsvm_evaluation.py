"""Auswertung der One-Class-SVM-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für die One-Class SVM, den LOF,
den Isolation Forest und die robuste Schätzung der Wurzel (dazu ECOD als Kontrast), Sweeps, Experimente auf Abruf (Szenarien, γ × ν-Raster, ν-Kalibrierung, Lücke, Standardisierung, Kosten) und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import ocsvm_algorithm as svm
import ocsvm_ecod as ecod
import ocsvm_lof as lof
import ocsvm_isolation_forest as isf
import ocsvm_ee_algorithm as alg
import ocsvm_constants as C
import ocsvm_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------

# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


def standardise(X):
    """Kennzahlen auf Mittelwert 0 und Streuung 1 (One-Class SVM und LOF rechnen mit Abständen; der Isolation Forest und die robuste Schätzung brauchen das nicht)."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


@dataclass(frozen=True)
class Settings:
    nu: float = C.DEFAULT_NU                            # One-Class SVM: nu (obere Schranke der markierten Trainingspunkte, untere der Stützvektoren)
    gamma_factor: float = C.DEFAULT_GAMMA_FACTOR        # gamma als Vielfaches der Skala 1 / (p * Var) (standardisiert: 1 / p)
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": f < 0 (OCSVM), LOF 1.5, Score 0.5 (Isolation Forest), chi²-Quantil (robust); "share": der erwartete Anteil für alle vier
    quantile: float = C.DEFAULT_QUANTILE                # chi²-Quantil der Wurzel
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    standardize: bool = True                            # Merkmale vor der One-Class SVM und dem LOF standardisieren
    start: int = 0                                      # Seed des Isolation Forest und der MCD-Starts (die One-Class SVM ist deterministisch)


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)
DETECTORS = ("ocsvm", "lof", "iforest", "robust")
DETECTOR_NAMES = {"ocsvm": "One-Class SVM", "lof": "LOF", "iforest": "Isolation Forest", "robust": "robust (MCD)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")
IF_TREES, IF_PSI, IF_CUTOFF, LOF_CUTOFF, LOF_K = C.DEFAULT_TREES, C.DEFAULT_PSI, C.DEFAULT_CUTOFF_IF, C.DEFAULT_CUTOFF_LOF, C.DEFAULT_LOF_K


def lof_k(n):
    """k des LOF im Vergleich: 20, höchstens n / 2 (k nahe n macht LOF unbrauchbar)."""
    return int(max(3, min(LOF_K, n // 2)))


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise)


def fit_svm(X, settings):
    """One-Class SVM auf (standardisierten) Kennzahlen: (Fit, transformierte Daten). gamma = Faktor * 1 / (p * Var)."""
    Z = standardise(X) if settings.standardize else X
    return svm.fit_ocsvm(Z, settings.nu, settings.gamma_factor * svm.gamma_scale(Z)), Z


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)
    p_total: int                  # Zahl der Merkmale einschließlich Rauschmerkmale
    fit: svm.Fit                  # die One-Class SVM
    forest_if: isf.Forest
    values: dict                  # Detektor -> Anomalie-Wert je Tour (One-Class SVM: -f, LOF, Isolation Forest: Score, robust: quadrierter Mahalanobis-Abstand)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: dict               # Detektor -> F1, wenn der wahre Anteil bekannt wäre (die k größten Werte)
    seconds: dict
    ecod_auc: float               # ECOD (zweiseitig) als Kontrast: AUC
    ecod_oracle_f1: float


_ROOT_CACHE = {}


def _root_fits(X, key, start):
    """Klassische und robuste Schätzung der Wurzel (unabhängig von den Reglern der One-Class SVM: werden je Aufnahme nur einmal gerechnet)."""
    cache_key = (key, start)
    if key is not None and cache_key in _ROOT_CACHE:
        return _ROOT_CACHE[cache_key]
    t0 = time.perf_counter()
    classical = alg.fit_classical(X)
    t1 = time.perf_counter()
    robust = alg.fit_mcd(X, C.DEFAULT_SUPPORT, start, reweight=C.DEFAULT_REWEIGHT)
    out = (classical, robust, t1 - t0, time.perf_counter() - t1)
    if key is not None:
        if len(_ROOT_CACHE) > 600:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[cache_key] = out
    return out


def _thresholds(values, settings, robust):
    """Markierung und Schwellenwert je Detektor: Standard = f < 0 (One-Class SVM: Score > 0), LOF 1.5, Score 0.5 (Isolation Forest), chi²-Quantil (robust), sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(values[d], settings.share / 100.0)
            thr[d] = float(np.sort(values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["ocsvm"] = C.FLAG_EPS
        thr["lof"] = LOF_CUTOFF
        thr["iforest"] = IF_CUTOFF
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None, root_key=None):
    secs = {}
    X = ds.X
    p_total = X.shape[1]
    t0 = time.perf_counter()
    fit, Z = fit_svm(X, settings)
    secs["ocsvm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lof_values = lof.fit_lof(standardise(X) if settings.standardize else X, lof_k(len(X))).lof
    secs["lof"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    forest_if = isf.fit_forest(X, IF_TREES, min(IF_PSI, len(X)), settings.start)
    paths_if = isf.path_lengths(forest_if, X)
    secs["iforest"] = time.perf_counter() - t0
    classical, robust, secs["classical"], secs["robust"] = _root_fits(X, root_key, settings.start)
    values = {"ocsvm": svm.score(fit), "lof": lof_values, "iforest": isf.score_from_paths(paths_if, forest_if.psi), "robust": robust.d2}
    flags, thr = _thresholds(values, settings, robust)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    m_sv = {"n_support": fit.n_support, "n_bound": fit.n_bound, "iterations": fit.iterations, "converged": fit.converged}
    scores["ocsvm"].update(m_sv)
    oracle = {d: flag_metrics(isf.flag_top(values[d], ds.anomaly.mean()), ds.anomaly)["f1"] for d in DETECTORS}
    e = ecod.score(X, "twosided")
    return Analysis(ds, settings, params, p_total, fit, forest_if, values, classical, robust, scores, flags, oracle, secs, roc_auc(e, ds.anomaly), flag_metrics(isf.flag_top(e, ds.anomaly.mean()), ds.anomaly)["f1"])


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params, root_key=params)


def score_maps(X2, settings=Settings(), grid=80, pad=0.5):
    """Entscheidungsfunktion der One-Class SVM und der Isolation-Forest-Score auf einem Raster für zweidimensionale Kennzahlen (Darstellung der gelernten Grenze): (xs, ys, F_svm [grid, grid], S_if [grid, grid], Stützvektoren-Maske [n]).
    Die Kennzahlen werden standardisiert, das Raster liegt in den Originaleinheiten."""
    X2 = np.asarray(X2, dtype=float)
    lo, hi = X2.min(axis=0), X2.max(axis=0)
    span = hi - lo
    xs = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], grid)
    ys = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    mu, sd = X2.mean(axis=0), np.where(X2.std(axis=0) < 1e-12, 1.0, X2.std(axis=0))
    Z = (X2 - mu) / sd
    fit = svm.fit_ocsvm(Z, settings.nu, settings.gamma_factor * svm.gamma_scale(Z))
    F = svm.decision_function(fit, (pts - mu) / sd).reshape(grid, grid)
    forest = isf.fit_forest(X2, IF_TREES, min(IF_PSI, len(X2)), settings.start)
    S = isf.score_from_paths(isf.path_lengths(forest, pts), forest.psi).reshape(grid, grid)
    return xs, ys, F, S, fit.alpha > 1e-12


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "nu": (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
    "gamma_factor": C.GAMMA_FACTORS,
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
}
SWEEP_LABELS = {"n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen",
                "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "nu": "ν der One-Class SVM", "gamma_factor": "γ der One-Class SVM (Vielfaches von γ₀)",
                "quantile": "chi²-Quantil der Schwelle (robust)", "share": "angenommener Anteil der Anomalien [%]"}
SETTING_PARAMETERS = ("nu", "gamma_factor", "quantile", "share")


def _record(a):
    out = {f"{d}_{k}": a.scores[d][k] for d in DETECTORS for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    out["ocsvm_support_share"] = a.fit.n_support / a.ds.n
    out["ocsvm_bound_share"] = a.fit.n_bound / a.ds.n
    out["ocsvm_iterations"] = float(a.fit.iterations)
    out["ecod_auc"] = a.ecod_auc
    out["ecod_oracle_f1"] = a.ecod_oracle_f1
    for d in DETECTORS:
        out[f"{d}_oracle_f1"] = a.oracle_f1[d]
        out[f"{d}_seconds"] = a.seconds[d]
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _analyse_seed(seed, settings, kw):
    data = {**DEFAULT_DATA, **kw}
    ds = make_dataset(seed=seed, **data)
    return analyse(ds, settings, None, root_key=(tuple(sorted(data.items())), seed))


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(_analyse_seed(s, settings, kw)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der vier Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


SCENARIOS = (
    ("Standardfall", {}),
    ("gekrümmter Normalbereich", {"curvature": 1.0}),
    ("dichte Gruppe 30 %", {"kind": "cluster", "contamination": 30}),
    ("Lücke, 2 Betriebsarten", {"kind": "gap", "n_modes": 2}),
    ("Lücke, 3 Betriebsarten", {"kind": "gap", "n_modes": 3}),
    ("Korrelationsbruch", {"kind": "decorrelated"}),
    ("45 % verstreut", {"contamination": 45}),
    ("40 Rauschmerkmale", {"n_noise": 40}),
)


def scenario_table(settings=Settings(), scenarios=SCENARIOS, **base):
    """Die Detektoren in den Szenarien: AUC, F1 und F1 mit bekanntem Anteil (Mittel über die festen Sweep-Datensätze), dazu ECOD (zweiseitig) als Kontrast."""
    rows = []
    for label, extra in scenarios:
        row = _mean_over_seeds(settings, **{**base, **extra})
        row["scenario"] = label
        rows.append(row)
    return rows


def _ocsvm_only(seed, settings, kw):
    """Aufnahme, One-Class SVM und Wahrheit ohne die Vergleichsdetektoren: (Score, Anomalie-Maske, Fit)."""
    ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **kw})
    fit, _ = fit_svm(ds.X, settings)
    return svm.score(fit), ds.anomaly, fit


def grid_table(nus=(0.02, 0.05, 0.10, 0.20, 0.30), gammas=C.GAMMA_FACTORS, **base):
    """γ × ν-Raster der One-Class SVM: AUC, F1 an der Schwelle f < 0 und Anteil der Stützvektoren (Mittel über die festen Sweep-Datensätze)."""
    cells = []
    for nu in nus:
        for g in gammas:
            st = Settings(nu=nu, gamma_factor=g)
            aucs, f1s, svs, fas = [], [], [], []
            for seed in C.SWEEP_SEEDS:
                s, y, fit = _ocsvm_only(seed, st, base)
                aucs.append(roc_auc(s, y))
                m = flag_metrics(s > C.FLAG_EPS, y)
                f1s.append(m["f1"])
                fas.append(m["false_alarm"])
                svs.append(fit.n_support / len(y))
            cells.append({"nu": nu, "gamma_factor": g, "auc": float(np.mean(aucs)), "f1": float(np.mean(f1s)), "false_alarm": float(np.mean(fas)), "support_share": float(np.mean(svs))})
    return cells


def nu_table(nus=(0.02, 0.05, 0.10, 0.20, 0.30, 0.50), **base):
    """ν-Kalibrierung: Anteil der markierten Trainingspunkte und der Stützvektoren bei (fast) reinen Normalen (1 % Anomalien) und bei den 10 % des Standardfalls; ν soll obere Schranke des ersten und untere Schranke des zweiten sein."""
    rows = []
    for nu in nus:
        row = {"nu": nu}
        for label, cont in (("pure", 1), ("std", DEFAULT_DATA["contamination"])):
            flagged, svs, fas = [], [], []
            for seed in C.SWEEP_SEEDS:
                s, y, fit = _ocsvm_only(seed, Settings(nu=nu), {**base, "contamination": cont})
                flagged.append(float((s > C.FLAG_EPS).mean()))
                svs.append(fit.n_support / len(y))
                fas.append(float((s[~y] > C.FLAG_EPS).mean()))
            row[f"{label}_flagged"], row[f"{label}_support"], row[f"{label}_false_alarm"] = float(np.mean(flagged)), float(np.mean(svs)), float(np.mean(fas))
        rows.append(row)
    return rows


def gap_table(gammas=C.GAMMA_FACTORS, **base):
    """Lücke zwischen den Betriebsarten: AUC der One-Class SVM in Abhängigkeit von γ für 2 und 3 Betriebsarten (ν = 0.1 und ν = 0.3), dazu der LOF und der Isolation Forest als Bezug (feste Werte)."""
    rows = []
    for n_modes in (2, 3):
        for nu in (0.10, 0.30):
            aucs = {}
            for g in gammas:
                aucs[g] = float(np.mean([roc_auc(*_ocsvm_only(seed, Settings(nu=nu, gamma_factor=g), {**base, "kind": "gap", "n_modes": n_modes})[:2]) for seed in C.SWEEP_SEEDS]))
            rows.append({"n_modes": n_modes, "nu": nu, "by_gamma": aucs})
    return rows


def standardise_table(**base):
    """One-Class SVM und LOF mit und ohne Standardisierung (Mittel über die festen Sweep-Datensätze)."""
    rows = []
    for std in (True, False):
        r = _mean_over_seeds(Settings(standardize=std), **base)
        rows.append({"standardize": std, "ocsvm_auc": r["ocsvm_auc"], "ocsvm_f1": r["ocsvm_f1"], "lof_auc": r["lof_auc"], "lof_f1": r["lof_f1"]})
    return rows


def cost_table(settings=Settings(), **base):
    """Rechenzeit der One-Class SVM, des LOF und des Isolation Forest über die Tourenzahl bei 12 und bei 52 Merkmalen; dazu Speicher der Kernmatrix (n² Zahlen) und Iterationen des Lösers."""
    times = []
    for n_noise in (0, 40):
        for n in (100, 300, 600):
            rows = [_analyse_seed(seed, settings, {**base, "n": n, "n_noise": n_noise}) for seed in C.SWEEP_SEEDS[:3]]
            times.append({"n": n, "p": 12 + n_noise, "kernel_mb": n * n * 8 / 1e6, "iterations": float(np.mean([a.fit.iterations for a in rows])),
                          **{d: float(np.mean([a.seconds[d] for a in rows])) for d in ("ocsvm", "lof", "iforest")}})
    return {"times": times}


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen über ν (Schwelle f < 0); (2) F1 aller vier Detektoren bei ½-, 1- und 2-fach angenommenem Anteil."""
    cut = sweep("nu", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": int(round(true_share * factor))}), **base)
        row["x"], row["factor"] = int(round(true_share * factor)), factor
        wrong.append(row)
    return {"cutoff": cut, "wrong_share": wrong}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem ein Detektor als besser gilt
GAP_AUC = 0.8
RANK_AUC = 0.6               # ab dieser AUC gilt die Lücke in der Rangfolge als gesehen
F1_WIN_MARGIN = 0.10          # F1-Abstand an der Schwelle, ab dem die One-Class SVM als besser gilt
THRESHOLD_F1_DROP = 0.15


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke (keiner findet sie) bzw. nur in der Rangfolge sichtbar, ν kleiner als der Anteil, anderer Detektor besser, One-Class SVM besser (in der Rangfolge oder an der Schwelle),
    falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds = a.ds
    e = a.scores["ocsvm"]
    others = {d: a.scores[d]["auc"] for d in ("lof", "iforest", "robust")}
    best_other = max(others.values())
    best_other_f1 = max(a.scores[d]["f1"] for d in ("lof", "iforest", "robust"))
    data = {"n": ds.n, "p": a.p_total, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": int(ds.anomaly.sum()), "nu": a.settings.nu,
            "gamma_factor": a.settings.gamma_factor, "best_other_auc": best_other, "best_other_f1": best_other_f1, "ecod_auc": a.ecod_auc, **{f"oracle_{d}": a.oracle_f1[d] for d in DETECTORS},
            **{f"{d}_{k}": v for d in DETECTORS for k, v in a.scores[d].items()}}
    if ds.kind == "gap":
        if e["auc"] >= RANK_AUC and best_other < RANK_AUC:
            return "warning", "gap_rank", data
        if max(best_other, e["auc"]) < GAP_AUC:
            return "warning", "gap", data
    if 100.0 * ds.anomaly.mean() > 100.0 * a.settings.nu and best_other - e["auc"] >= WIN_MARGIN:
        return "warning", "nu_low", data
    if best_other - e["auc"] >= WIN_MARGIN:
        return "warning", "others_win", data
    if e["auc"] - best_other >= WIN_MARGIN or (e["f1"] - best_other_f1 >= F1_WIN_MARGIN and e["auc"] >= best_other - 0.02):
        return "success", "ocsvm_wins", data
    if e["auc"] >= 0.95 and a.oracle_f1["ocsvm"] - e["f1"] > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    return "success", "comparable", data
