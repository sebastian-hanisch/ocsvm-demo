"""Unabhängige Orakel für die One-Class SVM (SMO-Löser), den LOF, ECOD und die Kennzahlen der Auswertung.

- Duales Problem: Zielwert 1/2 a'Ka gegen libsvm (scikit-learn) UND gegen SLSQP (scipy) auf winzigen Instanzen, Nebenbedingungen
  (Summe = nu n, Schranken), KKT-Bedingungen, Entscheidungsfunktion gegen scikit-learn, die nu-Eigenschaft (Anteil f < 0 <= nu <= Anteil Stützvektoren).
- LOF gegen sklearn.LocalOutlierFactor (transduktiv und novelty); ECOD gegen direkte Schleifen über die Definition (ECDF links/rechts) und scipy.stats.skew.
- Kennzahlen (AUC, mittlere Präzision, ROC-Kurve) gegen scikit-learn - auch bei Bindungen im Score.
- Die aus den Vorgängern kopierten Bausteine (Isolation Forest, MCD) gegen die exakte erwartete Pfadlänge bzw. vollständige Aufzählung.
"""

import itertools
import math

import numpy as np
import pytest

import ocsvm_algorithm as svm
import ocsvm_ecod as ecod
import ocsvm_ee_algorithm as alg
import ocsvm_evaluation as ev
import ocsvm_isolation_forest as isf
import ocsvm_lof as lof

sk_svm = pytest.importorskip("sklearn.svm")
sk_nb = pytest.importorskip("sklearn.neighbors")
metrics = pytest.importorskip("sklearn.metrics")
optimize = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")


def _data(rng, n, p):
    X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) * rng.uniform(0.5, 2)
    X[: n // 5] += rng.normal(size=p) * 4
    return X


def test_dual_solution_against_libsvm_slsqp_kkt_and_nu_property():
    rng = np.random.default_rng(1)
    checked = 0
    for it in range(60):
        n, p = int(rng.integers(8, 70)), int(rng.integers(1, 6))
        X = _data(rng, n, p)
        nu = float(rng.choice([rng.uniform(0.02, 0.9), 0.05, 0.1, 0.2, 0.5, 1.0]))
        gamma = float(rng.choice([0.01, 0.1, 1.0, 5.0])) / p
        fit = svm.fit_ocsvm(X, nu, gamma)
        a, K = fit.alpha, fit.kernel
        assert fit.converged
        assert a.sum() == pytest.approx(nu * n, abs=1e-8) and a.min() >= -1e-12 and a.max() <= 1 + 1e-12
        f = svm.train_decision(fit)
        assert np.all(f[a <= 1e-9] >= -1e-4) and np.all(f[a >= 1 - 1e-9] <= 1e-4)
        assert np.all(np.abs(f[(a > 1e-9) & (a < 1 - 1e-9)]) <= 1e-4)                       # KKT
        assert (f < -1e-5).mean() <= nu + 1e-9 <= (a > 1e-9).mean() + 2e-9                   # nu-Eigenschaft
        obj = 0.5 * a @ K @ a
        try:
            sk = sk_svm.OneClassSVM(kernel="rbf", gamma=gamma, nu=nu, tol=1e-7).fit(X)
        except ValueError:
            continue                                                                         # libsvm bricht bei extremen Daten selbst ab
        ska = np.zeros(n)
        ska[sk.support_] = sk.dual_coef_.ravel()
        assert obj <= 0.5 * ska @ K @ ska + 1e-7 * max(1.0, obj)                             # nie schlechter als libsvm
        assert obj == pytest.approx(0.5 * ska @ K @ ska, rel=1e-6)
        Z = _data(rng, 12, p)
        assert np.abs(svm.decision_function(fit, Z) - sk.decision_function(Z)).max() < 5e-3
        if n <= 12 and nu < 1:
            res = optimize.minimize(lambda x: 0.5 * x @ K @ x, np.full(n, nu), jac=lambda x: K @ x, bounds=[(0, 1)] * n,
                                    constraints=[{"type": "eq", "fun": lambda x: x.sum() - nu * n}], method="SLSQP", options=dict(ftol=1e-14, maxiter=500))
            assert res.fun == pytest.approx(obj, abs=1e-7)
        checked += 1
    assert checked >= 30


def test_kernel_and_gamma_scale_against_scikit_learn():
    rng = np.random.default_rng(0)
    pw = pytest.importorskip("sklearn.metrics.pairwise")
    for _ in range(30):
        n, p = int(rng.integers(3, 30)), int(rng.integers(1, 6))
        X, Y = _data(rng, n, p), _data(rng, 7, p)
        g = float(rng.uniform(0.01, 3))
        assert np.allclose(svm.rbf_kernel(X, Y, g), pw.rbf_kernel(X, Y, gamma=g), atol=1e-12)
        assert svm.gamma_scale(X) == pytest.approx(sk_svm.OneClassSVM(gamma="scale").fit(X)._gamma, rel=1e-12)


def test_lof_against_scikit_learn():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n, p = int(rng.integers(6, 80)), int(rng.integers(1, 6))
        X = _data(rng, n, p)
        k = int(rng.integers(1, n))
        fit = lof.fit_lof(X, k)
        sk = sk_nb.LocalOutlierFactor(n_neighbors=min(k, n - 1)).fit(X)
        assert np.allclose(fit.lof, -sk.negative_outlier_factor_, rtol=1e-6, atol=1e-6)
        Q = _data(rng, 8, p)
        skn = sk_nb.LocalOutlierFactor(n_neighbors=min(k, n - 1), novelty=True).fit(X)
        assert np.allclose(lof.score_new(X, fit, Q), -skn.score_samples(Q), rtol=1e-6, atol=1e-6)


def test_ecod_against_the_definition():
    rng = np.random.default_rng(3)
    for it in range(30):
        n, p = int(rng.integers(5, 40)), int(rng.integers(1, 5))
        X = _data(rng, n, p)
        if it % 3 == 0:
            X = np.round(X, 1)                                                               # Bindungen
        left = np.array([[np.mean(X[:, j] <= X[i, j]) for j in range(p)] for i in range(n)])
        right = np.array([[np.mean(X[:, j] >= X[i, j]) for j in range(p)] for i in range(n)])
        sk = stats.skew(X, axis=0)
        assert np.allclose(ecod.skewness(X), sk)
        ul, ur = -np.log(left), -np.log(right)
        paper = np.maximum(np.maximum(ul.sum(axis=1), ur.sum(axis=1)), np.where(sk < 0, ul, ur).sum(axis=1))
        assert np.allclose(ecod.score(X, "paper"), paper)
        assert np.allclose(ecod.score(X, "both"), np.maximum(ul, ur).sum(axis=1))
        assert np.allclose(ecod.score(X, "twosided"), -np.log(np.minimum(1.0, 2.0 * np.minimum(left, right))).sum(axis=1))


def test_metrics_against_scikit_learn_including_tied_scores():
    rng = np.random.default_rng(1)
    for it in range(100):
        n = int(rng.integers(6, 80))
        s = rng.normal(size=n)
        if it % 2 == 0:
            s = np.round(s, 1)
        y = rng.random(n) < rng.uniform(0.1, 0.6)
        if y.all() or not y.any():
            continue
        assert ev.roc_auc(s, y) == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-12)
        assert ev.average_precision(s, y) == pytest.approx(metrics.average_precision_score(y, s), abs=1e-12)
        fpr, tpr = ev.roc_curve(s, y)
        f2, t2, _ = metrics.roc_curve(y, s, drop_intermediate=False)
        assert np.trapezoid(tpr, fpr) == pytest.approx(np.trapezoid(t2, f2), abs=1e-12)
    s, y = np.array([1.0, 1.0, 1.0, 1.0]), np.array([True, False, False, False])
    assert ev.average_precision(s, y) == pytest.approx(0.25) == pytest.approx(ev.average_precision(s[::-1], y[::-1]))


def _c(m):
    return 0.0 if m <= 1 else (1.0 if m == 2 else 2 * (math.log(m - 1) + 0.5772156649015329) - 2 * (m - 1) / m)


def _expected_path(S, depth, limit):
    m, p = S.shape
    valid = [f for f in range(p) if S[:, f].max() > S[:, f].min()]
    if m == 1 or depth == limit or not valid:
        return np.full(m, depth + _c(m))
    out = np.zeros(m)
    for f in valid:
        order = np.argsort(S[:, f])
        xs = S[order, f]
        for i in range(1, m):
            gap = xs[i] - xs[i - 1]
            if gap <= 0:
                continue
            res = np.zeros(m)
            res[order[:i]] = _expected_path(S[order[:i]], depth + 1, limit)
            res[order[i:]] = _expected_path(S[order[i:]], depth + 1, limit)
            out += gap / (xs[-1] - xs[0]) * res / len(valid)
    return out


def test_copied_isolation_forest_matches_the_exact_expected_path_length():
    S = np.random.default_rng(1).normal(size=(6, 2))
    limit = isf.height_limit(6)
    rng = np.random.default_rng(101)
    mean = np.mean([isf.tree_path_length(isf.build_tree(S, rng, limit), S) for _ in range(6000)], axis=0)
    assert np.abs(mean - _expected_path(S, 0, limit)).max() < 0.05


def test_copied_mcd_finds_the_exhaustive_optimum_on_tiny_data():
    rng = np.random.default_rng(4)
    for it in range(12):
        n, p = int(rng.integers(8, 11)), int(rng.integers(1, 3))
        X = rng.normal(size=(n, p))
        X[: int(rng.integers(1, 4))] += rng.uniform(4, 10)
        h = alg.subset_size(n, p, 0.5)

        def logdet(sub):
            return np.linalg.slogdet(np.atleast_2d(np.cov(X[list(sub)].T)))[1]

        best = min(logdet(sub) for sub in itertools.combinations(range(n), h))
        fit = alg.fit_mcd(X, 0.5, it, reweight=False)
        assert best - 1e-9 <= logdet(fit.support) <= best + 1e-6
