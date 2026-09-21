"""One-Class SVM: Handinstanzen (zwei und drei Punkte, von Hand gelöst), Dualbedingungen und KKT, die Schranken für ν, Kernel-Eigenschaften, Grenzfälle von γ, Kreuzprüfung gegen scikit-learn (libsvm);
die kopierten Komponenten der Vorgänger (LOF, Isolation Forest, χ², MCD, ECOD)."""

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.svm import OneClassSVM

import ocsvm_algorithm as svm
import ocsvm_constants as C
import ocsvm_ecod as ecod
import ocsvm_ee_algorithm as alg
import ocsvm_isolation_forest as isf
import ocsvm_lof as lof


def _data(n=200, p=4, n_out=10, shift=5.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.standard_normal((n - n_out, p)), shift + 0.5 * rng.standard_normal((n_out, p))])
    return X, np.arange(n) >= n - n_out


# --- Handinstanzen ------------------------------------------------------------------------------------------------------------------------


def test_two_points_with_nu_one_have_alpha_one_and_rho_one_plus_kernel():
    X = np.array([[0.0], [1.0]])
    f = svm.fit_ocsvm(X, 1.0, 1.0)
    k = np.exp(-1.0)
    assert np.allclose(f.alpha, [1.0, 1.0]) and f.rho == pytest.approx(1.0 + k) and f.converged
    assert np.allclose(svm.train_decision(f), 0.0, atol=1e-12)                      # beide liegen genau auf der Grenze


def test_three_symmetric_points_with_nu_one_third_match_the_closed_form():
    """Punkte -1, 0, 1, ν n = 1: alpha = (a, 1 - 2a, a) mit a = (1 - k1) / (3 - 4 k1 + k2), k1 = exp(-γ), k2 = exp(-4γ); alle drei sind freie Stützvektoren, also f = 0 an allen."""
    X = np.array([[-1.0], [0.0], [1.0]])
    for gamma in (1.0, 2.0, 3.0):
        k1, k2 = np.exp(-gamma), np.exp(-4.0 * gamma)
        a = (1.0 - k1) / (3.0 - 4.0 * k1 + k2)
        f = svm.fit_ocsvm(X, 1.0 / 3.0, gamma, tol=1e-12)
        assert np.allclose(f.alpha, [a, 1 - 2 * a, a], atol=1e-8) and np.allclose(svm.train_decision(f), 0.0, atol=1e-8)
        assert f.rho == pytest.approx(a * (1 + k2) + (1 - 2 * a) * k1, abs=1e-8)


def test_decision_function_far_away_is_minus_rho_and_inside_is_positive():
    X, _ = _data(n=100, n_out=0)
    f = svm.fit_ocsvm(X, 0.1, svm.gamma_scale(X))
    far = svm.decision_function(f, np.full((1, 4), 50.0))
    assert far[0] == pytest.approx(-f.rho, abs=1e-9)
    assert svm.decision_function(f, X.mean(axis=0, keepdims=True))[0] > 0
    assert np.allclose(svm.score(f), -svm.train_decision(f)) and np.allclose(svm.score(f, X[:5]), -svm.decision_function(f, X[:5]))


# --- Dualbedingungen, KKT und die Schranken von ν ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nu,gmul", [(0.05, 0.1), (0.1, 1.0), (0.3, 2.0), (0.5, 8.0), (0.02, 0.25)])
def test_dual_constraints_and_kkt_conditions(nu, gmul):
    X, _ = _data()
    f = svm.fit_ocsvm(X, nu, gmul * svm.gamma_scale(X))
    assert f.converged and f.gap < C.SMO_TOL
    assert f.alpha.sum() == pytest.approx(nu * len(X), abs=1e-9) and (f.alpha >= -1e-12).all() and (f.alpha <= 1.0 + 1e-12).all()
    G = f.kernel @ f.alpha
    tol = 1e-4
    free = (f.alpha > 1e-9) & (f.alpha < 1 - 1e-9)
    assert (np.abs(G[free] - f.rho) < tol).all()                                   # freie Stützvektoren: f = 0
    assert (G[f.alpha <= 1e-9] >= f.rho - tol).all()                               # Nicht-Stützvektoren: f >= 0
    assert (G[f.alpha >= 1 - 1e-9] <= f.rho + tol).all()                           # obere Schranke: f <= 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("nu", [0.02, 0.1, 0.3, 0.5])
def test_nu_is_an_upper_bound_of_the_flagged_share_and_a_lower_bound_of_the_support_share(seed, nu):
    X, _ = _data(n=150, seed=seed)
    f = svm.fit_ocsvm(X, nu, svm.gamma_scale(X) * (0.25 + seed))
    flagged = float((svm.train_decision(f) < -C.FLAG_EPS).mean())
    support = f.n_support / len(X)
    assert flagged <= nu + 1e-9 and support >= nu - 1e-9


def test_invalid_nu_is_rejected():
    X, _ = _data(n=20, n_out=0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            svm.fit_ocsvm(X, bad, 1.0)


# --- Kernel und Grenzfälle von γ ----------------------------------------------------------------------------------------------------------------


def test_rbf_kernel_properties():
    X, _ = _data(n=60)
    K = svm.rbf_kernel(X, X, 0.3)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1.0) and (K > 0).all() and (K <= 1.0).all()
    assert np.linalg.eigvalsh(K).min() > -1e-9                                     # positiv semidefinit
    assert svm.rbf_kernel(X[:1], X[1:2], 0.3)[0, 0] == pytest.approx(np.exp(-0.3 * ((X[0] - X[1]) ** 2).sum()))


def test_gamma_scale_is_one_over_p_for_standardised_data():
    X, _ = _data(n=300)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    assert svm.gamma_scale(Z) == pytest.approx(1.0 / 4, rel=0.02) and svm.gamma_scale(3.0 * X) < svm.gamma_scale(X)


def test_huge_gamma_makes_every_tour_a_support_vector_with_equal_weight():
    X, _ = _data(n=80, n_out=0)
    f = svm.fit_ocsvm(X, 0.2, 1e4)
    assert f.n_support == len(X) and np.allclose(f.alpha, 0.2, atol=1e-6)


def test_tiny_gamma_ranks_by_distance_from_the_centre():
    """Bei sehr breitem Kernel ist die Entscheidungsfunktion fast eine umgekehrte Parabel: der Wert steigt mit dem Abstand vom Zentrum."""
    from scipy.stats import spearmanr
    X, _ = _data(n=200, p=3, n_out=0)
    f = svm.fit_ocsvm(X, 0.2, 1e-3)
    d = np.linalg.norm(X - X.mean(axis=0), axis=1)
    assert spearmanr(svm.score(f), d)[0] > 0.9


def test_deterministic():
    X, _ = _data()
    a = svm.fit_ocsvm(X, 0.1, 0.05)
    b = svm.fit_ocsvm(X, 0.1, 0.05)
    assert np.array_equal(a.alpha, b.alpha) and a.rho == b.rho and a.iterations == b.iterations


def test_the_solver_reports_when_the_iteration_limit_is_hit():
    X, _ = _data()
    f = svm.fit_ocsvm(X, 0.3, 0.5, max_iter=3)
    assert not f.converged and f.iterations == 3


# --- Kreuzprüfung gegen scikit-learn -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,p,nu,gmul", [(50, 3, 0.1, 1.0), (300, 12, 0.1, 1.0), (300, 12, 0.3, 8.0), (300, 12, 0.05, 0.1), (120, 6, 0.5, 2.0), (200, 4, 0.02, 4.0)])
def test_matches_scikit_learn(n, p, nu, gmul):
    rng = np.random.default_rng(n + p)
    X = rng.standard_normal((n, p))
    X[: n // 10] += 4
    g = svm.gamma_scale(X) * gmul
    f = svm.fit_ocsvm(X, nu, g, tol=1e-9)
    m = OneClassSVM(kernel="rbf", gamma=g, nu=nu, tol=1e-9).fit(X)
    Z = rng.standard_normal((100, p))
    assert np.abs(svm.decision_function(f, Z) - m.decision_function(Z)).max() < 1e-6
    a = np.zeros(n)
    a[m.support_] = m.dual_coef_[0]
    assert np.abs(f.alpha - a).max() < 1e-5 and f.rho == pytest.approx(float(m.offset_[0]), abs=1e-6) and f.n_support == len(m.support_)


# --- Kopierte Komponenten der Vorgänger -----------------------------------------------------------------------------------------------------


def test_copied_components_still_behave_like_their_predecessors():
    rng = np.random.default_rng(8)
    X = np.concatenate([rng.standard_normal((200, 3)), 7 + rng.standard_normal((10, 3))])
    y = np.arange(210) >= 200
    f = lof.fit_lof(X, 20)
    assert f.lof[y].min() > 1.5 and np.median(f.lof[~y]) < 1.2
    forest = isf.fit_forest(X, 100, 128, 0)
    sc = isf.score_from_paths(isf.path_lengths(forest, X), forest.psi)
    assert sc[y].min() > sc[~y].mean() and (sc >= 0).all() and (sc <= 1).all()
    r = alg.fit_mcd(X, 0.75, 0)
    assert r.d2[y].min() > alg.threshold(r, 0.975)
    assert alg.chi2_ppf(0.975, 12) == pytest.approx(chi2.ppf(0.975, 12), rel=1e-6)
    left, right = ecod.tail_probabilities(np.array([[1.0], [2.0], [2.0], [3.0], [10.0]]))
    assert np.allclose(left[:, 0], [0.2, 0.6, 0.6, 0.8, 1.0]) and np.allclose(right[:, 0], [1.0, 0.8, 0.8, 0.4, 0.2])
