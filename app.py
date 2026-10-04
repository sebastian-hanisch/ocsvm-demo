"""One-Class SVM - eine gelernte Grenze um die normalen Touren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die One-Class SVM - und lässt stattdessen das Beispiel wachsen.
Siebtes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: ein eigener Ast direkt nach der Wurzel (Elliptic Envelope), neben LOF, Isolation Forest und ECOD. Die One-Class SVM lernt eine nichtlineare Grenze um die Normalen (Kernel-Trick) statt einer
angenommenen Ellipse - und braucht dafür zwei Regler, ν und γ. Gemessen wird, was das bringt - und was nicht. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ocsvm_algorithm as svm
import ocsvm_constants as C
from ocsvm_evaluation import (
    SWEEP_LABELS,
    SWEEP_VALUES,
    Settings,
    analyse_for,
    cost_table,
    gap_table,
    grid_table,
    nu_table,
    roc_curve,
    scenario_table,
    score_maps,
    standardise_table,
    sweep,
    threshold_table,
    verdict,
)
from ocsvm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from ocsvm_visualization import (
    build_boundary,
    build_costs,
    build_cutoff,
    build_features,
    build_gap,
    build_grid,
    build_kernel,
    build_method_bars,
    build_nu,
    build_rank_compare,
    build_roc,
    build_scatter,
    build_scenarios,
    build_score_hist,
    build_similarity,
    build_standardise,
    build_sweep,
    build_wrong_share,
    projection,
)

st.set_page_config(page_title="One-Class SVM – Sebastian Hanisch", layout="wide")
BLUE_TXT, RED_TXT = "#1f77b4", "#d62728"


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _maps(data_params, settings, i, j):
    ds_X = _analysis(data_params, settings).ds.X
    return score_maps(ds_X[:, [i, j]], settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings, values):
    return sweep(parameter, values=values, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _scenarios(settings):
    return scenario_table(settings)


@st.cache_data(show_spinner=False)
def _grid(base):
    return grid_table(**dict(base))


@st.cache_data(show_spinner=False)
def _nu_table(base):
    return nu_table(**dict(base))


@st.cache_data(show_spinner=False)
def _gap(base):
    return gap_table(**dict(base))


@st.cache_data(show_spinner=False)
def _standardise(base):
    return standardise_table(**dict(base))


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _costs(base, settings):
    return cost_table(settings, **dict(base))


st.title("🧭 One-Class SVM – eine gelernte Grenze um die Normalen")
st.markdown(
    """
Die Wurzel dieser Linie nimmt an, die normalen Touren bildeten **eine Gauß-Wolke** mit elliptischer Grenze. Die **One-Class SVM** lernt die Grenze stattdessen aus den Daten: ein **Kernel** misst, wie ähnlich zwei Touren sind, und die Methode sucht die kleinste Kuppel-Summe,
die einen vorgegebenen Anteil der Touren außen lässt - **nichtlinear**, ohne Verteilungsannahme. Der Preis: **zwei Regler**. **ν** legt fest, welchen Anteil der Trainingstouren die Grenze außen lassen darf (er ist eine obere Schranke für den markierten und eine untere für den
Stützvektor-Anteil), und **γ** die Breite der Kuppeln. Diese Demo misst, wie stark das Ergebnis an diesen beiden Reglern hängt - und wo die Grenze die Ellipse tatsächlich schlägt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - siebtes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario, LOF, Isolation Forest und die robuste Schätzung der Wurzel sind wortgleich aus den Vorgänger-Demos übernommen (die klassische Schätzung entfällt, damit die Balken lesbar bleiben); ECOD tritt nur als Kontrast auf (im Szenarien-Experiment; seine AUC steht auch in den Hilfetexten und im Lücken-Hinweis). "
    "Die Linie hat keinen Konvergenzpunkt; die One-Class SVM ist ein eigener Ast, ihr Nachfolger ist Deep SVDD (gelernte Merkmalsabbildung statt festem Kernel; eigene Demo)."
)

with st.expander("So funktioniert die One-Class SVM", expanded=True):
    st.markdown(
        """
1. **Kernel.** Zwei Touren mit standardisierten Kennzahlen $x, z$ haben die Ähnlichkeit $K(x, z) = \\exp(-\\gamma\\,\\lVert x - z\\rVert^2)$: 1 bei gleichen Touren, gegen 0 bei weit entfernten. Die Breite steuert $\\gamma$; die Voreinstellung ist $\\gamma_0 = 1 / p$ (wie in scikit-learn).
2. **Gelernte Grenze.** Gesucht sind Gewichte $\\alpha_i \\in [0, 1]$ mit Summe $\\nu n$, die $\\tfrac12 \\alpha^\\top K \\alpha$ minimieren. Die Entscheidungsfunktion $f(x) = \\sum_i \\alpha_i K(x_i, x) - \\rho$ ist positiv im dichten Innern und negativ außerhalb;
   Touren mit $\\alpha_i > 0$ heißen **Stützvektoren**, sie liegen am Rand oder außerhalb.
3. **Schwelle.** Eine Tour ist Anomalie, wenn $f < 0$. Damit gilt (Schölkopf u. a.): **höchstens ein Anteil ν der Trainingstouren wird markiert, und mindestens ein Anteil ν sind Stützvektoren.** Die Schwelle hat also den Anteil eingebaut - man muss ihn kennen, und zwar schon für die Rangfolge, weil die Grenze auf den Trainingsdaten gelernt wird.
4. **Löser.** Die Demo löst das duale Problem selbst (SMO, wie libsvm) und gleicht mit scikit-learn auf 1e-6 ab.

Was **nicht** vorausgesetzt wird: eine Verteilungsform, ein einziger konvexer Normalbereich. Was die Methode **voraussetzt**: passende Werte für ν (≥ Anteil der Anomalien) und γ (passend zur Struktur der Daten), standardisierte Merkmale und Platz für die $n \\times n$-Kernmatrix.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren (bei ν = 0.1 und γ₀). Die AUC der One-Class SVM ist bei 20 / 30 / 50 / 100 / 200 / 400 / 600 Touren 0.76 / 0.88 / 0.86 / 0.92 / 0.94 / 0.95 / 0.96, der F1 an der Schwelle f < 0 0.00 / 0.00 / 0.00 / 0.10 / 0.64 / 0.72 / 0.76: "
             "bei ν n ≤ 5 Touren (bis 50 Touren) markiert sie nichts. Der LOF hat 0.96 / 0.94 / 0.88 / 0.87 / 0.91 / 0.94 / 0.90, der Isolation Forest 0.60 / 0.67 / 0.78 / 0.90 / 0.96 / 0.96 / 0.95.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). AUC der One-Class SVM bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen: 0.90 / 0.96 / 0.95 / 0.95 / 0.95 / 0.93, F1 0.70 / 0.67 / 0.66 / 0.67 / 0.68 / 0.68 - "
             "weitgehend unabhängig von p, weil γ₀ = 1 / p mitwächst.",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. Bei γ₀ sinkt die AUC der One-Class SVM bei 0 / 5 / 10 / 20 / 30 / 40 Rauschmerkmalen von 0.95 auf 0.91 und der Recall an der Schwelle von 0.52 auf 0.25 (F1 0.67 → 0.39; LOF-Recall 1.00 / 0.93 / 0.71 / 0.18 / 0.03 / 0.00). "
             "Mit γ = 0.1 · γ₀ bleibt sie bei AUC 0.99-1.00 und F1 0.92-0.95.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. AUC der One-Class SVM bei 1 / 2 / 3 Betriebsarten 0.95 / 0.95 / 0.91 (F1 0.67 / 0.76 / 0.77), robust 1.00 / 0.95 / 0.85 - die gelernte Grenze verkraftet mehrere Gruppen besser als die eine Ellipse.",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 steigt der F1 der One-Class SVM von 0.67 auf 0.78 / 0.84 / 0.85 / 0.85 (AUC 0.95 / 0.99 / 0.99 / 0.98 / 0.98), "
             "während der LOF von 0.94 auf 0.74 und die robuste Ellipse von 0.84 auf 0.38 fallen: die Krümmung ist die Stärke der gelernten Grenze.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen. AUC der One-Class SVM bei 0 / 0.25 / 0.5 / 0.75 / 1.0: 0.92 / 0.95 / 0.97 / 0.97 / 0.96, der F1 an der Schwelle sinkt dabei von 0.70 auf 0.53 (der Recall von 0.57 auf 0.37).",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind. Bei festem ν = 0.1 ist die AUC der One-Class SVM bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % 1.00 / 1.00 / 0.95 / 0.77 / 0.56 / 0.37 / 0.32 - ab einem Anteil über ν fällt sie, ab 40 % **unter Raten** "
             "(LOF 1.00 / 1.00 / 1.00 / 0.99 / 0.90 / 0.72 / 0.64, Isolation Forest und ECOD überall etwa 1.00, robust 1.00 bis 0.87). Die Grenze wird auf den Trainingsdaten gelernt; ν muss größer als der Anteil sein.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe (bei ν = 0.1): bei 10 % / 30 % AUC der One-Class SVM 0.71 / 0.74 (LOF 0.35 / 0.47, Isolation Forest 0.95 / 0.70, robust 1.00 / 0.49, ECOD 0.98 / 0.84). "
             "In der Lücke: 2 Betriebsarten 0.64, 3 Betriebsarten 0.82 - als einziger Detektor deutlich über Raten (LOF 0.53 / 0.36, Isolation Forest 0.54 / 0.27, robust 0.40 / 0.38, ECOD 0.04 / 0.01). "
             "Korrelationsbruch: Merkmale der Anomalien je Spalte aus den Normalen neu gezogen - One-Class SVM 0.88, LOF 0.99, robust 1.00, Isolation Forest 0.80, ECOD 0.56.",
    )
    if kind not in ("gap", "decorrelated"):
        seed_widget("strength_slider")
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind.",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))

    st.markdown("**One-Class SVM**")
    nu = st.slider(
        "ν (Anteil außerhalb der Grenze)", *bounds("nu_slider"), key="nu_slider", step=0.01, format="%.2f",
        help="Obere Schranke des Anteils markierter Trainingstouren und untere Schranke des Anteils der Stützvektoren. Bei 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 (Standardfall, γ₀): AUC 0.89 / 0.89 / 0.90 / 0.95 / 1.00 / 1.00 / 1.00, "
             "Recall an der Schwelle 0.00 / 0.00 / 0.01 / 0.52 / 1.00 / 1.00 / 1.00, Fehlalarmrate 0 % / 0 % / 0 % / 0.2 % / 9 % / 21 % / 44 %, F1 0.00 / 0.00 / 0.03 / 0.67 / 0.71 / 0.51 / 0.34. "
             "Erst ein ν deutlich über dem Anteil (10 %) macht die Rangfolge perfekt - aber die Schwelle markiert dann etwa ν der Touren.",
    )
    gamma = st.select_slider(
        "γ (Vielfaches von γ₀ = 1 / p)", options=list(C.GAMMA_FACTORS), key="gamma_slider", format_func=lambda g: f"{g:g} · γ₀",
        help="Breite der Kuppeln: klein = breite Kuppeln (fast eine glatte Wanne), groß = schmale (jede Tour ihre eigene). Bei 0.02 / 0.05 / 0.1 / 0.25 / 0.5 / 1 / 2 / 4 / 8 / 16 / 32 · γ₀ (Standardfall, ν = 0.1): "
             "F1 0.95 / 0.96 / 0.95 / 0.90 / 0.81 / 0.67 / 0.44 / 0.03 / 0.00 / 0.00 / 0.00, AUC 1.00 / 1.00 / 1.00 / 0.96 / 0.94 / 0.95 / 0.95 / 0.90 / 0.83 / 0.68 / 0.60. "
             "Die Voreinstellung γ₀ liegt hier nicht im guten Bereich; für gekrümmten Normalbereich, Lücke und Korrelationsbruch ist ein größeres γ besser (Experiment γ × ν).",
    )
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: One-Class SVM bei f < 0, LOF über 1.5, Isolation Forest über Score 0.5, robust über dem χ²-Quantil. Erwarteter Anteil: bei allen vieren werden die größten Werte markiert - "
             "dann entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        seed_widget("quantile_slider")
        quantile = st.slider(
            "Schwelle: χ²-Quantil (robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei der robusten Schätzung als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90.",
        )
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
    else:
        seed_widget("share_slider")
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle vier Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % bei der One-Class SVM (Standardfall, γ₀) "
                 "0.33 / 0.64 / 0.67 / 0.63 / 0.38, bei LOF und Isolation Forest 0.33 / 0.67 / 0.97 / 0.67 / 0.40, bei robust 0.33 / 0.67 / 0.90 / 0.66 / 0.40.",
        )
        st.session_state["_share_kept"] = share
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "nu_slider": float(nu), "gamma_slider": float(gamma), "threshold_kind_select": threshold_kind,
    "quantile_slider": float(quantile), "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(nu=float(round(nu, 2)), gamma_factor=float(gamma), threshold_kind=threshold_kind, quantile=float(quantile), share=int(share))
with st.spinner("Löse das duale Problem..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
ss, ls, ifs, rs = (a.scores[d] for d in ("ocsvm", "lof", "iforest", "robust"))
n_anom = int(ds.anomaly.sum())
p_total = a.p_total
fit = a.fit
values = a.values["ocsvm"]
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- One-Class SVM in Aktion -----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 One-Class SVM in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Kernel", 3: "3 · Grenze", 4: "4 · Wert", 5: "5 · Ergebnis"}
if "ocsvm_step" not in st.session_state or st.session_state.get("ocsvm_step_owner") != data_key:
    st.session_state["ocsvm_step"] = 1
    st.session_state["ocsvm_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ocsvm_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
normal_idx = np.flatnonzero(~ds.anomaly)
anom_idx = np.flatnonzero(ds.anomaly)
i_anom = int(anom_idx[np.argsort(values[anom_idx])[len(anom_idx) // 2]])                          # eine typische Sonderfahrt: die mit dem mittleren Wert
i_norm = int(normal_idx[np.argmin(np.abs(P[normal_idx]).sum(axis=1))])                             # eine normale Tour nahe der Mitte
# Merkmalspaar für die Darstellung: beim Korrelationsbruch das am stärksten korrelierte Paar der Normalen, sonst die ersten zwei
if ds.kind == "decorrelated" and ds.p >= 2:
    corr = np.abs(np.corrcoef(ds.X[~ds.anomaly][:, : ds.p].T))
    np.fill_diagonal(corr, 0.0)
    pair_i, pair_j = (int(v) for v in np.unravel_index(np.argmax(corr), corr.shape))
else:
    pair_i, pair_j = 0, min(1, ds.p - 1)
names_all = ds.names
Z = fit.X                                                                                                  # die standardisierten Kennzahlen, auf denen die Grenze gelernt wurde
g0 = svm.gamma_scale(Z)
sq = (Z ** 2).sum(axis=1)
d_mat = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0))
np.fill_diagonal(d_mat, np.inf)
nn_dist = float(np.median(d_mat.min(axis=1)))
np.fill_diagonal(d_mat, 0.0)
med_dist = float(np.median(d_mat[np.triu_indices(len(Z), 1)]))
K_row = fit.kernel[i_norm]
K_row_a = fit.kernel[i_anom]


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {names_all[pair_i]} gegen {names_all[pair_j]}** (Einheiten wie gemessen)" + (" - das am stärksten korrelierte Paar der Normalen" if ds.kind == "decorrelated" else ""))
            c2.plotly_chart(build_features(ds.X, ds.anomaly, names_all, pair_i, pair_j), width="stretch", key="step_features")
        elif current_step == 2:
            c1, c2 = st.columns(2)
            c1.markdown("**Der Kernel: Ähnlichkeit über den Abstand** (gewähltes γ dick; gepunktet die Stufen des Reglers)")
            factors = [(f, f * g0, abs(f - settings.gamma_factor) < 1e-12) for f in (0.1, 0.5, 1.0, 4.0, 16.0)]
            if not any(cur for _, _, cur in factors):
                factors.append((settings.gamma_factor, settings.gamma_factor * g0, True))
            c1.plotly_chart(build_kernel(Z, fit.gamma, factors, [(nn_dist, "nächster Nachbar"), (med_dist, "zufällige Tour")]), width="stretch", key="step_kernel")
            c2.markdown("**Ähnlichkeit einer normalen Tour zu allen anderen** (Kernel-Wert, Raute = gewählte Tour)")
            c2.plotly_chart(build_similarity(P, ds.anomaly, K_row, i_norm), width="stretch", key="step_similarity")
        elif current_step == 3:
            xs, ys, F, S, sv = _maps(data_params, settings, pair_i, pair_j)
            st.markdown(f"**Die gelernte Grenze in der Ebene {names_all[pair_i]} gegen {names_all[pair_j]}** (die One-Class SVM wird dafür auf nur diese zwei Merkmale neu trainiert; schwarze Linie: f = 0, umkreist: Stützvektoren)")
            st.plotly_chart(build_boundary(xs, ys, F, S, ds.X[:, [pair_i, pair_j]], ds.anomaly, sv, [names_all[pair_i], names_all[pair_j]]), width="stretch", key="step_boundary")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            c1.markdown("**One-Class-SVM-Wert −f je Tour** (Schwelle gestrichelt)")
            c1.plotly_chart(build_score_hist(values, ds.anomaly, ss["threshold"]), width="stretch", key="step_score_hist")
            c2.markdown("**One-Class SVM gegen die Wurzel**: Rang jeder Tour bei beiden")
            c2.plotly_chart(build_rank_compare(values, a.values["robust"], ds.anomaly), width="stretch", key="step_rank_compare")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("ocsvm", "lof", "iforest", "robust")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {p_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Die Ebene ist die der zwei größten Streuungsrichtungen der robusten Schätzung in standardisierten Kennzahlen; die One-Class SVM rechnet mit allen {p_total} standardisierten Kennzahlen auf einmal."
               + (" Beim Korrelationsbruch haben die Anomalien in jedem Merkmal normale Werte - nur die Kombination stimmt nicht." if ds.kind == "decorrelated" else ""))
elif step == 2:
    st.caption(f"γ = {settings.gamma_factor:g} · γ₀ = {fit.gamma:.4f} (γ₀ = 1 / p = {g0:.4f}). Der Abstand zum nächsten Nachbarn beträgt typischerweise (Median) {nn_dist:.2f}, der zu einer zufälligen Tour {med_dist:.2f} (standardisierte Einheiten): dort ist die Ähnlichkeit "
               f"{np.exp(-fit.gamma * nn_dist ** 2):.2f} bzw. {np.exp(-fit.gamma * med_dist ** 2):.2f}. Die gewählte normale Tour hat zu den anderen Touren im Mittel die Ähnlichkeit {np.delete(K_row, i_norm).mean():.2f}, die Sonderfahrt {np.delete(K_row_a, i_anom).mean():.2f}: "
               "die Methode beruht darauf, dass Anomalien zu wenigen anderen Touren ähnlich sind.")
elif step == 3:
    st.caption(f"Mit allen {p_total} Kennzahlen: {fit.n_support} Stützvektoren ({fit.n_support / ds.n:.0%} der Touren; ν = {settings.nu:.2f} verlangt mindestens {settings.nu:.0%}), davon {fit.n_bound} an der oberen Schranke (α = 1, außerhalb oder auf der Grenze); "
               f"markiert werden {ss['n_flagged']} Touren ({ss['n_flagged'] / ds.n:.0%}; ν erlaubt höchstens {settings.nu:.0%}). ρ = {fit.rho:.3f}, der Löser brauchte {fit.iterations} Schritte" + ("." if fit.converged else " und ist **nicht** konvergiert (Iterationslimit)."))
elif step == 4:
    st.caption(f"Mittlerer Wert −f der normalen Touren {values[~ds.anomaly].mean():.3f}, der Sonderfahrten {values[ds.anomaly].mean():.3f}; die Schwelle ist f = 0 und markiert {ss['n_flagged']} von {ds.n} Touren. "
               "Rechts: liegen die Punkte nahe der Diagonalen, urteilen One-Class SVM und Wurzel gleich; Punkte weit von der Diagonalen sind Touren, bei denen die gelernte Grenze und die Ellipse verschiedener Meinung sind.")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Detektoren gefunden haben – One-Class SVM gegen LOF gegen Isolation Forest gegen Wurzel")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: One-Class SVM bei f < 0, LOF über 1.5, Isolation Forest über Score 0.5, robust über dem χ²-Quantil). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (One-Class SVM)", f"{ss['auc']:.2f}", help="Rangfolge der Anomalie-Werte; darunter die anderen Detektoren.")
m1.caption(f"LOF {ls['auc']:.2f} · Isolation Forest {ifs['auc']:.2f} · robust {rs['auc']:.2f}")
m2.metric("Recall (One-Class SVM)", _pct(ss["recall"]), help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m2.caption(f"LOF {_pct(ls['recall'])} · Isolation Forest {_pct(ifs['recall'])} · robust {_pct(rs['recall'])}")
m3.metric("Fehlalarmrate (One-Class SVM)", f"{ss['false_alarm']:.1%}", help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m3.caption(f"LOF {ls['false_alarm']:.1%} · Isolation Forest {ifs['false_alarm']:.1%} · robust {rs['false_alarm']:.1%}")
m4.metric("F1 (One-Class SVM)", f"{ss['f1']:.2f}", help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter die F1 der anderen Detektoren und die der One-Class SVM, wenn der wahre Anteil bekannt wäre.")
m4.caption(f"LOF {ls['f1']:.2f} · Isolation Forest {ifs['f1']:.2f} · robust {rs['f1']:.2f} · One-Class SVM mit bekanntem Anteil {a.oracle_f1['ocsvm']:.2f}")

_t = vd
if code == "ocsvm_wins":
    if _t["ocsvm_auc"] - _t["best_other_auc"] >= 0.05:
        st.success(f"✅ Die One-Class SVM ist besser: AUC {_t['ocsvm_auc']:.2f} gegen {_t['best_other_auc']:.2f} beim besten anderen Detektor (LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}). "
                   f"An der Schwelle: Recall {_pct(_t['ocsvm_recall'])}, F1 {_t['ocsvm_f1']:.2f}, Fehlalarmrate {_t['ocsvm_false_alarm']:.0%}.")
    else:
        st.success(f"✅ An der Schwelle ist die One-Class SVM besser: F1 {_t['ocsvm_f1']:.2f} (Recall {_pct(_t['ocsvm_recall'])}, Fehlalarmrate {_t['ocsvm_false_alarm']:.0%}) gegen {_t['best_other_f1']:.2f} beim besten anderen Detektor "
                   f"(LOF {_t['lof_f1']:.2f}, Isolation Forest {_t['iforest_f1']:.2f}, robust {_t['robust_f1']:.2f}); die Rangfolge ist gleichauf (AUC {_t['ocsvm_auc']:.2f} gegen {_t['best_other_auc']:.2f}). "
                   f"Voraussetzung: ν = {_t['nu']:.2f} und γ = {_t['gamma_factor']:g} · γ₀ passen zu den Daten.")
elif code == "comparable":
    st.success(f"✅ Die One-Class SVM ist ebenbürtig: AUC {_t['ocsvm_auc']:.2f} (LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}); bei der Schwelle F1 {_t['ocsvm_f1']:.2f} "
               f"(LOF {_t['lof_f1']:.2f}, Isolation Forest {_t['iforest_f1']:.2f}, robust {_t['robust_f1']:.2f}), mit bekanntem Anteil {_t['oracle_ocsvm']:.2f}. Dafür mussten ν = {_t['nu']:.2f} und γ = {_t['gamma_factor']:g} · γ₀ stimmen.")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist gut (AUC {_t['ocsvm_auc']:.2f}), aber f < 0 markiert nur {_t['ocsvm_n_flagged']} Touren (Recall {_pct(_t['ocsvm_recall'])}, Fehlalarmrate {_t['ocsvm_false_alarm']:.0%}): "
               f"F1 {_t['ocsvm_f1']:.2f} gegen {_t['oracle_ocsvm']:.2f} mit bekanntem Anteil. Der Anteil, den die Grenze außen lässt, hängt an ν ({_t['nu']:.2f}) und γ ({_t['gamma_factor']:g} · γ₀), nicht an den Daten. "
               f"LOF {_t['lof_f1']:.2f}, Isolation Forest {_t['iforest_f1']:.2f}.")
elif code == "nu_low":
    st.warning(f"⚠️ ν ist kleiner als der Anteil der Anomalien: ν = {_t['nu']:.2f}, die Daten enthalten {_t['contamination']:.0f} % Sonderfahrten. Die Grenze wird auf den Trainingsdaten gelernt und umschließt die Anomalien mit: "
               f"AUC {_t['ocsvm_auc']:.2f} (LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}). Ein ν über dem Anteil stellt die Rangfolge wieder her - wenn man den Anteil kennt.")
elif code == "gap_rank":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: nur die One-Class SVM hat hier eine Rangfolge (AUC {_t['ocsvm_auc']:.2f}; LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}, ECOD {_t['ecod_auc']:.2f}), "
               f"aber die Schwelle f < 0 findet sie nicht (Recall {_pct(_t['ocsvm_recall'])}, F1 {_t['ocsvm_f1']:.2f}). Mit breiteren Kuppeln (kleinerem γ) fällt auch die Rangfolge unter Raten.")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['ocsvm_auc']:.2f} bei der One-Class SVM, {_t['lof_auc']:.2f} beim LOF, {_t['iforest_auc']:.2f} beim Isolation Forest, {_t['robust_auc']:.2f} robust: keiner findet sie zuverlässig. "
               "Die One-Class SVM sieht die Lücke nur mit passendem γ (etwa 1 · γ₀); mit breiteren Kuppeln liegt die Lücke im Innern der gelernten Grenze und wirkt normaler als die Normalen.")
elif code == "others_win":
    st.warning(f"⚠️ Ein anderer Detektor ist besser: AUC One-Class SVM {_t['ocsvm_auc']:.2f}, LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}; an der Schwelle F1 {_t['ocsvm_f1']:.2f} gegen {_t['best_other_f1']:.2f}. "
               f"Bei ν = {_t['nu']:.2f} und γ = {_t['gamma_factor']:g} · γ₀ ist die gelernte Grenze für diese Daten ungünstig gezogen; ein anderes γ ändert das Bild stark (Experiment γ × ν).")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}"), ("Schwelle", "threshold", "{:.2f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "One-Class SVM": [r[2].format(ss[r[1]]) for r in rows], "LOF": [r[2].format(ls[r[1]]) for r in rows], "Isolation Forest": [r[2].format(ifs[r[1]]) for r in rows],
              "robust (MCD)": [r[2].format(rs[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Parameter", "Bewertung", "mittlerer Wert (normal / Anomalie)"],
              "One-Class SVM": [f"{a.seconds['ocsvm'] * 1000:.1f} ms", f"ν = {settings.nu:.2f}, γ = {settings.gamma_factor:g} · γ₀", f"Entscheidungsfunktion f ({fit.n_support} Stützvektoren)", f"{values[~ds.anomaly].mean():.3f} / {values[ds.anomaly].mean():.3f}"],
              "LOF": [f"{a.seconds['lof'] * 1000:.1f} ms", "k = 20 (höchstens n / 2)", "Dichte gegenüber den Nachbarn", f"{a.values['lof'][~ds.anomaly].mean():.2f} / {a.values['lof'][ds.anomaly].mean():.2f}"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest_if.trees)} Bäume × ψ = {a.forest_if.psi}", "Pfadlänge in zufälligen Bäumen", f"{a.values['iforest'][~ds.anomaly].mean():.2f} / {a.values['iforest'][ds.anomaly].mean():.2f}"]})
    st.caption(f"Die One-Class SVM ist deterministisch, rechnet mit standardisierten Kennzahlen und speichert eine Kernmatrix aus {ds.n} × {ds.n} = {ds.n ** 2:,} Zahlen ({ds.n ** 2 * 8 / 1e6:.1f} MB). Die klassische Schätzung der Wurzel entfällt in dieser Demo.".replace(",", "."))

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Daten und Reglern ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind in ("gap", "decorrelated") and k in ("strength",)) or (kind == "gap" and k == "n_modes") or (threshold_kind == "share" and k == "quantile") or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise), "contamination": int(contamination), "strength": float(strength),
           "nu": float(settings.nu), "gamma_factor": float(settings.gamma_factor), "quantile": float(quantile), "share": int(share)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings, SWEEP_VALUES[sweep_param])
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle. "
               "Bei ν und γ ändern sich nur die Linien der One-Class SVM.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wo die gelernte Grenze trifft und wo nicht: acht Szenarien")
if st.button("Die Detektoren in acht Szenarien vergleichen (dauert etwa 30 Sekunden)", key="scenarios_start"):
    st.session_state["scenarios_on"] = True
if st.session_state.get("scenarios_on"):
    with st.spinner("Rechne 8 Szenarien × 5 Aufnahmen..."):
        sc_rows = _scenarios(settings)
    st.plotly_chart(build_scenarios(sc_rows), width="stretch", key="scenarios_chart")
    st.caption("Mittel über 5 feste Datensätze, One-Class SVM mit den Reglern der Seitenleiste (Standard ν = 0.1, γ₀), ECOD in der zweiseitigen Variante als Kontrast. Im **Standardfall** ist die AUC 0.95 gegen 1.00 bei LOF, Isolation Forest und robust; "
               "beim **gekrümmten Normalbereich** 0.98 (LOF 1.00, Isolation Forest 1.00, robust 1.00 - die Rangfolge der anderen ist auch hier perfekt, aber ihre Schwelle nicht: F1 LOF 0.74, robust 0.38 gegen 0.85). "
               "**Dichte Gruppe (30 %)**: 0.74 (LOF 0.47, Isolation Forest 0.70, robust 0.49, ECOD 0.84). **Lücke**: 0.64 (2 Betriebsarten) und 0.82 (3) - die anderen 0.27-0.54, ECOD 0.04 und 0.01. **Korrelationsbruch**: 0.88 (LOF 0.99, Isolation Forest 0.80, robust 1.00, ECOD 0.56). "
               "**45 % verstreut**: 0.32 - unter Raten, weil ν = 0.1 kleiner als der Anteil ist (Isolation Forest 1.00, ECOD 1.00, robust 0.87, LOF 0.64). **40 Rauschmerkmale**: 0.91 (LOF 0.99, Isolation Forest 0.99, robust 0.88, ECOD 0.97). "
               "Rechts der F1 mit bekanntem Anteil: im Standardfall 0.67 gegen 0.97 (LOF und Isolation Forest).")

st.markdown("---")

st.subheader("🔬 Das Raster aus γ und ν")
if st.button("γ × ν-Raster im Standardfall berechnen (dauert etwa 20 Sekunden)", key="grid_start"):
    st.session_state["grid_on"] = True
if st.session_state.get("grid_on"):
    with st.spinner("Rechne 5 ν × 11 γ × 5 Aufnahmen..."):
        gr = _grid(base_data)
    st.plotly_chart(build_grid(gr), width="stretch", key="grid_chart")
    st.caption("Mittel über 5 feste Datensätze, Daten wie in der Seitenleiste; Zeilen ν, Spalten γ in Vielfachen von γ₀. Im Standardfall: bei ν ≥ 0.2 ist die AUC von γ = 0.02 · γ₀ bis γ = 4 · γ₀ gleich 1.00 (ν = 0.3: bis 8 · γ₀), "
               "aber der F1 an der Schwelle f < 0 bleibt bei ν = 0.2 und 0.3 bis γ = 1 · γ₀ nur 0.67-0.71 bzw. 0.50-0.51 - die Schwelle markiert etwa ν der Touren. Die einzigen Zellen mit F1 ≥ 0.85 liegen bei ν = 0.1 mit γ ≤ 0.25 · γ₀ (0.90-0.96) "
               "und auf einer Diagonalen aus größerem ν und größerem γ (ν = 0.2 mit γ = 8 · γ₀: 0.87; ν = 0.3 mit γ = 16 · γ₀: 0.87): **ν und γ wirken zusammen**, das gute F1 liegt auf einem schmalen Band. Bei ν = 0.02 ist die AUC unruhig (1.00 / 1.00 / 0.98 / 0.40 / 0.60 / 0.89 bei 0.02 bis 1 · γ₀).")

st.markdown("---")

st.subheader("🔬 ν als Anteil: die Schranken der Theorie")
if st.button("ν-Kalibrierung berechnen (dauert etwa 10 Sekunden)", key="nu_start"):
    st.session_state["nu_on"] = True
if st.session_state.get("nu_on"):
    with st.spinner("Rechne 6 ν-Werte × 2 Anteile × 5 Aufnahmen..."):
        nt = _nu_table(tuple(kv for kv in base_data if kv[0] not in ("contamination",)))
    st.plotly_chart(build_nu(nt), width="stretch", key="nu_chart")
    st.caption("Anteil der markierten Trainingstouren (durchgezogen) und der Stützvektoren (gestrichelt) über ν, bei fast reinen Normalen (1 % Anomalien) und im Standardfall (10 %), γ₀; Mittel über 5 feste Datensätze. Die Schranken der Theorie halten in jeder Zelle: "
               "markiert ≤ ν ≤ Stützvektoren. Bei ν = 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 werden bei fast reinen Normalen 0.0 % / 1.9 % / 6.3 % / 16.9 % / 27.9 % / 48.7 % markiert (Stützvektoren 12.3 % / 13.0 % / 15.7 % / 23.3 % / 32.6 % / 51.3 %), "
               "im Standardfall 0.0 % / 0.1 % / 5.4 % / 18.3 % / 28.9 % / 49.5 % (Stützvektoren 15.5 % / 15.5 % / 16.7 % / 21.3 % / 30.8 % / 50.6 %). "
               "Bei größerem ν markiert die Schwelle also **so viele Touren, wie ν erlaubt** - unabhängig davon, ob es Anomalien gibt: bei fast reinen Normalen ist die Fehlalarmrate bei ν = 0.2 gleich 16 %, im Standardfall 9 %. "
               "ν ist ein Anteil, keine Fehlalarmrate: die Schranke ist bei kleinem ν (0.02 und 0.05) weit unterschritten, weil die Stützvektoren auf der Grenze f = 0 liegen und dort nicht markiert werden.")

st.markdown("---")

st.subheader("🔬 Die Lücke zwischen den Betriebsarten")
if st.button("Lücke über γ berechnen (dauert etwa 20 Sekunden)", key="gap_start"):
    st.session_state["gap_on"] = True
if st.session_state.get("gap_on"):
    with st.spinner("Rechne 2 × 2 × 11 Einstellungen × 5 Aufnahmen..."):
        gp = _gap(tuple(kv for kv in base_data if kv[0] not in ("kind", "n_modes", "contamination", "strength")))
    st.plotly_chart(build_gap(gp), width="stretch", key="gap_chart")
    st.caption("AUC der One-Class SVM in der Lücke (10 % Anomalien; Mittel über 5 feste Datensätze) über γ, für 2 und 3 Betriebsarten und ν = 0.1 bzw. 0.3. Bei breiten Kuppeln (γ ≤ 0.25 · γ₀) ist die AUC **unter Raten**: 0.02 (2 Betriebsarten) und 0.00 (3) - "
               "die gelernte Grenze umschließt die Lücke wie die Ellipse und ECOD, sie liegt im Innern. Erst ab γ = 1 · γ₀ trennt die Grenze die Betriebsarten: AUC 0.64 (2) und 0.82 (3) bei ν = 0.1, 0.46 und 0.57 bei ν = 0.3; danach fällt sie zu 0.5 hin. "
               "Selbst dort markiert die Schwelle die Lücke kaum (Recall bei ν = 0.1 unter 4 %).")

st.markdown("---")

st.subheader("🔬 Rauschmerkmale und Standardisierung")
if st.button("Rauschmerkmale 0-40 für zwei γ und die Standardisierung berechnen (dauert etwa 30 Sekunden)", key="noise_start"):
    st.session_state["noise_on"] = True
if st.session_state.get("noise_on"):
    with st.spinner("Rechne 6 Werte × 2 γ × 5 Aufnahmen und die Standardisierung..."):
        nb = tuple(kv for kv in base_data if kv[0] != "n_noise")
        n_wide = _sweep("n_noise", nb, Settings(nu=settings.nu, gamma_factor=0.1, threshold_kind=threshold_kind, quantile=float(quantile), share=int(share)), SWEEP_VALUES["n_noise"])
        n_std = _sweep("n_noise", nb, Settings(nu=settings.nu, gamma_factor=1.0, threshold_kind=threshold_kind, quantile=float(quantile), share=int(share)), SWEEP_VALUES["n_noise"])
        sd = _standardise(base_data)
    c1, c2 = st.columns(2)
    c1.markdown("**γ = γ₀ (Voreinstellung)**")
    c1.plotly_chart(build_sweep(n_std, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_gamma1_chart")
    c2.markdown("**γ = 0.1 · γ₀ (breiter Kernel)**")
    c2.plotly_chart(build_sweep(n_wide, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_gamma01_chart")
    st.caption("Mittel über 5 feste Datensätze, ν wie in der Seitenleiste. Bei γ₀ sinkt die AUC der One-Class SVM von 0.95 auf 0.91 und der F1 an der Schwelle von 0.67 auf 0.39 (Recall 0.52 → 0.25); bei γ = 0.1 · γ₀ bleibt die AUC bei 0.99-1.00 und der F1 bei 0.92-0.95 - "
               "bei 40 Rauschmerkmalen ist das der beste Detektor an der Schwelle (LOF 0.00, Isolation Forest 0.56, robust 0.35): der breite Kernel mittelt unabhängiges Rauschen weg. "
               "**Standardisierung** (Standardfall): ohne sie sinkt die AUC der One-Class SVM von 0.95 auf 0.79 (F1 0.67 → 0.68), die des LOF von 1.00 auf 0.87 (F1 0.94 → 0.68) - auf Rohdaten dominiert das Merkmal mit der größten Einheit.")
    st.plotly_chart(build_standardise(sd), width="stretch", key="standardise_chart")

st.markdown("---")

st.subheader("🔬 Die Schwelle")
if st.button("ν über die Schwelle vergleichen (dauert etwa 20 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 7 ν-Werte × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**One-Class SVM: F1, Recall, Fehlalarmrate und AUC je ν (Schwelle f < 0)**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.caption("Links (Standardfall, γ₀): F1 bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 gleich 0.00 / 0.00 / 0.03 / 0.67 / 0.71 / 0.51 / 0.34, Recall 0.00 / 0.00 / 0.01 / 0.52 / 1.00 / 1.00 / 1.00 - die Schwelle f < 0 liefert nur in einem schmalen Bereich um ν = 0.1 bis 0.2 brauchbare Werte. "
               "Rechts: ein falsch angenommener Anteil bringt alle vier Detektoren auf fast denselben F1 (One-Class SVM 0.64 / 0.67 / 0.63, LOF und Isolation Forest 0.67 / 0.97 / 0.67), denn dann entscheidet nur die Rangfolge; "
               "die One-Class SVM verliert im Standardfall schon beim wahren Anteil gegen LOF und Isolation Forest, weil ihre Rangfolge schlechter ist (AUC 0.95).")

st.markdown("---")

st.subheader("🔬 Kosten")
if st.button("Rechenzeit berechnen (dauert etwa 20 Sekunden)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Messe die Rechenzeit..."):
        ct = _costs(tuple(kv for kv in base_data if kv[0] not in ("n", "n_noise")), settings)
    st.plotly_chart(build_costs(ct["times"]), width="stretch", key="cost_chart")
    st.caption("Mittel über 3 feste Datensätze (Zeiten rechnerabhängig, nur die Größenordnungen zählen). Die One-Class SVM braucht im gemessenen Bereich (bis 600 Touren) höchstens einige zehn ms (bei 600 Touren etwa 13-22 ms in mehreren Läufen) - so viel wie der LOF (bei 600 Touren etwa 15-20 ms) und weit weniger als der Isolation Forest (etwa 70-620 ms). "
               "Die Kernmatrix wächst quadratisch: 0.08 / 0.72 / 2.9 MB bei 100 / 300 / 600 Touren (8 · n² Byte); bei 10 000 Touren wären es 800 MB - gerechnet, nicht gemessen. Der Löser braucht etwa n Schritte (109 / 330 / 589 bei 100 / 300 / 600 Touren, 12 Merkmale).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **ν ist größer als der Anteil der Anomalien** | Sonst wird die Grenze um die Anomalien mitgezogen: bei ν = 0.1 und 45 % Anomalien AUC **0.32** (unter Raten), ab 30 % schon 0.56; mit ν = 0.5 wieder 0.99. Isolation Forest und ECOD haben hier 1.00 - ohne Anteil. Selbst bei ν = Anteil (10 %) ist die AUC nur 0.95, erst ν = 0.2 macht sie perfekt. | ECOD, Isolation Forest |
| **γ passt zur Struktur der Daten** | Ein Regler entscheidet über das Ergebnis: im Standardfall F1 0.95 bei γ = 0.1 · γ₀ und 0.67 bei der Voreinstellung γ₀, bei γ = 8 · γ₀ markiert die Schwelle **keine einzige** Tour (F1 0.00, 43 % Stützvektoren). Für Lücke (AUC 0.82 bei γ₀, 0.00 bei γ ≤ 0.25 · γ₀), Krümmung und Korrelationsbruch liegt das beste γ woanders. | Deep SVDD (gelernte Abbildung), LOF, Isolation Forest |
| **Die Schwelle f < 0 ist ein Anteil** | Sie markiert höchstens ν der Touren, unabhängig von den Daten: bei fast reinen Normalen mit ν = 0.2 markiert sie 16.9 % der Touren (Fehlalarmrate 16 %). Bei 20-50 Touren und ν = 0.1 markiert sie **nichts** (F1 0.00). | ECOD (Fisher-Quantil), LOF (1.5) |
| **Standardisierte Merkmale** | Ohne Standardisierung sinkt die AUC von 0.95 auf 0.79 (LOF von 1.00 auf 0.87). | - |
| **Viele irrelevante Merkmale** | Mit der Voreinstellung γ₀ sinkt der Recall an der Schwelle bei 40 Rauschmerkmalen von 0.52 auf 0.25 (F1 0.39); erst ein breiter Kernel (0.1 · γ₀) stellt F1 0.92 wieder her. | Feature Bagging, breiter Kernel |
| **Die Grenze folgt der Struktur, nicht der Abhängigkeit** | Der Korrelationsbruch wird mit γ₀ nur bedingt erkannt (AUC 0.88, F1 0.30; LOF 0.99, robust 1.00); mit ν = 0.3 und γ = 2 · γ₀ steigt die AUC auf 0.98, aber der F1 bleibt bei 0.57, weil die Schwelle ν der Touren markiert. | LOF, Wurzel |
| **Die Lücke** | Nur bei passendem γ über Raten (AUC 0.64 bzw. 0.82 mit 2 bzw. 3 Betriebsarten bei γ₀), und die Schwelle findet sie kaum (Recall 3 % bzw. 1 %). | (keiner) |
| **Platz für die Kernmatrix** | 8 · n² Byte: 2.9 MB bei 600 Touren, 800 MB bei 10 000 (gerechnet). Im gemessenen Bereich ist die Rechenzeit unkritisch (höchstens einige zehn ms). | Isolation Forest, ECOD |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie: die Wurzel Elliptic Envelope, LOF, Feature Bagging, Isolation Forest, Extended IF, ECOD, Deep SVDD und der Autoencoder (alle gebaut). "
    "Keiner ist überlegen: die One-Class SVM lernt eine Grenze, die der Krümmung folgt und bei passendem γ Rauschmerkmale wegmittelt - zum Preis von zwei Reglern, von denen der eine (ν) den Anteil der Anomalien kennen muss und der andere (γ) je nach Daten verschieden liegt."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Primales Problem.** Mit der Merkmalsabbildung $\phi$ des RBF-Kernels $K(x, z) = e^{-\gamma\lVert x - z\rVert^2}$ sucht die One-Class SVM die Hyperebene $(w, \rho)$ mit größtem Abstand zum Ursprung im Merkmalsraum:
$\min_{w, \xi, \rho}\ \tfrac12\lVert w\rVert^2 + \tfrac{1}{\nu n}\sum_i \xi_i - \rho$ unter $\langle w, \phi(x_i)\rangle \ge \rho - \xi_i$, $\xi_i \ge 0$.

**Duales Problem** (Skalierung von libsvm): $\min_\alpha\ \tfrac12\alpha^\top K\alpha$ unter $0 \le \alpha_i \le 1$ und $\sum_i \alpha_i = \nu n$. Entscheidungsfunktion $f(x) = \sum_i \alpha_i K(x_i, x) - \rho$; als Anomalie gilt $f(x) < 0$.
Freie Stützvektoren ($0 < \alpha_i < 1$) liegen genau auf der Grenze, $f(x_i) = 0$; Stützvektoren mit $\alpha_i = 1$ liegen außerhalb ($f \le 0$); alle anderen Touren innerhalb ($f > 0$).

**Schranken für ν.** Es gilt (Schölkopf u. a.): $\dfrac{\#\{i : f(x_i) < 0\}}{n} \le \nu \le \dfrac{\#\{i : \alpha_i > 0\}}{n}$. Die Demo prüft das an jeder Aufnahme und über alle Sweeps.

**SMO.** In jedem Schritt wird ein Paar $(i, j)$ mit $\alpha_i < 1$, $\alpha_j > 0$ gewählt (zweite Ordnung nach Fan, Chen und Lin), $\alpha_i$ um $d$ erhöht und $\alpha_j$ um $d$ gesenkt, $d = (G_j - G_i)/(K_{ii} + K_{jj} - 2K_{ij})$ (auf die Schranken beschnitten), mit dem Gradienten $G = K\alpha$; Abbruch, wenn $\max_{j \in \text{low}} G_j - \min_{i \in \text{up}} G_i < 10^{-6}$.

**Voreinstellung von γ.** $\gamma_0 = 1 / (p \cdot \mathrm{Var}(X))$; bei standardisierten Daten $1/p$. Die Regler der Demo sind Vielfache von $\gamma_0$.

**Grenzen.** (1) ν muss größer als der Anteil der Anomalien sein. (2) γ passt nur zu einer Struktur. (3) Die Schwelle markiert höchstens den Anteil ν. (4) Speicher $8n^2$ Byte.

Implementiert in `ocsvm_algorithm.py` (Kernel, SMO, Entscheidungsfunktion), `ocsvm_lof.py`, `ocsvm_isolation_forest.py` und `ocsvm_ee_algorithm.py` (LOF, Isolation Forest, χ², MCD; wortgleich aus den Vorgängern), `ocsvm_ecod.py` (ECOD als Kontrast),
`ocsvm_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen, Korrelationsbruch), `ocsvm_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Anomalie-Erkennung: Ellipse bis Autoencoder](https://sebastianhanisch.net/konzepte-anomalie-erkennung.html)."
)
