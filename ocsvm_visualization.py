"""Plotly-Visualisierungen der One-Class-SVM-Demo: Touren in der Ebene der größten Streuung, der Kernel und die Ähnlichkeit zu einer Tour, die gelernte Grenze in der Ebene zweier Merkmale, Wert-Histogramm mit Schwelle,
ROC-Kurven, Kennzahlen-Balken, Sweeps und die Experimente (Szenarien, γ × ν-Raster, ν-Kalibrierung, Lücke, Standardisierung, Schwelle, Kosten). Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ocsvm_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, PINK = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
DETECTOR_COLORS = {"ocsvm": PINK, "lof": ORANGE, "iforest": PURPLE, "robust": TEAL, "ecod": GREEN}
DETECTOR_NAMES = {"ocsvm": "One-Class SVM", "lof": "LOF", "iforest": "Isolation Forest", "robust": "robust (MCD)", "ecod": "ECOD (zweiseitig)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke", "decorrelated": "Korrelationsbruch"}
DETECTORS = ("ocsvm", "lof", "iforest", "robust")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_score_hist(values, anomaly, thr, title="One-Class-SVM-Wert −f"):
    """Histogramm des One-Class-SVM-Werts −f (Schwelle: f < 0) (normale Touren blass, Sonderfahrten kräftig); senkrechte Linie = Schwelle."""
    hi = float(max(values.max(), thr * 1.05))
    lo = float(min(values.min(), thr * 0.95))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 50.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.55, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values[anomaly], xbins=bins, marker_color=RED, opacity=0.9, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=title), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_rank_compare(svm_values, robust_values, anomaly):
    """Rang der One-Class SVM gegen Rang der Wurzel (robuste Mahalanobis-Abstände) je Tour: Touren, die die One-Class SVM und die Wurzel gleich einschätzen, liegen auf der Diagonalen; Sonderfahrten als Rauten."""
    n = len(svm_values)
    re_ = np.argsort(np.argsort(svm_values)) / (n - 1)
    rr = np.argsort(np.argsort(robust_values)) / (n - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr[~anomaly], y=re_[~anomaly], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.5), name="normale Touren", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rr[anomaly], y=re_[anomaly], mode="markers", marker=dict(size=9, color=RED, symbol="diamond"), name="Sonderfahrten", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Rang bei der Wurzel (robuste Mahalanobis-Abstände)", range=[-0.02, 1.02]),
                      yaxis=dict(title="Rang bei der One-Class SVM", range=[-0.02, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der vier Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der vier Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der vier Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der vier Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


# --- Schritte: Kernel und Grenze ---------------------------------------------------------------------------------------------------------


def build_kernel(Z, gamma, factors, dist_marks):
    """Der RBF-Kernel exp(-gamma * d²) über den Abstand d (standardisierte Kennzahlen) für gamma = Faktor * gamma_0 mit den gewählten Faktoren (der aktuelle ist dick), dazu senkrechte Linien bei typischen Abständen
    (`dist_marks` = [(Abstand, Text)]: z. B. zum nächsten Nachbarn und zu einer zufälligen anderen Tour). Z wird nur für die Achsenlänge gebraucht."""
    dmax = float(max(m[0] for m in dist_marks)) * 1.6
    d = np.linspace(0.0, dmax, 200)
    fig = go.Figure()
    for f, gamma_f, current in factors:
        fig.add_trace(go.Scatter(x=d, y=np.exp(-gamma_f * d ** 2), mode="lines", line=dict(color=PINK if current else GRAY, width=4 if current else 1.5, dash="solid" if current else "dot"),
                                 name=f"γ = {f:g} · γ₀" + (" (gewählt)" if current else ""), hoverinfo="skip"))
    for dist, text in dist_marks:
        fig.add_vline(x=float(dist), line=dict(color=BLUE, dash="dash"), annotation_text=text, annotation_position="top")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="Abstand zweier Touren (standardisierte Kennzahlen)"), yaxis=dict(title="Ähnlichkeit K", range=[0, 1.05]),
                      legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_similarity(P, anomaly, similarity, idx):
    """Touren in der Projektionsebene, gefärbt nach der Kernel-Ähnlichkeit zur gewählten Tour (Raute mit schwarzem Rand)."""
    fig = go.Figure(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", marker=dict(size=7, color=similarity, colorscale="Blues", cmin=0.0, cmax=1.0, showscale=True, colorbar=dict(title="K", thickness=10)), hoverinfo="skip",
                               showlegend=False))
    fig.add_trace(go.Scatter(x=[P[idx, 0]], y=[P[idx, 1]], mode="markers", marker=dict(size=15, color=RED if anomaly[idx] else GREEN, symbol="diamond", line=dict(width=2, color="black")), hoverinfo="skip", name="gewählte Tour"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_boundary(xs, ys, F, S, X2, anomaly, sv, names):
    """Gelernte Grenze in der Ebene zweier Rohmerkmale (die One-Class SVM wird dafür auf nur diese zwei Merkmale neu trainiert): links die Entscheidungsfunktion f (schwarze Linie f = 0, außerhalb f < 0),
    rechts zum Vergleich der Score des Isolation Forest (Linie bei 0.5). Umkreiste Punkte sind Stützvektoren."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("One-Class SVM: Entscheidungsfunktion f", "Isolation Forest: Score"), horizontal_spacing=0.10)
    lim = max(float(F.max()), 1e-9)                                                          # Farbskala symmetrisch um 0, nach außen (f < 0) gesättigt, damit das Innere sichtbar bleibt
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=F, colorscale="RdBu", zmid=0.0, zmin=-lim, zmax=lim, showscale=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Contour(x=xs, y=ys, z=F, contours=dict(start=0.0, end=0.0, size=1.0, coloring="none"), line=dict(color="black", width=3), showscale=False, showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=S, colorscale="YlOrRd", zmin=0.3, zmax=0.8, showscale=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Contour(x=xs, y=ys, z=S, contours=dict(start=0.5, end=0.5, size=1.0, coloring="none"), line=dict(color="black", width=3), showscale=False, showlegend=False, hoverinfo="skip"), row=1, col=2)
    normal = ~anomaly
    for col in (1, 2):
        fig.add_trace(go.Scatter(x=X2[normal, 0], y=X2[normal, 1], mode="markers", marker=dict(size=5, color=BLUE, line=dict(width=0.5, color="white")), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=X2[anomaly, 0], y=X2[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond", line=dict(width=0.5, color="white")), hoverinfo="skip", showlegend=False), row=1, col=col)
    if sv.any():
        fig.add_trace(go.Scatter(x=X2[sv, 0], y=X2[sv, 1], mode="markers", marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=1.5, color="black")), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.update_xaxes(title=names[0])
    fig.update_yaxes(title=names[1], col=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------


def _heat(fig, cells, key, col, title, zmax=1.0):
    nus = sorted({c["nu"] for c in cells})
    gs = sorted({c["gamma_factor"] for c in cells})
    z = [[next(c[key] for c in cells if c["nu"] == nu and c["gamma_factor"] == g) for g in gs] for nu in nus]
    fig.add_trace(go.Heatmap(x=[f"{g:g}" for g in gs], y=[f"{nu:g}" for nu in nus], z=z, zmin=0.0, zmax=zmax, colorscale="Viridis", text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=9),
                             showscale=False, hoverinfo="skip"), row=1, col=col)


def build_grid(cells):
    """γ × ν-Raster der One-Class SVM: AUC (links) und F1 an der Schwelle f < 0 (rechts); Zeilen ν, Spalten γ in Vielfachen von γ₀."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 bei der Schwelle f < 0"), horizontal_spacing=0.10)
    _heat(fig, cells, "auc", 1, "AUC")
    _heat(fig, cells, "f1", 2, "F1")
    fig.update_xaxes(title="γ (Vielfaches von γ₀)", type="category")
    fig.update_yaxes(title="ν", type="category", col=1)
    fig.update_yaxes(type="category", col=2)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_nu(rows):
    """Anteil der markierten Trainingspunkte (durchgezogen) und der Stützvektoren (gestrichelt) über ν bei (fast) reinen Normalen (1 % Anomalien) und im Standardfall (10 %); die Diagonale ist ν."""
    xs = [r["nu"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", line=dict(color="black", dash="dot"), name="ν", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["pure_flagged"] for r in rows], mode="lines+markers", name="markiert (fast reine Normale)", line=dict(color=BLUE, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["pure_support"] for r in rows], mode="lines+markers", name="Stützvektoren (fast reine Normale)", line=dict(color=BLUE, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["std_flagged"] for r in rows], mode="lines+markers", name="markiert (Standardfall)", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["std_support"] for r in rows], mode="lines+markers", name="Stützvektoren (Standardfall)", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="ν", range=[0, 0.52]), yaxis=dict(title="Anteil der Trainingstouren", range=[0, 0.6]), legend=dict(orientation="h", y=-0.45))
    return lock_axes(fig)


def build_gap(rows):
    """AUC der One-Class SVM in der Lücke zwischen den Betriebsarten über γ (Vielfaches von γ₀), für 2 und 3 Betriebsarten und ν = 0.1 bzw. 0.3; die gepunktete Linie bei 0.5 ist Raten."""
    fig = go.Figure()
    colors = {(2, 0.10): PINK, (2, 0.30): ORANGE, (3, 0.10): TEAL, (3, 0.30): PURPLE}
    for r in rows:
        gs = sorted(r["by_gamma"], key=float)
        fig.add_trace(go.Scatter(x=[f"{float(g):g}" for g in gs], y=[r["by_gamma"][g] for g in gs], mode="lines+markers", name=f"{r['n_modes']} Betriebsarten, ν = {r['nu']:g}", line=dict(color=colors[(r["n_modes"], r["nu"])], width=3),
                                 hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="γ (Vielfaches von γ₀)", type="category"), yaxis=dict(title="AUC in der Lücke", range=[0, 1.02]), legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


def build_standardise(rows):
    """One-Class SVM und LOF mit und ohne Standardisierung: AUC und F1 an der Schwelle."""
    fig = go.Figure()
    labels = ["One-Class SVM AUC", "One-Class SVM F1", "LOF AUC", "LOF F1"]
    for r, name, color in ((rows[0], "standardisiert", BLUE), (rows[1], "Rohdaten", GRAY)):
        y = [r["ocsvm_auc"], r["ocsvm_f1"], r["lof_auc"], r["lof_f1"]]
        fig.add_trace(go.Bar(x=labels, y=y, name=name, marker_color=color, text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_scenarios(rows):
    """AUC der vier Detektoren und von ECOD (zweiseitig, Kontrast) in den Szenarien (links) und F1 mit bekanntem Anteil (rechts); die Linie bei 0.5 ist Raten."""
    labels = [r["scenario"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC", "F1 mit bekanntem Anteil"), horizontal_spacing=0.10)
    for det in DETECTORS + ("ecod",):
        for col, key in ((1, f"{det}_auc"), (2, f"{det}_oracle_f1")):
            y = [r[key] for r in rows]
            fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], showlegend=col == 1, text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=7),
                                 hoverinfo="skip"), row=1, col=col)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    fig.update_yaxes(range=[0, 1.15])
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(height=540, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_cutoff(rows):
    """One-Class SVM: F1 (durchgezogen), Recall (gepunktet) und Fehlalarmrate (gestrichelt) über ν bei der Schwelle f < 0."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["ocsvm_f1"] for r in rows], mode="lines+markers", name="F1", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ocsvm_recall"] for r in rows], mode="lines+markers", name="Recall", line=dict(color=PINK, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ocsvm_false_alarm"] for r in rows], mode="lines+markers", name="Fehlalarmrate", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ocsvm_auc"] for r in rows], mode="lines+markers", name="AUC", line=dict(color=GRAY, width=2), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="ν", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_costs(times):
    """Rechenzeit (Modell anpassen und Wert je Tour bestimmen) der One-Class SVM, des LOF und des Isolation Forest über die Tourenzahl bei 12 und 52 Merkmalen; über den Balken der One-Class SVM der Speicher der Kernmatrix."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("12 Merkmale", "52 Merkmale (40 Rauschmerkmale)"), horizontal_spacing=0.10)
    for col, p in ((1, 12), (2, 52)):
        rows = [t for t in times if t["p"] == p]
        for key, color, name in (("ocsvm", PINK, "One-Class SVM"), ("lof", ORANGE, "LOF"), ("iforest", PURPLE, "Isolation Forest")):
            y = [t[key] for t in rows]
            texts = [f"{v:.4f} s" + (f" · {t['kernel_mb']:.1f} MB" if key == "ocsvm" else "") for v, t in zip(y, rows)]
            fig.add_trace(go.Bar(x=[f"n = {t['n']}" for t in rows], y=y, name=name, marker_color=color, showlegend=col == 1, text=texts, textposition="outside", textfont=dict(size=8), hoverinfo="skip"), row=1, col=col)
    fig.update_yaxes(title="Sekunden", col=1)
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
