"""Defaults, Regler-Grenzen und Presets für die One-Class-SVM-Demo (Anomalie-Erkennung an Lieferrouten-Kennzahlen; Szenario, LOF, Isolation Forest und Vergleichsschätzer aus den Vorgänger-Demos)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
DEFAULT_N_NOISE = 0
N_NOISE_MIN, N_NOISE_MAX = 0, 40
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap", "decorrelated")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten", "decorrelated": "Korrelationsbruch (Randverteilungen normal)"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- One-Class SVM und Vergleichsdetektoren -----------------------------------------------------------------------------------
DEFAULT_NU = 0.10                                       # nu: obere Schranke des Anteils markierter Trainingspunkte, untere Schranke des Anteils der Stützvektoren
NU_MIN, NU_MAX = 0.01, 0.50
DEFAULT_GAMMA_FACTOR = 1.0                              # gamma als Vielfaches der Skala gamma_0 = 1 / (p * Var(X)) = 1 / p bei standardisierten Daten
GAMMA_FACTORS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
GAMMA_MIN, GAMMA_MAX = 0.02, 32.0
DEFAULT_TREES = 100                                     # Isolation Forest im Vergleich
DEFAULT_PSI = 256
DEFAULT_CUTOFF_IF = 0.5                                 # nominelle Score-Schwelle des Isolation Forest
DEFAULT_CUTOFF_LOF = 1.5                                # LOF-Schwelle im Vergleich
DEFAULT_LOF_K = 20                                      # LOF im Vergleich: k = 20 (höchstens n / 2)
THRESHOLD_KINDS = ("standard", "share")
THRESHOLD_LABELS = {"standard": "Standard (f < 0 bzw. LOF 1.5 bzw. Score 0.5 bzw. χ²-Quantil)", "share": "erwarteter Anteil (für alle vier)"}
DEFAULT_THRESHOLD_KIND = "standard"
DEFAULT_SHARE = 10                                      # angenommener Anteil der Anomalien [%] (= der wahre im Standardfall)
SHARE_MIN, SHARE_MAX = 1, 45
SMO_TOL = 1e-6                                          # KKT-Toleranz des Lösers (strenger als libsvm mit 1e-3, damit die Schranken von nu bis auf Rundung halten)
FLAG_EPS = 1e-6                                         # Markierung bei f < -FLAG_EPS: freie Stützvektoren liegen numerisch bei f = 0
SMO_MAX_ITER = 100_000

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_noise=DEFAULT_N_NOISE, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION,
                kind=DEFAULT_KIND, strength=DEFAULT_STRENGTH, nu=DEFAULT_NU, gamma=DEFAULT_GAMMA_FACTOR, threshold_kind=DEFAULT_THRESHOLD_KIND, quantile=DEFAULT_QUANTILE, share=DEFAULT_SHARE, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleines γ (0.1 · γ₀)": _preset(gamma=0.1),
    "Auswendiglernen (γ = 8 · γ₀)": _preset(gamma=8.0),
    "40 Rauschmerkmale, kleines γ": _preset(n_noise=40, gamma=0.1),
    "Gekrümmter Normalbereich": _preset(curvature=1.0),
    "Lücke, 3 Betriebsarten": _preset(n_modes=3, kind="gap"),
    "Viele Anomalien (45 %), ν = 0.1": _preset(contamination=45),
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Im Mittel über fünf Aufnahmen: 300 Touren, 12 Merkmale, 10 % verstreute Anomalien, ν = 0.1 (der wahre Anteil) und die Voreinstellung γ = γ₀ = 1 / p (wie 'scale' in scikit-learn). AUC 0.95 (LOF, Isolation Forest und robust 1.00), "
                                    "an der Schwelle f < 0 F1 0.67 (Recall 0.52, praktisch keine Fehlalarme) gegen 0.94 (LOF), 0.96 (Isolation Forest) und 0.84 (robust): die Grenze ist bei dieser Breite zu eng gezogen.",
    "Kleines γ (0.1 · γ₀)": "Im Mittel über fünf Aufnahmen: dieselben Daten, aber der Kernel zehnmal breiter (γ = 0.1 · γ₀). AUC 1.00, F1 0.95 (Recall 0.95, Fehlalarmrate unter 1 %) - so gut wie LOF (0.94) und Isolation Forest (0.96), "
                            "und besser als die robuste Schätzung (0.84). Dieselbe Methode, nur ein anderer Regler: F1 0.67 → 0.95.",
    "Auswendiglernen (γ = 8 · γ₀)": "Im Mittel über fünf Aufnahmen: der Kernel achtmal schmaler. AUC nur 0.83, und die Schwelle f < 0 markiert keine einzige Tour (Recall 0.00, F1 0.00): 43 % der Touren werden Stützvektoren, "
                                    "jede Tour hat ihre eigene Kuppel und liegt auf der Grenze. Selbst mit bekanntem Anteil wäre F1 nur 0.20.",
    "40 Rauschmerkmale, kleines γ": "Im Mittel über fünf Aufnahmen: 40 unabhängige Rauschmerkmale, γ = 0.1 · γ₀. AUC 0.99, F1 0.92 (Recall 0.90, Fehlalarmrate 1 %) - gegen LOF 0.00 (Recall 0.00), Isolation Forest 0.56 (Recall 0.39) und robust 0.35 (AUC 0.88): "
                                    "der breite Kernel mittelt das Rauschen weg. Mit der Voreinstellung γ₀ ist F1 nur 0.39.",
    "Gekrümmter Normalbereich": "Im Mittel über fünf Aufnahmen: die normale Fläche ist gebogen (Krümmung 1), Voreinstellung γ₀. AUC 0.98, F1 0.85 - besser als LOF (0.74) und die robuste Ellipse (0.38, sie umschließt die Krümmung nicht), "
                                "aber hinter dem Isolation Forest (0.95). Die gelernte Grenze folgt der Krümmung, der Preis ist die Wahl von ν und γ.",
    "Lücke, 3 Betriebsarten": "Im Mittel über fünf Aufnahmen: drei Betriebsarten, 10 % Anomalien in der Lücke dazwischen. Die One-Class SVM hat als einzige eine Rangfolge (AUC 0.82; LOF 0.36, Isolation Forest 0.27, robust 0.38, ECOD 0.01 - alle unter Raten), "
                              "aber die Schwelle f < 0 findet die Lücke nicht (Recall 0.01, F1 0.01). Mit breiterem Kernel (γ ≤ 0.25 · γ₀) fällt auch sie unter Raten (AUC 0.00).",
    "Viele Anomalien (45 %), ν = 0.1": "Im Mittel über fünf Aufnahmen: 45 % Anomalien, aber ν = 0.1. AUC 0.32 - **unter Raten**: die Anomalien sind die Mehrheit der Trainingsdaten, die Grenze umschließt sie. Isolation Forest und ECOD 1.00, robust 0.87, LOF 0.64. "
                                      "Mit ν = 0.5 wäre die AUC 0.99 und F1 0.93: die One-Class SVM braucht ν größer als den Anteil - schon für die Rangfolge.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen der Detektoren (ocsvm_*, lof_*, iforest_*, robust_*) und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"ocsvm_auc": (0.85, 1.0), "ocsvm_f1": (0.4, 0.85), "lof_f1": (0.85, 1.0), "verdict": ("comparable", "others_win")},
    "Kleines γ (0.1 · γ₀)": {"ocsvm_auc": (0.95, 1.0), "ocsvm_f1": (0.85, 1.0), "verdict": ("comparable",)},
    "Auswendiglernen (γ = 8 · γ₀)": {"ocsvm_auc": (0.6, 0.95), "ocsvm_f1": (0.0, 0.1), "ocsvm_recall": (0.0, 0.1), "verdict": ("others_win",)},
    "40 Rauschmerkmale, kleines γ": {"ocsvm_auc": (0.95, 1.0), "ocsvm_f1": (0.75, 1.0), "lof_recall": (0.0, 0.2), "verdict": ("ocsvm_wins",)},
    "Gekrümmter Normalbereich": {"ocsvm_auc": (0.9, 1.0), "ocsvm_f1": (0.7, 1.0), "robust_f1": (0.0, 0.6), "verdict": ("comparable", "others_win")},
    "Lücke, 3 Betriebsarten": {"ocsvm_auc": (0.6, 1.0), "lof_auc": (0.0, 0.6), "robust_auc": (0.0, 0.6), "ocsvm_recall": (0.0, 0.2), "verdict": ("gap_rank",)},
    "Viele Anomalien (45 %), ν = 0.1": {"ocsvm_auc": (0.0, 0.5), "iforest_auc": (0.9, 1.0), "verdict": ("nu_low",)},
}
