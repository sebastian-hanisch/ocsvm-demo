# One-Class SVM – eine gelernte Grenze um die Normalen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ocsvm-demo.streamlit.app/)**

Siebtes Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **One-Class SVM** (Schölkopf, Platt, Shawe-Taylor, Smola und Williamson) – an einem wachsenden Beispiel, gegen den [LOF](../lof-demo), den [Isolation Forest](../isolation-forest-demo)
und die robuste Schätzung der Wurzel ([elliptic-envelope-demo](../elliptic-envelope-demo)); [ECOD](../ecod-demo) tritt als Kontrast im Szenarien-Experiment auf. Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in den Vorgängern und der [pca-demo](../pca-demo)
(Szenario, LOF, Isolation Forest und Wurzel-Schätzer wortgleich übernommen, per Test gegen eingefrorene Werte geprüft; die klassische Schätzung entfällt, damit die Balken lesbar bleiben; die Anomalie-Art **Korrelationsbruch** kommt aus der ECOD-Demo).

**Einordnung in die Reihe (die Kanten des Graphen):** die One-Class SVM ist ein **eigener Ast direkt nach der Wurzel**. Die Wurzel kann nur **eine konvexe Ellipse**; die One-Class SVM lernt stattdessen eine **nichtlineare Grenze** (Kernel-Trick, RBF) und braucht dafür **zwei Regler, ν und γ**.
Ergebnis in Kürze: **bei passendem γ ist sie so gut wie LOF und Isolation Forest (F1 0.95) und bei 40 Rauschmerkmalen der beste Detektor an der Schwelle (F1 0.92 gegen 0.00 / 0.56 / 0.35); bei der Voreinstellung γ = 1 / p ist sie schwächer (F1 0.67), bei zu schmalem Kernel markiert die Schwelle nichts (F1 0.00), und ν muss größer als der Anteil der Anomalien sein – schon für die Rangfolge (AUC 0.32 bei 45 % Anomalien und ν = 0.1).**
Die Linie hat **keinen Konvergenzpunkt**; der Nachfolger ist [deepsvdd-demo](../deepsvdd-demo) (gelernte Merkmalsabbildung statt festem Kernel; Ergebnis: die Abhängigkeit von γ zieht auf Epochen, Breite und Netz-Seed um, und das Training verschlechtert die Rangfolge).
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ecod-demo                  (Kontrast: verteilungsfrei)                              [gebaut]
  ├─ lof-demo → feature-bagging-demo (lokale Dichte; Ensembles gegen viele Merkmale)     [beide gebaut]
  ├─ ocsvm-demo → deepsvdd-demo (gelernte Grenze; gelernte Abbildung)                    [dieses Stück; Deep SVDD gebaut]
  ├─ isolation-forest-demo → extended-isolation-forest-demo (Zufallsbäume)               [beide gebaut]
  └─ autoencoder-anomalie-demo  (Rekonstruktionsfehler)                                  [gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; One-Class SVM mit ν = 0.1 und γ = γ₀ = 1 / p (Voreinstellung wie 'scale' in scikit-learn), Schwelle f < 0, LOF k = 20 und Schwelle 1.5, Isolation Forest 100 Bäume × ψ = 256 und Schwelle 0.5, robust χ²-Quantil 0.975; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall, Voreinstellung | ⚠️ AUC **0.95** (LOF, Isolation Forest und robust 1.00), an der Schwelle F1 **0.67** (Recall 0.52, Fehlalarmrate 0.2 %) gegen 0.94 (LOF), 0.96 (Isolation Forest) und 0.84 (robust); mit bekanntem Anteil 0.67 gegen 0.97. Die Grenze ist bei dieser Breite zu eng gezogen |
| **γ ist der entscheidende Regler** | ✅/❌ F1 bei γ = 0.02 / 0.05 / 0.1 / 0.25 / 0.5 / 1 / 2 / 4 / 8 / 16 / 32 · γ₀: **0.95** / 0.96 / 0.95 / 0.90 / 0.81 / 0.67 / 0.44 / 0.03 / **0.00** / 0.00 / 0.00 (AUC 1.00 bis 0.60). Bei 8 · γ₀ werden 43 % der Touren Stützvektoren („Auswendiglernen“), die Schwelle markiert **keine einzige Tour**. Mit breitem Kernel (γ = 0.1 · γ₀) ist die One-Class SVM so gut wie LOF und Isolation Forest |
| ν und die Schwelle | ⚠️ ν ist eine **obere Schranke** des Anteils markierter Trainingstouren und eine **untere Schranke** des Stützvektor-Anteils – in jeder gemessenen Zelle erfüllt (bei fast reinen Normalen ν = 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5: markiert 0.0 / 1.9 / 6.3 / 16.9 / 27.9 / 48.7 %, Stützvektoren 12.3 / 13.0 / 15.7 / 23.3 / 32.6 / 51.3 %). Die Schwelle markiert also **so viele Touren, wie ν erlaubt**, unabhängig davon, ob es Anomalien gibt (bei ν = 0.2 und fast reinen Normalen Fehlalarmrate 16 %). F1 bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5: 0.00 / 0.00 / 0.03 / 0.67 / 0.71 / 0.51 / 0.34 (AUC 0.89 / 0.89 / 0.90 / 0.95 / 1.00 / 1.00 / 1.00) |
| γ × ν | ➖ AUC = 1.00 für ν ≥ 0.2 über einen breiten Bereich von γ; das gute F1 liegt aber auf einem **schmalen Band**: F1 ≥ 0.85 nur bei ν = 0.1 mit γ ≤ 0.25 · γ₀ (0.90–0.96), bei ν = 0.2 mit γ = 8 · γ₀ (0.87) und ν = 0.3 mit γ = 16 · γ₀ (0.87) – ν und γ wirken zusammen |
| **ν muss größer als der Anteil sein** | ❌ Bei festem ν = 0.1 ist die AUC bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % Anomalien 1.00 / 1.00 / 0.95 / 0.77 / 0.56 / **0.37 / 0.32 – unter Raten** (LOF 1.00 … 0.64, robust 1.00 … 0.87, Isolation Forest und ECOD überall etwa 1.00): die Grenze wird auf den Trainingsdaten gelernt und umschließt die Anomalien mit. Mit ν = 0.5 bei 45 %: AUC 0.99, F1 0.93. Selbst bei ν = wahrer Anteil (10 %) ist die AUC nur 0.95 – erst ν = 0.2 macht sie perfekt |
| Gekrümmter Normalbereich | ✅ Die gelernte Grenze folgt der Krümmung: F1 bei Krümmung 0 / 0.25 / 0.5 / 0.75 / 1 steigt von 0.67 auf 0.78 / 0.84 / 0.85 / 0.85 (AUC 0.99–0.98), während der LOF von 0.94 auf 0.74 und die robuste Ellipse von 0.84 auf 0.38 fallen; der Isolation Forest bleibt bei 0.95–0.98 |
| Rauschmerkmale | ✅/❌ Bei γ₀: AUC 0.95 → 0.91, Recall 0.52 → 0.25, F1 0.67 → 0.39 bei 40 Rauschmerkmalen (LOF-Recall 0.00, robust AUC 0.88). **Mit γ = 0.1 · γ₀ bleibt AUC 0.99–1.00 und F1 0.92–0.95: bei 40 Rauschmerkmalen der beste Detektor an der Schwelle** (LOF 0.00, Isolation Forest 0.56, robust 0.35) – der breite Kernel mittelt unabhängiges Rauschen weg |
| Dichte Gruppe | ➖ AUC bei 10 % / 30 % (ν = 0.1): 0.71 / 0.74 (LOF 0.35 / 0.47, Isolation Forest 0.95 / 0.70, robust 1.00 / 0.49, **ECOD 0.98 / 0.84**). Mit ν = 0.3 und γ = 0.1 · γ₀ bei 30 %: F1 0.46 gegen 0.24 (Isolation Forest), 0.07 (robust), 0.00 (LOF) |
| Lücke zwischen Betriebsarten | ⚠️ **Als einziger Detektor deutlich über Raten**: AUC 0.64 (2 Betriebsarten) und 0.82 (3) bei γ₀ (LOF 0.53 / 0.36, Isolation Forest 0.54 / 0.27, robust 0.40 / 0.38, ECOD 0.04 / 0.01) – aber die Schwelle findet die Lücke kaum (Recall 3 % / 1 %). Bei breitem Kernel (γ ≤ 0.25 · γ₀) ist die AUC **unter Raten** (0.02 bzw. 0.00): die Lücke liegt im Innern der Grenze, wie bei der Ellipse |
| Korrelationsbruch | ⚠️ AUC **0.88** (γ₀; F1 0.30) gegen LOF 0.99, robust 1.00, Isolation Forest 0.80, ECOD 0.56. Mit ν = 0.3 und γ = 2 · γ₀ AUC 0.98, aber F1 nur 0.57, weil die Schwelle ν der Touren markiert |
| Kleine Stichproben | ❌ Bei ν = 0.1 markiert die Schwelle bei 20–50 Touren **nichts** (F1 0.00; AUC 0.76 / 0.88 / 0.86); F1 bei 100 / 200 / 400 / 600 Touren 0.10 / 0.64 / 0.72 / 0.76 (LOF 0.87 / 0.91 / 0.94 / 0.90) |
| Standardisierung | ➖ Ohne sie sinkt die AUC von 0.95 auf 0.79 (F1 unverändert bei 0.67 / 0.68); LOF von 1.00 auf 0.87 (F1 0.94 → 0.68) |
| Falsch angenommener Anteil | ➖ bringt alle vier auf fast denselben F1: F1 bei ½× / 1× / 2× des wahren Anteils One-Class SVM 0.64 / 0.67 / 0.63, LOF und Isolation Forest 0.67 / 0.97 / 0.67 |
| Rechenzeit und Speicher | ➖ Im gemessenen Bereich (bis 600 Touren) höchstens einige zehn ms (bei 600 Touren etwa 13–22 ms, rechnerabhängig) – so viel wie der LOF (etwa 15–20 ms bei 600 Touren), weit weniger als der Isolation Forest (etwa 70–620 ms). Die Kernmatrix wächst quadratisch: 0.08 / 0.72 / 2.9 MB bei 100 / 300 / 600 Touren; bei 10 000 Touren wären es 800 MB (gerechnet, nicht gemessen) |

## Was die Demo zeigt

1. **One-Class SVM in Aktion** (Schritt-Slider + Abspielen): **Touren** (Ebene der größten Streuung und zwei Rohmerkmale) → **Kernel** (die Ähnlichkeit exp(−γ d²) über den Abstand mit den Stufen des Reglers und typischen Abständen; Ähnlichkeit einer Tour zu allen anderen) →
   **Grenze** (Entscheidungsfunktion f in der Ebene zweier Rohmerkmale, dafür neu trainiert, mit der Linie f = 0 und den umkreisten Stützvektoren, daneben der Score des Isolation Forest) → **Wert** (Verteilung von −f mit Schwelle; Rang bei der One-Class SVM gegen Rang bei der Wurzel) → **Ergebnis** (Kennzahlen und ROC-Kurven der vier Detektoren).
2. **Was die Detektoren gefunden haben:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke nur in der Rangfolge → Lücke unentdeckt → ν kleiner als der Anteil → anderer Detektor besser → One-Class SVM besser (Rangfolge oder Schwelle) → falsche Schwelle bei guter Rangfolge → ebenbürtig), Detailtabelle mit Rechenzeiten.
3. **📐 Sweeps** über Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, **ν**, **γ**, χ²-Quantil und angenommenen Anteil (feste Datensätze ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** acht Szenarien × vier Detektoren und ECOD, das γ × ν-Raster, ν-Kalibrierung (Schranken der Theorie), die Lücke über γ, Rauschmerkmale für zwei γ und Standardisierung, Schwelle (ν und falsch angenommener Anteil), Kosten.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (ν größer als der Anteil, γ passt zur Struktur, die Schwelle ist ein Anteil, Standardisierung, irrelevante Merkmale, Abhängigkeit, Lücke, Speicher).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten / Korrelationsbruch), Abstand (bei "Lücke" und "Korrelationsbruch" ausgeblendet, Wert bleibt erhalten),
**ν** (0.01–0.5), **γ** (0.02 bis 32 · γ₀, γ₀ = 1 / p), **Schwelle** (Standard: f < 0 und χ²-Quantil der Wurzel, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle vier).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC SVM | F1 SVM | AUC LOF | F1 LOF | AUC IF | F1 IF | AUC robust | F1 robust | Urteil |
|---|---|---|---|---|---|---|---|---|---|
| Standardfall (Voreinstellung) | 0.97 | 0.69 | 1.00 | 0.97 | 1.00 | 0.98 | 1.00 | 0.85 | ebenbürtig (bei anderen Seeds: anderer Detektor besser) |
| Kleines γ (0.1 · γ₀) | 1.00 | 0.92 | 1.00 | 0.97 | 1.00 | 0.98 | 1.00 | 0.85 | ebenbürtig |
| Auswendiglernen (γ = 8 · γ₀) | 0.82 | 0.00 | 1.00 | 0.97 | 1.00 | 0.98 | 1.00 | 0.85 | anderer Detektor besser |
| 40 Rauschmerkmale, kleines γ | 0.98 | 0.90 | 0.98 | 0.06 | 0.99 | 0.42 | 0.90 | 0.36 | One-Class SVM besser (Schwelle) |
| Gekrümmter Normalbereich | 1.00 | 0.85 | 0.99 | 0.74 | 1.00 | 0.97 | 1.00 | 0.39 | ebenbürtig |
| Lücke, 3 Betriebsarten | 0.82 | 0.04 | 0.30 | 0.00 | 0.32 | 0.00 | 0.31 | 0.00 | nur in der Rangfolge sichtbar |
| Viele Anomalien (45 %), ν = 0.1 | 0.37 | 0.26 | 0.64 | 0.16 | 1.00 | 0.83 | 0.96 | 0.85 | ν kleiner als der Anteil |

## Modell und Verfahren

- **Szenario** (`ocsvm_scenario.py`): wortgleich aus den Vorgängern (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke / Korrelationsbruch mit exaktem Anteil, Zusatzmerkmale bis p = 30, Rauschmerkmale).
- **One-Class SVM** (`ocsvm_algorithm.py`, numpy von Grund auf): duales Problem in der Skalierung von libsvm, min ½ αᵀKα mit 0 ≤ α ≤ 1 und Σα = ν n, RBF-Kernel; Löser **SMO** mit Paarauswahl nach zweiter Ordnung, Anfangswert wie libsvm, ρ aus den freien Stützvektoren, Toleranz 1e-6;
  Entscheidungsfunktion f(x) = Σ α_i K(x_i, x) − ρ, Anomalie bei f < 0. γ ist ein Vielfaches von γ₀ = 1 / (p · Var(X)) (bei standardisierten Daten 1 / p). Nicht konvergierte Läufe werden gemeldet, nicht verschluckt.
- **Isolation Forest** (`ocsvm_isolation_forest.py`), **LOF** (`ocsvm_lof.py`, k = 20, höchstens n / 2, standardisierte Merkmale), **ECOD** (`ocsvm_ecod.py`, zweiseitige Variante, nur als Kontrast) und **Wurzel** (`ocsvm_ee_algorithm.py`: χ² ohne scipy, klassisch, FastMCD): wortgleich aus den Vorgängern.
- **Auswertung** (`ocsvm_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate für vier Detektoren; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; Sweeps, Experiment-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen (vor dem Bau gemessen):** (1) "ν ist ein Anteil, den man kennen muss" – **bestätigt, und schärfer als gedacht**: schon für die **Rangfolge** (AUC 0.32 bei 45 % Anomalien und ν = 0.1), nicht nur für die Schwelle; selbst ν = wahrer Anteil reicht nicht (AUC 0.95).
  (2) "Es gibt ein γ-Fenster, die Standardskala liegt darin" – **widerlegt für die Skala**: F1 fällt monoton mit γ, das gute F1 (0.95) liegt bei einem zehnmal breiteren Kernel als die Voreinstellung. (3) "Gekrümmter Normalbereich: die One-Class SVM schlägt die Wurzel" – **bestätigt** (F1 0.85 gegen 0.38), auch gegen den LOF (0.74), nicht gegen den Isolation Forest (0.95).
  (4) "Lücke: mit passendem γ sichtbar" – **halb**: nur in der Rangfolge (AUC 0.64 / 0.82), nicht an der Schwelle (Recall 3 % / 1 %), und unter Raten bei breitem Kernel. (5) "Dichte Gruppe wird mit maskiert" – **bestätigt** (AUC 0.71 / 0.74 bei ν = 0.1; ν = 0.3 hilft an der Schwelle, nicht in der Rangfolge: 0.75 bei 30 %).
  (6) "Korrelationsbruch: wie LOF und robust sichtbar" – **halb**: AUC 0.88, mit passendem ν und γ 0.98, aber F1 höchstens 0.57. (7) "Rauschmerkmale verdünnen die Abstände wie beim LOF" – **nur für die Voreinstellung**; bei breitem Kernel bleibt F1 0.92. (8) "Viele Anomalien: ν muss steigen" – bestätigt. (9) "Ohne Standardisierung dominiert das größte Merkmal" – bestätigt (AUC 0.95 → 0.79).
  (10) "O(n²) macht die Methode teuer" – **im gemessenen Bereich nicht**: höchstens einige zehn ms bei 600 Touren (etwa 13–22 ms, rechnerabhängig); nur der Speicher (8 n² Byte) wächst spürbar, und das ist gerechnet, nicht gemessen.
- **Ein erster Lauf mit der Toleranz von libsvm (1e-3) und der Schwelle f < 0 täuschte:** freie Stützvektoren liegen numerisch bei f = 0 ± 1e-4, etwa die Hälfte wäre markiert worden, der Anteil markierter Touren lag über ν (11.3 % bei ν = 0.1) und die Fehlalarmrate bei 4.6 % statt 0.2 %. Mit Toleranz 1e-6 und einer Schwelle knapp unter 0 halten die Schranken der Theorie in jeder Zelle, und der F1 der Voreinstellung ist 0.67 statt 0.66 mit falschen Fehlalarmen.
- **Bei kleinem ν und kleinem n markiert die Schwelle nichts:** ν n ≤ 5 Touren (bis 50 Touren) → F1 0.00; auch bei ν = 0.01 / 0.02 im Standardfall.
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Rauschmerkmale sind unabhängige Spalten. Das gute Abschneiden des breiten Kernels gilt für dieses Szenario (niedrigdimensionale latente Struktur); das beste γ hängt von den Daten ab (Lücke, Krümmung und Korrelationsbruch wollen ein größeres γ). Literatur nur mit Namen: Schölkopf, Platt, Shawe-Taylor, Smola und Williamson (One-Class SVM); Fan, Chen und Lin (SMO mit Auswahl zweiter Ordnung, libsvm).

## Verifikation

- One-Class SVM: Handinstanzen (zwei Punkte mit ν = 1: α = (1, 1), ρ = 1 + e^(−γ); drei symmetrische Punkte mit geschlossener Lösung), **Dualbedingungen und KKT** (Σα = νn, 0 ≤ α ≤ 1, freie Stützvektoren f = 0, Nicht-Stützvektoren f ≥ 0, obere Schranke f ≤ 0), **die Schranken für ν** (markierter Anteil ≤ ν ≤ Stützvektor-Anteil, 24 Konfigurationen),
  Kernel-Eigenschaften (symmetrisch, positiv semidefinit, Diagonale 1), Grenzfälle von γ (riesig: jede Tour Stützvektor mit gleichem Gewicht; winzig: Rangfolge nach Abstand vom Zentrum), Determinismus, Iterationslimit wird gemeldet, ungültiges ν abgelehnt, **Kreuzprüfung gegen `sklearn.svm.OneClassSVM`** (6 Konfigurationen, Entscheidungsfunktion < 1e-6, α < 1e-5, ρ, Zahl der Stützvektoren).
- Übernommene Bausteine: LOF, Isolation Forest, Wurzel-Schätzer, ECOD-Schwanzwahrscheinlichkeiten. Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `n_noise` ändert nur angehängte Spalten, exakter Anomalie-Anteil, Geometrie der Arten, Korrelationsbruch (Spalten der Anomalien stammen aus den Normalen, Korrelation zerstört, Normale bit-identisch).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Szenarien-, γ × ν-, ν-Kalibrierungs-, Lücke-, Standardisierungs-, Schwellen- und Kostentabellen; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung/Verhältnis);
  alle 7 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, extreme γ und jede Art der Anomalien, Abspielen ohne doppelte Schlüssel, ausgeblendete Regler behalten ihre Werte, Sweep-Optionen folgen Schwellenart und Art der Anomalien, Experimente auf Abruf),
  Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Szenarien, γ × ν, ν-Kalibrierung, Lücke, Rauschmerkmale und Standardisierung, Schwelle, Kosten), 🚧 Grenzen, Mathe |
| `ocsvm_algorithm.py` | RBF-Kernel, SMO-Löser, Entscheidungsfunktion, γ₀ |
| `ocsvm_lof.py`, `ocsvm_isolation_forest.py`, `ocsvm_ee_algorithm.py`, `ocsvm_ecod.py` | LOF, Isolation Forest, χ²-Verteilung, klassische Schätzung, FastMCD, ECOD (wortgleich aus den Vorgängern) |
| `ocsvm_scenario.py`, `ocsvm_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien (auch Korrelationsbruch) und Rauschmerkmalen; Konstanten, Presets |
| `ocsvm_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Urteil |
| `ocsvm_presets.py`, `ocsvm_visualization.py` | Permalink/Presets (ausgeblendete Regler), Plotly-Figuren (achsengesperrt) |
| `tests/` | One-Class SVM (Handinstanzen, KKT, ν-Schranken, sklearn-Kreuzprüfung), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Anomalie-Erkennung: Ellipse bis Autoencoder](https://sebastianhanisch.net/konzepte-anomalie-erkennung.html).
