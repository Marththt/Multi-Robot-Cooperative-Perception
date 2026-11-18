# CONTESTO — Multi-Robot Cooperative Perception (tesi magistrale TLC, UnivAQ)

Stato al 16/11/2026 (pomeriggio: YOLO + marker network diventa la soluzione proposta). Questo file riassume decisioni, struttura, bug risolti e punti aperti del progetto (codice + tesi).

---

## 0. Regole di lavoro

- Codice **solo in inglese** (identificatori, commenti, docstring, messaggi).
- Consegnare **solo i file modificati**, uno per uno (no zip del progetto).
- L'assistente **non fa commit/push**: git lo esegue Alessio (Mac e PC1).
- **Mai cancellare** nelle cartelle dell'utente: spostare in `_to_delete/`.
- Non usare né salvare token GitHub incollati in chat.
- Report: niente sezione "next steps", niente riferimenti a versioni precedenti.
- Inquadramento della tesi (prof. Pangun Park): **feature engineering**, cioè `target → encoder → feature → fusion`.

---

## 1. Obiettivo

Una squadra di 4 robot osserva una scena statica e deve stimare **quanti** bersagli ci sono e **dove** sono. Il problema centrale è l'**associazione**: capire quali osservazioni di robot diversi appartengono allo stesso bersaglio.

- I marker ArUco sono **tutti uguali** (id 0): danno la classe dell'oggetto, non la sua identità.
- Tutti i metodi partono dagli stessi dati (immagini + pose dei robot). Cambiano l'**encoder** (immagine → feature) e la **fusione** (feature dei robot → stima).
- Metrica: una scena è **corretta** se il conteggio è esatto e ogni scatola ha una stima entro 15 cm.

---

## 2. Piattaforma e dati reali

| Voce | Valore |
|---|---|
| Robot | 4 × DJI RoboMaster EP (P0, P1, P3, P5), ingombro 30×25 cm |
| Localizzazione | OptiTrack (Motive, NatNet), posa planare (x, y, ψ) |
| Camera | lente 24–26 cm da terra, 15 cm avanti al centro del telaio; pitch 10–13° (22° per un robot), variabile nel tempo |
| Scatole | prisma triangolare, lato **23.5 cm**, altezza 30 cm; apotema 6.78 cm, circumraggio 13.6 cm |
| Dati | **444 scene da 105 layout**: 4 round per layout (7 round per gli 8 layout da 4 scatole), 5 frame per sosta |

### Gruppi di layout (nomi definitivi; le lettere sono eliminate dalla tesi)

| Vecchio | Nuovo nome | Layout | Scene |
|---|---|---|---|
| L + A | **base** | 55 | 244 |
| B | **distance** | 12 | 48 |
| C | **obstacles** | 8 | 32 |
| D | **white objects** | 8 | 32 |
| E | **light** | 8 | 32 |
| M | **crowded** | 14 | 56 |

- **Le due campagne sono fuse** in un solo data set. **Ogni analisi usa tutte le 444 scene**, inclusi calibrazione e modello del sensore. Non esistono più i "layout di riferimento".
- **Livelli di difficoltà** (`tools/scene_levels.py`): calcolati solo dalla geometria (ground truth OptiTrack + `camera_pose` grezza), mai dall'output di un metodo.
  - **easy**: scatole ≥ 25 cm, ogni scatola in vista chiara di ≥ 2 robot, niente ostacoli né oggetti bianchi.
  - **medium**: come easy, ma una scatola è in vista chiara di un solo robot.
  - **hard**: scatole < 25 cm, oppure una scatola non vista chiaramente da nessuno, oppure ostacoli/oggetti bianchi.
  - Conteggi: easy 151, medium 77, hard 216.

---

## 3. Metodi e nomi definitivi

| Nome nella tesi | Cosa è |
|---|---|
| **MHT** | Sempre e solo MHT **con modello del sensore** (l'MHT base è eliminato; resta una frase: 64.2 → 73.4%) |
| **marker network** | GNN con un nodo per rilevamento, sui vettori di feature ArUco (distanza, bearing, angolo di vista, posizione) |
| **YOLO + marker network** | **Soluzione proposta.** Marker network + raggi di bearing delle scatole segmentate da YOLO-seg (iniezione per triangolazione + cannot-link), §5 |

- **Le reti BEV (RGB network, mask network, mask + marker network) sono eliminate** dal codice e dalla tesi (16/11/2026): codice in `_to_delete/bev_cnn_2026-11-16/`.

- Classici restanti nel capitolo 7: zone and class, union, PHD, CPHD, PHD/CPHD order average.
- **Demo live**: non entra nella tesi; serve solo per un video per la visita del relatore.

---

## 4. Simulatore a livello di misura (`sim/`)

Il simulatore non renderizza immagini: produce direttamente i rilevamenti ArUco, scritti negli stessi file e campi dei dati reali.

### Generazione della scena
- Famiglie **circle** e **random**, arena 6×6 m.
- Scena nominale: N = 6 robot, B = 5 scatole, s_min = 30 cm, R = 1.5 m, r_b = 0.7 m.

### Modello del sensore, in 5 passi
1. **Visibilità**: distanza 0.15–3 m, campo visivo 88°.
2. **Occlusione**: segmento dalla camera al centro del marker.
3. **Rilevamento**: P_D logistica in (α, α², d, occlusione).
4. **Errore di misura**: componenti lungo e trasversale alla linea di vista, più bias di scala e bearing per robot.
5. **Falsi rilevamenti**: Poisson per cattura.

### Parametri stimati su tutte le 444 scene (anteprima cloud, deterministica)

**`sensor_calib.json`**

| Parametro | Valore |
|---|---|
| Correzione distanza A, B | 1.083, −0.117 m |
| Correzione bearing k, β₀ | 1.033, −0.0083 rad |
| Residui | 3.7 cm lungo, 5.3 cm trasversale |
| σ | 4.6 cm |
| p_D | 0.64 (5578/8658, match entro 30 cm) |

**`sensor_model.json`**

| Parametro | Valore |
|---|---|
| Coefficienti P_D, w | (3.54, 2.28, −2.44, −1.74, −7.05) |
| Seconda faccia | q = 0.015 (14 eventi) |
| Media lungo la linea di vista | (4.51, −0.03, −5.72) cm |
| Rumore lungo | s = (0.97, 0.66) cm |
| Rumore trasversale | t = (5.45, −0.54) cm |
| Bias per robot | scala 2.2%, bearing 1.8% |
| Rumore angolo di vista | 5.9° |
| Falsi rilevamenti | 0.153 per cattura (28% vicini 1.1–3.5 m; 56% al bordo del campo visivo) |
| Rilevamento reale | 4758/8658 = 0.55; non occluse 0.84; occluse 0.005 |

### Fedeltà del simulatore
- Modello stimato su **metà A** dei layout, verificato su **metà B** con 50 ripetizioni.
- Le liste dei layout sono in `run_day8_0.bat`. Il gruppo obstacles è escluso.

### Scenari di test (nomi definitivi; le sigle sono eliminate dalla tesi)

| Vecchio | Nome | Livelli |
|---|---|---|
| G0 | nominal | — |
| G1 | spacing | s_min 20/30/45/60 cm |
| G2 | robots | N 4/8/16 |
| E1 | false detections | k_f 1/2/4/8 |
| E2 | second faces | k_2 1/2/5 |
| E3 | occlusions | h 0/1/2 |
| S1 | dense boxes | B 4/8/12/16 a 20 cm |
| V1 | many boxes | B 20/24 |
| V2 | many robots | N 20/24 |

---

## 5. Soluzione proposta: YOLO + marker network (`yolo_gnn/`)

Tutto addestrato **solo in simulazione**; le scene reali sono solo test.

- **Marker network**: `gnn/train_sim.py` su `data/plan_v2` (3 seed), invariata.
- **YOLO-seg** (`yolo_gnn/train_seg.py`): yolov8n-seg, immagini `imgsim` 960×540, 1500+200 scene, 40 epoche, imgsz 960. Etichette = regione visibile esatta di ogni scatola (occlusa da scatole, ostacoli, robot, gripper). Pitch [8°, 32°], ostacoli e distrattori casuali.
  - Il `best.pt` sul Mac era a 1 epoca (la 2ª è divergita su MPS): non usarlo.
- **Segmentazione reale** (`yolo_gnn/segment_real.py`): tutti i frame (5 letture) → `data/yolo_real/cache/<run>/<frame>_ids.png`. Gamma 1.8 fisso sui frame reali (più scuri del simulato).
- **Raggi** (`yolo_gnn/hybrid.py`): per ogni maschera, bearing dalla colonna mediana + range monoculare (contatto a terra e altezza angolare 30 cm). Pitch/yaw della camera dai marker (`--calibrate-scope other-layouts` di default; `same-run` come confronto). Un raggio vale se la scatola è segmentata in ≥ `min_frac` delle 5 letture.
- **Iniezione**: due raggi "non spiegati" (nessun marker dello stesso robot, nessuna scatola della marker network lungo il raggio) di robot diversi che si incrociano dentro il cerchio dei robot, ai range attesi → nuova scatola (un nodo per raggio, la GNN li raggruppa).
- **Cannot-link**: due nodi su raggi diversi dello stesso robot (≥ 0.12 rad) per `cl_robots` robot → p_same = 0.
- **Protocollo onesto** (`yolo_gnn/evaluate.py`): le soglie (`min_frac`, `tol_sym`, `cl_robots`, `same_robot`) sono scelte con **cross-validation a 2 fold sui layout** (layout alternati dentro ogni gruppo): le soglie migliori su una metà si applicano all'altra e viceversa. Ogni scena è valutata con soglie scelte senza di essa. Tutto il resto è fissato a priori.
  - Le vecchie versioni (`test_hybrid_v2…v5`, `eval_stratified*`) erano tarate direttamente sulle scene di test, con livelli assegnati per indice e un MHT finto (p_D 0.95): **non usare i loro numeri**.
- **Ablazioni** (stesso protocollo): marker / + same-robot (senza YOLO) / + iniezione / + cannot-link / completo. Diagnostica dei raggi in `rays.csv` (scatole in vista trovate da marker e da YOLO).
- **Anteprima sul Mac** (vecchio modello GNN, YOLO a 1 epoca, maschere solo per le 388 scene della seconda campagna): marker 80.3%, + iniezione 82.1%, + cannot-link 81.9%, **completo 83.5 ± 0.7%**; contro l'MHT McNemar p < 1e-16 su ogni seed. Numeri da rifare sul PC1.

## 6. Struttura dei file

### Macchine e percorsi

| Dove | Percorso |
|---|---|
| Mac repo | `~/Desktop/Tesi/tesi_codice /Multi-Robot-Cooperative-Perception` (nota lo spazio in `tesi_codice `) |
| PC1 (training) | `C:\GNN for vision fault\Multi-Robot-Cooperative-Perception` (`.venv`) |
| PC laboratorio | `Z:\Vision-fault\statico` |
| Tesi | `~/Desktop/Tesi/tesi_latex` (`cap/`, `fig/`, `main.tex`, `Thesis_draft.pdf`) |
| Overleaf | `~/Desktop/overleaf_tesi` (template del prof: `main.tex`, `include/settings/`, `cap/`, `fig/`) |

### Dati
- `data/survey_rounds/` (8 layout da 7 round) e `data/survey_v2_best/` (97 layout; run scelta per layout).
- `data/survey_all/` contiene un collegamento per ogni layout (105): junction su PC1, symlink in cloud.

### Moduli principali

| File | Ruolo |
|---|---|
| `pipeline/calibrate.py` | calibrazione distanza/bearing; accetta più cartelle (`dir nargs='+'`) |
| `sim/calibrate.py` | modello del sensore; `--survey-dir nargs='+'`, `--layouts all`, `--rare-events all\|calibration`, `--dump DIR` (CSV per le figure) |
| `sim/geometry.py` | `BOX_SIDE_M = 0.235`, `APOTHEM_M = BOX_SIDE_M/(2√3)` |
| `vf/fusion/sensor_aware_mht.py` | MHT con modello del sensore (`APOTHEM_M` = 0.235/(2√3)) |
| `sim/plan.py`, `sim/generate.py`, `sim/fidelity.py` | scenari di test, replay delle scene reali, confronto di fedeltà |
| `pipeline/evaluate_runs.py` | tutti i classici su una cartella di survey → `scenes.csv`, `summary.csv` |
| `gnn/train_sim.py` | marker network (detection graph, full features, h2, 3 seed) |
| `yolo_gnn/camera.py` | run reali, pose della camera, calibrazione pitch/yaw dai marker |
| `yolo_gnn/train_seg.py`, `yolo_gnn/segment.py`, `yolo_gnn/segment_real.py` | dataset YOLO simulato e training, segmenter, cache delle maschere reali |
| `yolo_gnn/hybrid.py`, `yolo_gnn/evaluate.py` | raggi, iniezione, cannot-link; valutazione con CV a 2 fold e ablazioni |
| `sim/scaling.py` | scene scalabili: robot e scatole casuali in un quadrato di lato √((N+B)/1.25) (min 3 m), heading verso il maggior numero di scatole; set train/val/tune e sweep SC (N=B=4…50), SR (B=16, N 4…50), SB (N=16, B 4…50) |
| `yolo_gnn/segment_sim.py` | rendering delle scene scalabili (imgsim) e segmentazione YOLO → cache delle label |
| `yolo_gnn/evaluate_sim.py` | MHT, marker network e YOLO + marker network sulle scene scalabili; parametri liberi (densità di nascita MHT, soglie YOLO) scelti sul set `tune` per F1 medio; stesse metriche del reale + F1 e tempo per scena; McNemar e sign test contro MHT; `fig_scaling.png` |
| `tools/scene_levels.py` | livelli delle scene |
| `tools/real_tables.py` | **tutte le tabelle reali della tesi** (gruppo / scatole / distanza / livello / causa), più McNemar (`--pair A:B`) e oracolo (`--oracle A:B`) |
| `tools/fig_layouts.py` | Figura 3.1 (un layout per gruppo, scatole triangolari orientate) |
| `tools/fig_measurements.py` | figura delle misure reali (da `--dump`) |
| `tools/fig_scene_rounds.py`, `tools/draw_robot.py` | figura dei round; robot 30×25 cm |

### Run del giorno 9 (`docs/paste_in_powershell_PC1_day9.txt`; il giorno 8 non è mai partito ed è in `_to_delete/`)

- **Prima di lanciare**: installa `ultralytics==8.4.213` vincolato alla versione di torch già presente (il pip fallisce invece di sostituire torch CUDA).
- **`logs/run_day9_all.bat`**: controlli sul codice e su ultralytics/CUDA, junction `data/survey_all`, blocco 0, poi A e Y in parallelo, attesa dei file `.done`, poi S (simulazione), infine H (reale).
- **`logs/run_day9_0.bat`**: come il vecchio blocco 0 → `results/evaluation_all`, `results/fidelity_all`, `results/sensor_model_dump`.
- **`logs/run_day9_A.bat`** → `results/plan_v2`: `sim.plan`, MHT, marker network (3 seed), oracolo, marker network su `real_all`. Niente RGB network. Poi `sim.scaling` (`data/scale/`) e una seconda marker network addestrata su plan_v2 + scale → `results/scale/gnn/`.
- **`logs/run_day9_S.bat`** → `results/scale/`: `segment_sim` (tune + tutti gli sweep, 3 letture per robot) e `evaluate_sim` con le due marker network (`plan`, `scale`).
- **`logs/run_day9_Y.bat`** → `results/yolo_seg/box_seg`: training YOLO-seg in simulazione e segmentazione di tutti i frame reali.
- **`logs/run_day9_H.bat`** → `results/yolo_gnn`, `results/yolo_gnn_same_run`, `results/yolo_gnn/real_tables.txt`: valutazione ibrida + tabelle con McNemar contro MHT e marker.

---

## 7. Bug e problemi risolti

| Problema | Soluzione |
|---|---|
| Scatola simulata con lato 30 cm (apotema 8.7 cm) invece di 23.5; stesso valore nell'MHT, quindi anche sui dati reali | `sim/geometry.py` e `sensor_aware_mht.py` portati a 23.5 cm; rerun completo (giorno 8) |
| Calibrazione e modello del sensore stimati solo su L4–L11 (non rappresentativi: q due facce 0.010 contro 0.061 sul resto) | stima su tutte le 444 scene |
| `select_runs` richiedeva il conteggio dei frame (assente nei tgz grezzi) | `survey_v2_best` costruito dalla lista run di classical_v3 (R24 finisce con "F") |
| `evaluate_runs` non ha `--layouts` | si valutano tutte le scene e si filtra dopo |
| Tesi: numeri errati (falsi per scena in ch7, "89%" nell'abstract, conteggi layout in ch3, "three methods" in ch10, "0→89" in ch11) | corretti |
| Overfull box su Overleaf (ch5, ch9, ch11) | corretti |
| Staging dal device limitato a 7 livelli di cartella | copia prima in `_to_delete/check_nov13/` |
| git in sola lettura sul device | `GIT_OPTIONAL_LOCKS=0` |
| zip esistente non sovrascrivibile sul Mac (niente permesso di cancellare) | nuovo nome con python `zipfile` |
| Demo live lenta e a scatti | interpolazione a 80 ms, thread di campionamento a 0.03 s, `image_worker` in un processo separato, mappa fissa, niente bordo dell'arena |

---

## 8. Numeri attuali (scene reali, 444)

### Classici: definitivi (anteprima cloud con la scatola nuova e il modello stimato su tutte le scene)

| Metodo | Scene corrette |
|---|---|
| MHT | **73.4%** (esatti 75.0, pochi 13.7, troppi 11.3, RMSE 3.7 cm) |
| Union | 34.9% |
| PHD order average | 31.8% |
| CPHD order average | 22.7% |
| CPHD | 20.5% |
| PHD | 17.6% |
| Zone and class | 9.9% |

### Reti: da sostituire con la run del giorno 9

| Metodo | Scene corrette |
|---|---|
| marker network | 80.3% (vecchio simulatore con lato 30 cm) |
| YOLO + marker network | 83.5 ± 0.7% (anteprima Mac, CV a 2 fold, vedi §5) |

---|---|
| marker network | 80.3% (vecchio simulatore con lato 30 cm) |
| mask network | 63.2% |
| mask + marker network | 73.6% |

- Oracolo marker network + mask + marker: 87.4%.
- Oracolo MHT + marker network: 84.7%.

---

## 9. Stato della tesi

### Già fatto
- Tutto standardizzato su 444 scene, gruppi con nome, scenari con nome, senza ID dei layout.
- Figura 3.1 nuova (triangoli); figura delle misure rigenerata.
- Capitoli 5–8 con i parametri stimati su tutte le scene.
- Capitolo 7 riscritto: tabelle uniche con righe gruppo / scatole / distanza / livello; analisi di come falliscono i metodi su tutte le scene.
- Capitolo 11 ristrutturato (RGB / mask / mask + marker con lo stesso protocollo).
- Il PDF compila: 89 pagine, nessun overfull box.

### Segnaposto
- I numeri che dipendono dal giorno 8 sono marcati `% UPDATE-MONDAY`; nel PDF compaiono come "–%".

---

## 10. Prossimi passi

1. **Commit sul Mac** (Alessio):
   ```
   git add -A bevnet gnn tools demo yolo_gnn .gitignore CONTESTO.md \
           sim/scaling.py sim/geometry.py sim/calibrate.py vf/fusion/sensor_aware_mht.py imgsim/render.py pipeline/calibrate.py \
           docs/commands_day7.txt docs/paste_in_powershell_PC1_day9.txt
   git commit -m "YOLO + marker network as the proposed method; image BEV networks removed; real box size"
   git pull --no-rebase --no-edit && git push
   ```
   `git add -A bevnet` registra lo spostamento di `bevnet/` in `_to_delete/` (ignorato da git).
2. **PC1**: incollare `docs/paste_in_powershell_PC1_day9.txt` in PowerShell. Controllare `run_day9_{0,A,Y,S,H}.log` e i push.
3. **Verifica**: `results/evaluation_all/summary.csv` di PC1 deve dare MHT 73.4% come l'anteprima cloud. In `run_day9_Y.log`, la mAP di validazione di YOLO non deve crollare. In `results/yolo_gnn/fig_yolo_real.jpg`, le maschere devono stare sulle scatole.
4. **Tesi**:
   - Il capitolo 11 (reti BEV) diventa il capitolo della soluzione proposta: YOLO in simulazione, raggi, iniezione, cannot-link, protocollo CV, ablazioni, diagnostica dei raggi, confronto `other-layouts`/`same-run`.
   - Togliere RGB/mask/mask+marker da abstract, capitoli 12–13 e conclusioni.
   - Tabelle reali da `results/yolo_gnn/real_tables.txt`.
5. **Capitolo 8**: tabella di fedeltà da `results/fidelity_all`.
6. **Capitolo 10**: numeri da `results/plan_v2` (MHT e marker network).
7. **Capitolo 12**: tabelle per livello in simulazione (tool da scrivere).
8. **Chiusura**: togliere i marcatori `% UPDATE-MONDAY` / `% UPDATE-MASK`, ricompilare, consegnare in `tesi_latex` e in `overleaf_tesi`.
9. **Demo live**: i metodi immagine (`ImageNets`) importano ancora `bevnet` e non funzionano più. Marker e calibrazione usano `yolo_gnn.camera`. Fuori dalla tesi.

## 11. File spostati il 16/11/2026 (`_to_delete/bev_cnn_2026-11-16/`)

- `bevnet/` intero: reti BEV, Grounding DINO + SAM, il vecchio `train_yolo.py` (ora `yolo_gnn/train_seg.py`).
- `gnn/`: `test_388.py`, `test_hybrid_v2…v5.py`, `test_tri_fix.py`, `diag_missed.py`, `eval_stratified*.py`, `eval_yolo_gnn.py` (esplorativi, tarati sul test).
- `tools/`: `prompt_check.py`, `fig_image_report.py`, `fig_image_maps.py`.
- `docs/`: `commands_day8.txt`, `paste_in_powershell_PC1_day8.txt`.
- Restano nel repo, non tracciati: `data/bev_real_yolo_*`, `results/bev_mask_*/…_real_yolo*.csv`, `TEST_REALE_YOLO.jpg`, `yolov8n-seg.pt`, `results/yolo_seg/` (modello a 1 epoca). Da spostare quando vuoi.
