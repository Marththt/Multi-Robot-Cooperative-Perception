# Stato dei lavori: 16 novembre 2026

Tesi magistrale in Telecomunicazioni (UnivAQ). Relatore: prof. Pangun Park.
Quattro robot DJI RoboMaster con telecamera osservano scene statiche di scatole triangolari, tutte con lo stesso marker ArUco. I robot devono stimare quante scatole ci sono e dove sono. Il dataset reale ha 444 scene registrate in laboratorio, con ground truth OptiTrack.

---

## 1. Decisioni prese

- **La rete CNN/BEV è stata rimossa.** Il codice è archiviato in `_to_delete/bev_cnn_2026-11-16/bevnet/`.
- **La soluzione proposta è YOLO + GNN (Model C).** È modulare:
  1. il rilevatore ArUco e YOLO-seg producono raggi di direzione (bearing rays);
  2. la GNN basata sui report li fonde.
- **MHT è il riferimento classico.** Va confrontato con YOLO + GNN in modo onesto, senza gonfiare i risultati.
- **Validazione in simulazione con le stesse metriche**, fino a 50 robot e a un numero crescente di scatole in posizioni casuali.
- **Nuova rete end-to-end (Model B)**, addestrata solo in simulazione: prende le immagini e conta e localizza le scatole senza altre informazioni. Serve a mostrare il problema sim-to-real. Nel mondo reale la nostra soluzione resta il Model C.
- **La tesi è stata riordinata** secondo lo schema concordato con il professore (foto `~/Desktop/overleaf_tesi/IMG_7530.jpeg`).

---

## 2. Cosa è stato fatto

### 2.1 Tesi (`~/Desktop/overleaf_tesi/`)
- `main.tex` include sei capitoli:
  - `ch1_introduction`
  - `ch2_classical`
  - `ch3_gnn`
  - `ch4_environment`
  - `ch5_models`
  - `ch6_conclusions`
- I vecchi capitoli sono in `_to_delete/chapters_2026-11-16/`.
- I risultati mancanti sono segnati con `\tocomplete{}` e `% CHECK-DAY9`.
- La sezione del Model B (`sec:modelB`) è ancora un segnaposto. La riempiremo con i risultati di `results/e2e/`.
- La tesi non è ancora stata compilata: sul Mac non c'è LaTeX, va compilata su Overleaf.

### 2.2 Giorno 9 sul PC1 (in corso)
Gli script `docs/paste_in_powershell_PC1_day9.txt` e `docs/paste_in_powershell_PC1_day9_resume.txt` eseguono questi blocchi:

| Blocco | Cosa fa | Output | Commit/push |
|---|---|---|---|
| **0** | Calibrazione, modello del sensore, metodi classici su tutte le scene reali | `config/sensor_model*.json`, `results/evaluation_all/` | sì |
| **F** | Metodi classici sui replay simulati e fedeltà del simulatore | `results/evaluation_sim_replay_all/`, `results/fidelity_all/` | **no**: il commit lo fa il paste del giorno 10 |
| **A** | Piano v2: MHT (M2/M3), mappa dei fallimenti, GNN su marker (M4), oracolo, GNN su scene reali, scalabilità | `results/plan_v2/M2..M4`, `results/scale/gnn/` | sì |
| **Y** | Addestramento di YOLO-seg in simulazione, segmentazione dei frame reali | `results/yolo_seg/`, `data/yolo_real/cache` | sì |
| **S** | MHT, rete su marker e YOLO + rete su marker in simulazione, fino a 50 robot e 50 scatole | `results/scale/` | sì |
| **H** | YOLO + GNN su tutte le scene reali (calibrazione `other-layouts` e `same-run`), tabelle reali | `results/yolo_gnn/`, `results/yolo_gnn_same_run/` | sì |

**Problema risolto oggi.** Il blocco 0 sembrava bloccato da ore su `evaluate_runs ... sim_replay_all ... --quiet`. In realtà stava elaborando circa 10.000 scene simulate su un solo core senza stampare niente. Il resume ha:
- spostato quel passo nel blocco **F**, ora senza `--quiet`;
- avviato **A**, **Y** e **F** in parallelo;
- messo **S** e **H** in coda ad A e Y.

L'avviso `RuntimeWarning: invalid value encountered in cast` di `imgsim/render.py` è innocuo: produce qualche pixel nero sullo sfondo e le maschere non vengono toccate.

### 2.3 Nuovo pacchetto `e2e/` (Model B)
È già scritto e provato sul Mac, ma non è ancora committato.

| File | Contenuto |
|---|---|
| `e2e/geometry.py` | Griglia del pavimento ±2 m con celle da 5 cm (80×80), altezze di campionamento 5/15/25 cm, proiezione nella camera, target heatmap, decodifica dei picchi |
| `e2e/data.py` | Rendering delle scene `sim.plan` con gli stessi seed del piano v2 e lo stesso aspetto usato per addestrare YOLO: pitch 8–32°, ostacoli, distrattori. Salva le immagini in 256×144 in `data/e2e/<set>/shard_*.pkl` |
| `e2e/model.py` | `E2ENet`, circa 615k parametri. Ha quattro stadi (dettagli sotto) |
| `e2e/real.py` | Converte le scene reali (`data/survey_all`) nello stesso formato. Le pose vengono da OptiTrack, pitch e yaw sono calibrati sugli **altri** layout, si applica gamma 1.8 come per YOLO. `--check` disegna le scatole vere sui frame |
| `e2e/train.py` | AdamW + OneCycle, AMP, 20 epoche, patience 4. La soglia si sceglie sulla validazione. Test sui set G0/E3/S1 e sulle scene reali, con un confronto diretto con MHT sulle stesse scene |
| `docs/paste_in_powershell_PC1_day10_e2e.txt` | Script per il PC1: commit del blocco F, generazione dei dati, addestramento, test, commit e push |

I quattro stadi di `E2ENet`:
1. un encoder CNN per le immagini;
2. il sollevamento delle feature sulle celle del pavimento (`grid_sample`);
3. una **GNN di attenzione fra i robot** che vedono la stessa cella, con un bias dato dalla geometria della coppia (angolo fra le linee di vista e distanze);
4. un decoder U-Net che produce la heatmap dei centri e gli offset.

La rete non dipende dall'ordine dei robot: verificato, differenza massima 2e-7.

**Prove sul Mac.** Sono solo controlli che il codice funziona, **non** risultati per la tesi:
- il rendering richiede circa 0,2 s per scena e per core, e circa 70 KB per scena;
- overfit su 250 scene G0: 100% sulle scene di addestramento;
- 50 scene simulate mai viste: 90% corrette;
- 56 scene reali: **32,1%** corrette, contro **91,1%** di MHT sulle stesse scene. Questo è il gap sim-to-real che vogliamo mostrare.

**Limite da dichiarare in tesi.** La griglia è di ±2 m, quindi il Model B non può girare sui set di scalabilità (fino a 50 robot).

---

## 3. Domani: test da fare

### 3.1 Controllare la fine del giorno 9 sul PC1
1. Nella finestra principale deve comparire `ALL DONE`, e nella finestra "day9 F" deve comparire `BLOCK F DONE`.
2. Controllare che i commit siano arrivati su GitHub:
   ```
   git fetch
   git status
   git log origin/main..HEAD --oneline
   ```
   Se `git log` elenca dei commit, lanciare `git pull --no-rebase --no-edit` e poi `git push`. Gli errori di push sono in `run_day9_*_push.log`.
3. Se `git status` mostra cartelle `results/...` non tracciate, mandarmi l'output.

### 3.2 Giorno 10: Model B sul PC1
1. Sul Mac: commit e push di `e2e/`, `.gitignore` e `docs/paste_in_powershell_PC1_day10_e2e.txt`.
2. Sul PC1: incollare `docs/paste_in_powershell_PC1_day10_e2e.txt` in PowerShell. Lo script:
   - committa il blocco F;
   - genera 30.000 scene di training, 1.000 di validazione e 1.000 per ciascuno dei set G0, E3 e S1;
   - prepara le scene reali e la figura di controllo;
   - addestra la rete e la testa;
   - fa commit e push di `results/e2e/`.
3. Il tempo sul PC1 non è ancora stato misurato. Con 60 core, il rendering di circa 33.000 scene dovrebbe richiedere alcuni minuti. L'addestramento dipende dalla GPU.
4. Cose da controllare:
   - `data/e2e_real/fig_e2e_real_check.jpg`: i punti verdi devono stare sulle scatole;
   - `run_day10_B.log`: l'andamento delle loss e la soglia scelta;
   - `results/e2e/summary.csv`: percentuale di scene corrette, too few / too many / misplaced, RMSE, F1 sui set simulati e su quello reale, accanto a MHT.

### 3.3 Obiettivi dei risultati
- **Simulazione:** il Model B deve funzionare bene su G0/E3/S1. Questo mostra che una rete end-to-end impara il compito se i dati di test assomigliano a quelli di training.
- **Reale:** ci aspettiamo un calo netto rispetto a MHT e a YOLO + GNN. Questo è il problema sim-to-real.
- **YOLO + GNN (Model C):** deve essere migliore o almeno competitivo rispetto a MHT sulle scene reali, per livello di difficoltà (`results/scene_levels.csv`). Deve anche scalare fino a 50 robot (`results/scale/`).
- **Onestà:** riportiamo anche i casi in cui MHT vince. Non gonfiamo niente.

---

## 4. Dopo i risultati: lavoro sulla tesi
Bisogna riempire le tabelle e i segnaposto `\tocomplete{}` e `% CHECK-DAY9` usando questi file:

| Risultato | Fonte |
|---|---|
| Metodi classici su scene reali | `results/evaluation_all/` |
| MHT nel piano v2, mappa dei fallimenti | `results/plan_v2/M3/` |
| GNN su marker e oracolo | `results/plan_v2/M4/` |
| Fedeltà del simulatore | `results/fidelity_all/` |
| YOLO + GNN sul reale | `results/yolo_gnn/summary.csv`, `rays.csv`, `real_tables.txt` |
| Scalabilità (fino a 50 robot) | `results/scale/*` |
| Model B (end-to-end) | `results/e2e/summary.csv` |

Altri controlli rimasti aperti:
- rifare in inglese la figura dei layout di scalabilità;
- controllare se `inject` aggiunge scatole false quando il numero di robot N è grande;
- verificare che il `cannot_link` vettorizzato in `yolo_gnn/hybrid.py` dia gli stessi risultati del vecchio ciclo sui dati reali;
- spostare in `_to_delete/` i file rimasti non tracciati:
  - `data/bev_real_yolo_*`
  - `TEST_REALE_YOLO.jpg`
  - `yolov8n-seg.pt`
  - `results/yolo_seg/` (solo la copia vecchia sul Mac)
  - i vecchi `results/bev_mask_*`
