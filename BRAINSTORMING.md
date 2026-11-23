# BRAINSTORMING — Nuovo modello: immagine intera + misure → MLP → GNN

Riassunto della sessione alla lavagna con il prof (novembre 2026), da usare come contesto per una nuova chat. Il contesto generale del progetto (dati, metodi, file, stato della tesi) è in `CONTESTO.md`, nella cartella principale del repo.

---

## 1. Progetto in breve

- **Problema**: 4 robot (RoboMaster EP, pose da OptiTrack) osservano una scena statica di scatole a prisma triangolare. Bisogna stimare **quante scatole ci sono e dove sono**. Il nodo centrale è l'**associazione**: capire quali osservazioni di robot diversi appartengono alla stessa scatola.
- **Dati reali**: 444 scene da 105 layout.
- **Simulatori disponibili**:
  - a livello di misura (`sim/`);
  - di immagini (`imgsim/render.py`), con la geometria misurata della piattaforma.
- **Inquadramento della tesi**: **(Feature) + (Model)**. Ogni metodo è una feature estratta dai dati del robot più un modello che fonde le feature dei robot.

---

## 2. Lettura della lavagna

| Zona | Significato |
|---|---|
| **Centro (giallo)** | **Gap sim → real**: confronto tra i risultati in simulazione e sul reale, per arrivare a risultati validi sul reale. Il modello si addestra in simulazione e si verifica sulle scene reali. |
| **Sinistra** | **Pipeline attuale**: immagine → rilevamento (ArUco) → misure per oggetto (ρ, θ, …) → GNN → risultato. |
| **Centro-destra** | **Nuova pipeline richiesta dal prof**: Input → MLP → vettore di feature f → GNN. Dimensione della feature: obiettivo 2³ = 8. |
| **In alto a destra** | GNN **detection based** (tenuta) contro **robot based** (barrata). |
| **Colonna destra (viola)** | Scaletta della **presentazione di domani** (vedi §6). |

---

## 3. Richiesta del prof: rilassare l'input

### Input del modello, per ogni robot

1. **L'immagine intera**, senza elaborazioni: solo ridotta o riscalata. Niente YOLO, niente ArUco, niente segmentazione sull'immagine. La rete deve **scoprire da sola** cosa estrarre.
   - In simulazione, per ora, **scala di grigi**.
   - I **colori** serviranno quando si passa al reale.
2. **Le misure degli oggetti**: distanza **ρ** e angolo **θ** di ogni oggetto rilevato.
   - Il modello è **indifferente al sensore**: ρ e θ possono arrivare da ArUco, YOLO, lidar o altro. Conta la misura, non come è stata ottenuta.
   - Nel nostro caso vengono da ArUco perché era la soluzione più semplice e fattibile.
3. **La posa del robot**: **x, y e l'angolo ψ**.

**Nient'altro in ingresso.** Si tolgono tutte le feature fatte a mano di adesso: angolo di vista del marker, correzione di calibrazione distanza/bearing, posizione già calcolata sul pavimento, classe.

### Estrattore di feature

- Una **MLP** (non una CNN) riceve l'immagine intera e restituisce **k feature**.
- **Obiettivo k = 2³ = 8**; vanno bene anche di più. Bisogna però **capire e giustificare quante feature servono e perché**.
- **Niente CNN**, perché il modello deve poter essere applicato a **scenari su grande scala**.

### Fusione

- Una **GNN detection based** (un nodo per oggetto misurato) riceve le misure, la posa e le feature dell'immagine, e risolve il problema: conteggio, posizioni e associazione.
- L'uscita e le metriche restano quelle di adesso: scena corretta se il conteggio è esatto e ogni scatola ha una stima entro 15 cm.

### Confronto

- **MHT riceve lo stesso input ridotto** (ρ, θ, posa), così il confronto è alla pari.
- Confronto **in simulazione** e **sul reale**, per misurare il gap del centro della lavagna.

---

## 4. Decisioni prese

| Punto | Decisione |
|---|---|
| Immagine | Intera, solo ridimensionata; nessun detector o segmentatore applicato all'immagine |
| Estrattore | MLP sull'immagine appiattita; niente CNN |
| Misure | ρ, θ per oggetto, indifferenti al sensore (oggi da ArUco) |
| Posa | x, y, ψ del robot |
| Feature fatte a mano | tutte rimosse dall'input |
| Numero di feature | obiettivo 8, ammesse di più; va giustificato |
| Tipo di GNN | detection based |
| Immagini di addestramento | simulate (`imgsim`), in scala di grigi; colori per il passaggio al reale |
| Confronto | MHT con lo stesso input; simulazione contro reale |
| Reti su maschere / RGB (BEV) | **lavoro a parte, non fanno parte della tesi** |

---

## 5. Punti aperti (da decidere)

### 1. Come entrano le feature dell'immagine nella GNN

- **Proposta A** (la più semplice): il vettore z_i dell'immagine del robot i (k feature) viene concatenato a ogni nodo-rilevamento di quel robot. Il nodo contiene quindi `[ρ, θ, x_i, y_i, ψ_i, z_i]`.
- **Proposta B**: grafo eterogeneo, con un nodo-robot che porta z_i, collegato ai propri nodi-rilevamento, e nodi-rilevamento con `[ρ, θ, x, y, ψ]`.

### 2. Addestramento

- **End-to-end**: MLP e GNN insieme. Così le feature le scopre la rete, come chiede il prof.
- **Oppure in due fasi**: prima la MLP con un obiettivo ausiliario, poi la GNN.
- La mia proposta è end-to-end.

### 3. Quante feature e perché

- **Sweep** di k ∈ {2, 4, 8, 16, 32}: curva scene corrette contro k, si sceglie il ginocchio.
- **"Perché"**: un probe lineare dalle feature a grandezze note della scena (numero di scatole visibili, occlusioni, distanza media, luce) mostra cosa codifica ogni feature.

### 4. Risoluzione dell'immagine in ingresso alla MLP

Per esempio 64×36 o 128×72 in grigi, cioè 2304 o 9216 ingressi; ×3 con i colori.

### 5. Grande scala

Quanti robot, quante scatole e che area: ancora da definire, con eventuali nuovi scenari di test in simulazione.

### 6. Gap sim → real con i pixel grezzi

Con l'immagine intera l'aspetto conta direttamente. Va misurato con lo stesso modello addestrato in simulazione e valutato sulle 444 scene reali.

### 7. Conseguenze sulla tesi

Le reti su maschere/RGB escono dalla tesi, quindi i capitoli 11 e 12 e le conclusioni vanno rivisti: il capitolo 11 è da sostituire con il nuovo modello.

---

## 6. Presentazione di domani (colonna destra della lavagna)

Scaletta:

1. Intro
2. …
3. Algoritmi classici
4. GNN
   - struttura della GNN
   - input
   - dimensione e numero delle feature
5. Risultati, divisi per difficoltà: **easy / medium / hard**
6. **Ongoing**: il lavoro di adesso, cioè questo nuovo modello (immagine intera + misure → MLP → GNN)

Le slide vanno ad alto livello: lavoro, ambiente e strumenti, senza dettagli tecnici (numeri, specifiche hardware, parametri).

---

## 7. File utili nel repo

| File | Ruolo |
|---|---|
| `CONTESTO.md` | contesto completo del progetto |
| `imgsim/render.py` | simulatore di immagini (geometria misurata) |
| `bevnet/data.py` | generazione dei dataset di immagini simulate |
| `gnn/train_sim.py` | marker network attuale (GNN detection based), base per la nuova GNN |
| `sim/` | simulatore a livello di misura (ρ, θ simulati) |
| `pipeline/evaluate_runs.py` | metodi classici, compreso MHT |
| `tools/real_tables.py` | tabelle sulle scene reali (gruppi, livelli, McNemar, oracolo) |
