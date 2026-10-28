"""Registratore di episodi.

Perche' esiste adesso e non a ottobre
-------------------------------------
Ogni sessione col robot fatta senza registratore e' una sessione persa:
i dati non si recuperano dopo. Meglio registrare episodi imperfetti da
subito che episodi perfetti troppo tardi.

Cosa registra
-------------
Tutto quello che e' disponibile, grezzo, senza decidere adesso cosa
servira'. In particolare NON decide come sara' fatta l'azione per il
modello: salva sia gli angoli misurati sia gli impulsi comandati, e la
scelta si fa al momento dell'addestramento. Rifare quella scelta costa
una riscrittura di poche righe; rifare la raccolta dati costa
settimane.

Formato
-------
Una cartella per sessione, una sottocartella per episodio:

    dataset/
        2026-09-14_1030/
            info.json            schema, frequenza, versione SDK
            episode_000/
                meta.json        esito, durata, numero di campioni
                frames.jsonl     un oggetto JSON per campione
            episode_001/
                ...

JSONL invece di Parquet per una ragione pratica: si puo' scrivere
riga per riga mentre l'episodio e' in corso, quindi un crash o una
batteria scarica non portano via l'episodio intero. La conversione al
formato dataset di LeRobot e' un passaggio separato, offline, che si
fa quando serve.

Campi di ogni frame
-------------------
    t           istante assoluto (epoch)
    dt          secondi dall'inizio dell'episodio
    i           indice progressivo del campione
    s1, s2      angoli misurati dei giunti, in gradi
    arm_age     eta' della lettura in secondi (qualita' del dato)
    vx, vy, vyaw    comando al chassis
    pulse_shoulder, pulse_elbow    impulso eseguito: -1, 0, +1
    gripper     ultimo comando alla pinza: 0 aperta, 1 chiusa

I campi che non sono ancora disponibili (IR, IMU, immagini) non
vengono inventati: compariranno quando le sottoscrizioni saranno
attive, e gli episodi vecchi resteranno leggibili perche' ogni frame
e' un oggetto JSON indipendente.
"""

import json
import logging
import os
import time

try:
    import cv2
except ImportError:      # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3

# Qualita' JPEG. 85 e' il punto in cui l'artefatto non e' visibile e il
# file pesa circa un terzo rispetto a 100.
JPEG_QUALITY = 85

# Elenco delle fasi, replicato in info.json perche' il dataset deve
# essere leggibile senza il codice che l'ha prodotto.
try:
    from .teleop import FASI as FASI_NOTE
except Exception:      # pragma: no cover
    FASI_NOTE = []


class Recorder:
    """Scrive episodi su disco. Si aggancia a Teleop.on_tick."""

    def __init__(self, root="dataset", session=None, hz=None,
                 note=None, save_images=True):
        self.root = root
        self.save_images = save_images and cv2 is not None
        if save_images and cv2 is None:
            log.warning("OpenCV non disponibile: le immagini non "
                        "verranno salvate")
        self.session = session or time.strftime("%Y-%m-%d_%H%M")
        self.session_dir = os.path.join(root, self.session)
        self.hz = hz
        self.note = note

        self._episode = None          # indice dell'episodio corrente
        self._file = None
        self._t0 = None
        self._n = 0
        self._dropped = 0
        self._no_image = 0
        self._per_fase = {}
        self._episodes_done = []

        os.makedirs(self.session_dir, exist_ok=True)
        self._write_info()

    # ------------------------------------------------------------------
    def _write_info(self):
        info = {
            "schema_version": SCHEMA_VERSION,
            "session": self.session,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hz": self.hz,
            "note": self.note,
            "fields": [
                "t", "dt", "i",
                "s1", "s2", "arm_age",
                "vx", "vy", "vyaw",
                "pulse_shoulder", "pulse_elbow",
                "gripper",
                "image", "image_age",
                "fase", "istruzione",
            ],
            "images": self.save_images,
            "units": {
                "s1": "gradi", "s2": "gradi",
                "vx": "m/s", "vy": "m/s", "vyaw": "rad/s",
                "pulse_shoulder": "-1|0|+1",
                "pulse_elbow": "-1|0|+1",
                "gripper": "0 aperta, 1 chiusa",
                "image": "percorso relativo all'episodio, o null",
                "image_age": "secondi fra acquisizione e campione",
                "fase": "fase del compito in corso",
                "istruzione": "istruzione in linguaggio naturale, "
                              "da dare al modello come ingresso testuale",
            },
            "fasi": [
                {"nome": n, "istruzione": i}
                for n, i in FASI_NOTE
            ],
        }
        with open(os.path.join(self.session_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    @property
    def is_recording(self):
        return self._file is not None

    @property
    def episode_count(self):
        return len(self._episodes_done)

    @property
    def sample_count(self):
        return self._n

    def start_episode(self):
        if self.is_recording:
            log.warning("episodio gia' in corso")
            return self._episode

        idx = 0
        while os.path.exists(self._episode_dir(idx)):
            idx += 1
        self._episode = idx
        d = self._episode_dir(idx)
        os.makedirs(d, exist_ok=True)

        if self.save_images:
            os.makedirs(os.path.join(d, "images"), exist_ok=True)

        self._file = open(os.path.join(d, "frames.jsonl"), "w")
        self._t0 = time.time()
        self._n = 0
        self._dropped = 0
        self._no_image = 0
        self._per_fase = {}
        log.info("episodio %03d avviato", idx)
        return idx

    def _episode_dir(self, idx):
        return os.path.join(self.session_dir, f"episode_{idx:03d}")

    def stop_episode(self, esito="non_marcato"):
        """Chiude l'episodio corrente e scrive i metadati."""
        if not self.is_recording:
            return None

        durata = time.time() - self._t0
        self._file.close()
        self._file = None

        meta = {
            "episode": self._episode,
            "esito": esito,
            "samples": self._n,
            "dropped": self._dropped,
            "senza_immagine": self._no_image,
            "campioni_per_fase": dict(self._per_fase),
            "duration_s": round(durata, 2),
            "hz_effettivi": round(self._n / durata, 2) if durata else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(self._t0)),
        }
        with open(os.path.join(self._episode_dir(self._episode),
                               "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self._episodes_done.append(meta)
        log.info("episodio %03d chiuso: %s, %d campioni in %.1fs (%.1f Hz)",
                 self._episode, esito, self._n, durata,
                 meta["hz_effettivi"] or 0)
        idx = self._episode
        self._episode = None
        return idx

    # ------------------------------------------------------------------
    def on_tick(self, state, action, flags):
        """Gancio per Teleop. Registra un campione se l'episodio e' aperto.

        Gestisce anche il ciclo di vita: il tasto R apre e chiude
        l'episodio, i tasti 1 e 2 lo chiudono marcandone l'esito.
        """
        # --- ciclo di vita ---
        # La marcatura va gestita PRIMA del flag di registrazione.
        # Premendo 1 o 2 la teleoperazione spegne anche `recording`,
        # quindi con l'ordine inverso l'episodio verrebbe chiuso come
        # "non_marcato" un istante prima che il marcatore venga letto.
        mark = flags.get("mark")
        if mark and self.is_recording:
            self.stop_episode(esito=mark)
            return

        vuole_registrare = flags.get("recording", False)
        if vuole_registrare and not self.is_recording:
            self.start_episode()
        elif not vuole_registrare and self.is_recording:
            self.stop_episode()
            return

        if not self.is_recording:
            return

        fase = flags.get("fase")
        if fase:
            self._per_fase[fase] = self._per_fase.get(fase, 0) + 1

        # --- campione ---
        arm = state.get("arm")
        if arm is None:
            # Un frame senza posa non e' utilizzabile per addestrare.
            # Si conta invece di scriverlo: il conteggio finisce nei
            # metadati ed e' una misura di qualita' dell'episodio.
            self._dropped += 1
            return

        t = state.get("t", time.time())

        # --- immagine ---
        rel = None
        image_age = None
        img = state.get("image")
        if self.save_images:
            if img is None:
                # Un campione senza immagine resta comunque utile per
                # analizzare la traiettoria, ma non per addestrare un
                # VLA. Si conta, e il conteggio finisce nei metadati.
                self._no_image += 1
            else:
                image_age = state.get("image_age")
                nome = f"{self._n:06d}.jpg"
                rel = f"images/{nome}"
                percorso = os.path.join(
                    self._episode_dir(self._episode), "images", nome)
                try:
                    cv2.imwrite(percorso, img,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                except Exception as e:
                    log.warning("scrittura immagine fallita: %s", e)
                    rel = None
                    self._no_image += 1

        frame = {
            "t": round(t, 4),
            "dt": round(t - self._t0, 4),
            "i": self._n,
            "s1": round(arm[0], 2),
            "s2": round(arm[1], 2),
            "arm_age": round(state.get("arm_age") or 0.0, 3),
            "vx": round(action.get("vx", 0.0), 4),
            "vy": round(action.get("vy", 0.0), 4),
            "vyaw": round(action.get("vyaw", 0.0), 4),
            "pulse_shoulder": action.get("pulse_shoulder", 0),
            "pulse_elbow": action.get("pulse_elbow", 0),
            "gripper": action.get("gripper", 0.0),
            "image": rel,
            "image_age": round(image_age, 3) if image_age is not None
                         else None,
            "fase": flags.get("fase"),
            "istruzione": flags.get("istruzione"),
        }

        self._file.write(json.dumps(frame) + "\n")
        # Scrittura immediata: un episodio interrotto male resta
        # comunque leggibile fino all'ultimo campione.
        self._file.flush()
        self._n += 1

    # ------------------------------------------------------------------
    def close(self):
        if self.is_recording:
            self.stop_episode(esito="interrotto")

    def summary(self):
        if not self._episodes_done:
            return "Nessun episodio registrato."
        righe = [f"Sessione {self.session}: "
                 f"{len(self._episodes_done)} episodi"]
        per_esito = {}
        for m in self._episodes_done:
            per_esito[m["esito"]] = per_esito.get(m["esito"], 0) + 1
        for esito, n in sorted(per_esito.items()):
            righe.append(f"  {esito}: {n}")
        tot = sum(m["samples"] for m in self._episodes_done)
        persi = sum(m["dropped"] for m in self._episodes_done)
        senza_img = sum(m.get("senza_immagine", 0)
                        for m in self._episodes_done)
        righe.append(f"  campioni totali: {tot}")
        if self.save_images and tot:
            quota_img = 100.0 * senza_img / tot
            righe.append(f"  campioni senza immagine: {senza_img} "
                         f"({quota_img:.1f}%)")
            if quota_img > 5:
                righe.append("  ATTENZIONE: molti campioni senza "
                             "immagine, controlla lo stream")
        if persi:
            quota = 100.0 * persi / (tot + persi)
            righe.append(f"  campioni scartati: {persi} ({quota:.1f}%)")
            if quota > 10:
                righe.append("  ATTENZIONE: molti campioni senza posa, "
                             "il canale sta soffrendo")
        righe.append(f"  cartella: {self.session_dir}")
        return "\n".join(righe)


# ---------------------------------------------------------------------------
def read_episode(path):
    """Rilegge un episodio. Salta le righe corrotte invece di fallire."""
    frames = []
    with open(os.path.join(path, "frames.jsonl")) as f:
        for n, riga in enumerate(f, 1):
            riga = riga.strip()
            if not riga:
                continue
            try:
                frames.append(json.loads(riga))
            except json.JSONDecodeError:
                log.warning("%s riga %d corrotta, saltata", path, n)
    meta = {}
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return frames, meta