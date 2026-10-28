"""Braccio in spazio giunti, comandato via servo diretti.

Perche' non robotic_arm
-----------------------
Il modulo `robotic_arm` dell'SDK DJI non risponde su questo robot: i
comandi vengono accettati ma il braccio non si muove, e `sub_position`
riporta sempre (0, 0). I servo invece rispondono, e `get_angle` legge
angoli ASSOLUTI che sopravvivono allo spegnimento. Quindi si lavora a
quel livello.

Il modello, ricavato da 20 comandi misurati
-------------------------------------------
    lettura = comando + 180        (entrambi i servo)

Scala 1:1 a meno del rumore (fit 0.978 e 0.992), offset 179.98 e 179.77.
Cambia solo lo zero, ruotato di mezzo giro.

Convenzione
-----------
Gli angoli pubblici di questo modulo sono quelli di `get_angle`, cioe'
quelli che si leggono davvero. La conversione verso le unita' di
`moveto` e' interna e non compare nel codice chiamante.
"""

import json
import logging
import os
import threading
import time

from .arm_safety import SAFE_REGION

log = logging.getLogger(__name__)

# Offset fra le unita' di moveto() e quelle di get_angle().
SERVO_OFFSET = 180.0

# Indici dei servo: 1 = spalla, 2 = gomito.
SHOULDER = 1
ELBOW = 2

MOVE_TIMEOUT = 6.0
SETTLE = 0.6

# Frequenza del lettore di sfondo degli angoli.
# get_angle e' una richiesta SINCRONA: il chiamante aspetta la risposta
# del robot. Ogni lettore indipendente moltiplica il traffico, e oltre
# una certa soglia il canale wifi va in timeout. Per questo esiste UN
# SOLO lettore, e tutti gli altri leggono la cache.
POLL_HZ = 10

# Isteresi misurata, in gradi. Entita' della sovracorsa in approach_from.
HYSTERESIS = 1.2

# --------------------------------------------------------------------------
# Impulsi in velocita'
# --------------------------------------------------------------------------
# drive_speed(index, speed) muove il giunto finche' non arriva lo stop.
# Misure su due sessioni, con durata comandata 0.05 s:
#     speed  5 -> +5.25 / +3.40 gradi
#     speed 10 -> +11.10
#     speed 20 -> +21.95
# Lo spostamento e' proporzionale a speed, con coefficiente fra 0.68 e
# 1.10 gradi per unita'. Il coefficiente e' stabile dentro una sessione
# e cambia fra sessioni: dipende dalla latenza di rete, che decide
# quanto a lungo il giunto si muove davvero prima dello stop.
#
# Conseguenza di progetto: NON si alza la velocita' per andare piu'
# veloce (e' cosi' che speed 20 e 30 hanno colpito il sensore). Si usa
# una velocita' bassa e si ripetono gli impulsi.
PULSE_SPEED = 5
PULSE_DURATION = 0.05

# Gradi per unita' di speed, caso peggiore osservato piu' margine.
EXCURSION_PER_UNIT = 1.4

# Margine minimo richiesto oltre all'escursione prevista.
PULSE_SAFETY_MARGIN = 4.0

# --------------------------------------------------------------------------
# Posa di riposo
# --------------------------------------------------------------------------
# E' la posa in cui il braccio sta durante navigazione e avvicinamento.
# Conta piu' di quanto sembri: la camera e' montata sul braccio, quindi
# questa posa decide COSA INQUADRA il robot mentre cerca l'oggetto. Con
# il braccio troppo alto l'immagine e' tutta parete e soffitto, e il
# modello non puo' vedere un oggetto che non e' nel fotogramma.
#
# Va scelta guardando il video, non a intuito, e poi tenuta identica per
# tutta la raccolta dati: se cambia fra un episodio e l'altro, cambia il
# punto di vista e il modello deve imparare anche quella variabilita'.
REST_POSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "rest_pose.json")

# --------------------------------------------------------------------------
# Bersaglio del tasto B
# --------------------------------------------------------------------------
# Una seconda posa salvabile, diversa dalla posa di riposo: serve per
# tornare senza jog manuale a un punto di partenza scelto (tipicamente
# l'inizio della fase "afferra"), quando si registrano molti episodi
# di seguito con --fase e ci si ritrova a rifare IJKL identici ogni
# volta. Stesso meccanismo di rientro della posa di riposo (impulsi
# veri, registrati), bersaglio diverso.
#
# Se non e' stata ancora salvata nessuna posizione, si parte da questo
# valore: la posa (262.7, 85.7) indicata come punto di partenza per
# "afferra". E' sovrascrivibile in ogni momento con V.
DEFAULT_WAYPOINT = (262.7, 85.7)

WAYPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "waypoint.json")

# --------------------------------------------------------------------------
# Posa di riposo
# --------------------------------------------------------------------------
# Posa in cui il braccio sta durante ricerca, avvicinamento e trasporto.
# Non e' una scelta geometrica ma FOTOGRAFICA: la camera e' montata sul
# braccio, quindi questa posa decide cosa inquadra. Va scelta guardando
# l'immagine, non gli angoli.
#
# Se il file non esiste si usa il centro della regione sicura, che e'
# sicuro ma quasi certamente inquadra male.
HOME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "arm_home.json")


class ArmError(RuntimeError):
    """Errore di comando del braccio."""


class UnsafePose(ArmError):
    """La posa richiesta cade fuori dalla regione sicura."""


class Arm:
    """Braccio a 2 giunti, comandato in angoli di giunto.

    Tutti gli angoli sono in gradi, nella convenzione di get_angle().
    """

    def __init__(self, ep_servo, region=None, clamp_by_default=True):
        self._servo = ep_servo
        self.region = region or SAFE_REGION
        self.clamp_by_default = clamp_by_default
        self._lock = threading.Lock()

        # Cache degli angoli, alimentata dal lettore di sfondo.
        self._cache = None
        self._cache_t = None
        self._read_errors = 0
        self._poller = None
        self._poll_stop = threading.Event()

        # Azione in corso: l'SDK ne accetta UNA sola alla volta.
        self._pending = None
        self._pending_t = None

    # ------------------------------------------------------------------
    # Lettura
    # ------------------------------------------------------------------
    def start_polling(self, hz=POLL_HZ):
        """Avvia il lettore di sfondo degli angoli.

        Da chiamare una volta sola. Dopo, angles() ritorna la cache
        senza generare traffico: e' quello che permette a watchdog,
        teleoperazione e registratore di coesistere senza saturare il
        canale.
        """
        if self._poller is not None:
            return self
        self._poll_stop.clear()
        self._poller = threading.Thread(target=self._poll_loop,
                                        args=(1.0 / hz,), daemon=True)
        self._poller.start()
        # Attendi il primo campione, cosi' angles() non ritorna None.
        deadline = time.time() + 3.0
        while self._cache is None and time.time() < deadline:
            time.sleep(0.02)
        if self._cache is None:
            raise ArmError("il lettore degli angoli non riceve dati")
        return self

    def stop_polling(self):
        self._poll_stop.set()
        if self._poller:
            self._poller.join(timeout=1.0)
        self._poller = None

    def _poll_loop(self, period):
        while not self._poll_stop.is_set():
            try:
                a = self._read_direct()
                self._cache = a
                self._cache_t = time.time()
                self._read_errors = 0
            except Exception:
                self._read_errors += 1
                if self._read_errors in (5, 20):
                    log.warning("lettura angoli fallita %d volte di fila",
                                self._read_errors)
            time.sleep(period)

    def _read_direct(self):
        """Lettura sincrona diretta. Usare solo dal poller."""
        with self._lock:
            s1 = float(self._servo.get_angle(index=SHOULDER))
            s2 = float(self._servo.get_angle(index=ELBOW))
        # Dopo un timeout l'SDK restituisce 0.0: non e' una posa reale, e
        # prenderla per buona ha gia' fatto scattare il watchdog a vuoto.
        if s1 == 0.0 and s2 == 0.0:
            raise ArmError("lettura (0.0, 0.0): valore di timeout")
        return (s1, s2)

    def angles(self, samples=None, max_age=1.0):
        """Angoli correnti (spalla, gomito).

        Se il lettore di sfondo e' attivo ritorna la cache, che non
        costa traffico. Altrimenti fa una lettura sincrona.

        samples resta accettato per compatibilita' ma viene ignorato
        quando la cache e' attiva.
        """
        if self._poller is not None:
            if self._cache is None:
                raise ArmError("nessun campione disponibile")
            eta = time.time() - self._cache_t
            if eta > max_age:
                log.warning("angoli vecchi di %.1f s", eta)
            return self._cache
        return self._read_direct()

    @property
    def reading_age(self):
        """Eta' dell'ultima lettura, in secondi."""
        if self._cache_t is None:
            return float("inf")
        return time.time() - self._cache_t

    def is_safe(self, s1=None, s2=None):
        if s1 is None or s2 is None:
            s1, s2 = self.angles()
        return self.region.is_safe(s1, s2)

    def margin(self, s1=None, s2=None):
        """Gradi dal bordo della regione sicura. Negativo se fuori."""
        if s1 is None or s2 is None:
            s1, s2 = self.angles()
        return self.region.margin_to_boundary(s1, s2)

    # ------------------------------------------------------------------
    # Conversione
    # ------------------------------------------------------------------
    @staticmethod
    def _to_command(angle):
        return int(round(angle - SERVO_OFFSET))

    @staticmethod
    def _to_angle(command):
        return command + SERVO_OFFSET

    # ------------------------------------------------------------------
    # Comando
    # ------------------------------------------------------------------
    def _action_busy(self):
        """C'e' un'azione ancora in corso?

        L'SDK RoboMaster accetta UNA sola azione alla volta: inviarne
        una seconda mentre la prima e' attiva produce
        "Robot is already performing 1 action(s)" e il comando viene
        perso in silenzio.
        """
        a = self._pending
        if a is None:
            return False
        try:
            if getattr(a, "is_completed", False):
                self._pending = None
                return False
        except Exception:
            pass
        stato = str(getattr(a, "state", ""))
        if stato in ("action_succeeded", "action_failed", "action_exception"):
            self._pending = None
            return False
        # Non si resta appesi per sempre su un'azione che non si chiude.
        if self._pending_t and time.time() - self._pending_t > MOVE_TIMEOUT:
            log.warning("azione scaduta dopo %.1f s, la considero finita",
                        MOVE_TIMEOUT)
            self._pending = None
            return False
        return True

    def _move_joint_raw(self, index, angle, wait=True, skip_if_busy=False):
        """Muove un giunto senza controlli di sicurezza. Uso interno.

        skip_if_busy: se True e c'e' gia' un'azione in corso, il comando
        viene SCARTATO invece di essere inviato. Serve alla
        teleoperazione, dove conta l'ultimo bersaglio e non la coda:
        meglio perdere un incremento che intasare il canale.
        """
        if skip_if_busy and self._action_busy():
            return "skipped"

        cmd = self._to_command(angle)
        try:
            with self._lock:
                action = self._servo.moveto(index=index, angle=cmd)
            self._pending = action
            self._pending_t = time.time()
            if wait:
                action.wait_for_completed(timeout=MOVE_TIMEOUT)
                self._pending = None
            return getattr(action, "state", "n/d")
        except Exception as e:
            self._pending = None
            raise ArmError(f"moveto(index={index}, angle={cmd}) fallita: {e}")

    def move_to(self, s1, s2, wait=True, clamp=None, order=None):
        """Porta il braccio agli angoli (s1, s2).

        clamp: se True (predefinito) una posa fuori regione viene
        riportata dentro invece di sollevare eccezione.

        order: sequenza di indici che stabilisce quale giunto muovere
        prima. Se None, la sceglie _safe_order: passare per una posa
        intermedia non sicura e' il modo piu' comune di sbattere anche
        quando partenza e arrivo sono entrambi validi.

        Ritorna (s1_effettivo, s2_effettivo, saturato).
        """
        clamp = self.clamp_by_default if clamp is None else clamp

        if not self.region.is_safe(s1, s2):
            if not clamp:
                raise UnsafePose(
                    f"posa ({s1:.1f}, {s2:.1f}) fuori dalla regione sicura"
                )
            s1, s2, _ = self.region.clamp(s1, s2)
            log.warning("posa saturata a (%.1f, %.1f)", s1, s2)
            saturato = True
        else:
            saturato = False

        if order is None:
            order = self._safe_order(s1, s2)

        for index in order:
            target = s1 if index == SHOULDER else s2
            # Anche con wait=False bisogna attendere che l'azione
            # precedente si chiuda, o il secondo giunto non parte.
            if not wait:
                deadline = time.time() + MOVE_TIMEOUT
                while self._action_busy() and time.time() < deadline:
                    time.sleep(0.02)
            self._move_joint_raw(index, target, wait=wait)
            if wait:
                time.sleep(SETTLE)

        got1, got2 = self.angles()
        return got1, got2, saturato

    def _safe_order(self, s1, s2):
        """Decide quale giunto muovere per primo.

        Si valutano entrambe le pose intermedie e si sceglie quella che
        resta dentro la regione. Se vanno bene entrambe, si muove prima
        il giunto con lo spostamento maggiore.
        """
        try:
            cur1, cur2 = self.angles()
        except ArmError:
            return (SHOULDER, ELBOW)

        prima_spalla = self.region.is_safe(s1, cur2)
        prima_gomito = self.region.is_safe(cur1, s2)

        if prima_spalla and not prima_gomito:
            return (SHOULDER, ELBOW)
        if prima_gomito and not prima_spalla:
            return (ELBOW, SHOULDER)
        if not prima_spalla and not prima_gomito:
            log.warning(
                "entrambe le pose intermedie escono dalla regione: il "
                "percorso passa per una zona non verificata"
            )
        if abs(s1 - cur1) >= abs(s2 - cur2):
            return (SHOULDER, ELBOW)
        return (ELBOW, SHOULDER)

    def move_joint(self, index, angle, wait=True, clamp=None):
        """Muove un solo giunto, lasciando l'altro dov'e'."""
        cur1, cur2 = self.angles()
        if index == SHOULDER:
            return self.move_to(angle, cur2, wait=wait, clamp=clamp,
                                order=(SHOULDER,))
        return self.move_to(cur1, angle, wait=wait, clamp=clamp,
                            order=(ELBOW,))

    def approach_from(self, s1, s2, direction=-1, overshoot=None):
        """Posizionamento fine con approccio da direzione fissa.

        I giunti hanno ~1 grado di isteresi: lo stesso comando atterra
        piu' in alto se ci si arriva dal basso. Approcciando sempre
        dalla stessa direzione l'errore diventa sistematico invece che
        casuale, quindi ripetibile.

        direction: -1 approccia dal basso, +1 dall'alto.
        """
        overshoot = HYSTERESIS * 2 if overshoot is None else overshoot
        pre1, pre2, _ = self.region.clamp(s1 + direction * overshoot,
                                          s2 + direction * overshoot)
        self.move_to(pre1, pre2)
        return self.move_to(s1, s2)

    # ------------------------------------------------------------------
    # Impulsi in velocita' (teleoperazione)
    # ------------------------------------------------------------------
    def _valuta_pulse(self, index, direction, speed, safety_margin):
        """Verifica preventiva di un impulso, senza eseguirlo.

        Fattorizzata fuori da pulse() perche' serve anche a chi deve
        SCEGLIERE fra piu' mosse possibili prima di provarne una (il
        rientro automatico: se il giunto preferito e' bloccato vicino
        al bordo, meglio saperlo subito e provare l'altro giunto,
        invece di richiedere sempre lo stesso impulso e vederselo
        rifiutare ogni ciclo senza mai muoversi).

        Ritorna (ammesso, margine_previsto, margine_attuale, previsto).
        """
        try:
            s1, s2 = self.angles()
        except ArmError:
            return False, None, None, None

        soglia = PULSE_SAFETY_MARGIN if safety_margin is None else safety_margin

        escursione = abs(speed) * EXCURSION_PER_UNIT * direction
        if index == SHOULDER:
            previsto = (s1 + escursione, s2)
        else:
            previsto = (s1, s2 + escursione)

        margine_previsto = self.region.margin_to_boundary(*previsto)
        margine_attuale = self.region.margin_to_boundary(s1, s2)

        # Due condizioni alternative.
        #
        # 1. La posa risultante e' sicura con margine pieno: caso normale.
        #
        # 2. La posa risultante e' ancora sotto soglia, MA migliora il
        #    margine: rientro. Senza questa seconda condizione, da una
        #    posa vicina al bordo verrebbe rifiutato ogni movimento,
        #    compresi quelli che riportano dentro, e il braccio
        #    resterebbe bloccato dove si trova.
        ok_pieno = margine_previsto >= soglia
        ok_rientro = margine_previsto > margine_attuale + 0.5

        return (ok_pieno or ok_rientro), margine_previsto, margine_attuale, previsto

    def can_pulse(self, index, direction, speed=PULSE_SPEED,
                  safety_margin=None):
        """Come pulse(), ma solo la verifica: non muove niente.

        Serve a chi vuole sapere PRIMA se un impulso verrebbe accettato,
        per scegliere fra piu' mosse possibili (vedi _step_auto_return
        in teleop.py). Ritorna (ammesso, margine_previsto).
        """
        if direction not in (1, -1):
            raise ValueError("direction deve essere +1 o -1")
        ammesso, margine_previsto, _, _ = self._valuta_pulse(
            index, direction, speed, safety_margin)
        return ammesso, margine_previsto

    def pulse(self, index, direction, speed=PULSE_SPEED,
              duration=PULSE_DURATION, safety_margin=None):
        """Un impulso breve su un giunto, con verifica preventiva.

        Il principio: non si sorveglia il movimento, lo si LIMITA. Su
        questo canale get_angle consegna 1-4 letture al secondo, quindi
        una sicurezza che legga durante il moto arriva sempre tardi.
        Qui invece si calcola PRIMA quanto l'impulso puo' spostare il
        giunto nel caso peggiore, si verifica che la posa risultante
        resti dentro la regione, e poi si comanda alla cieca.

        direction: +1 o -1.

        safety_margin: soglia di margine richiesta per il "caso
        normale" (vedi _valuta_pulse). Se None usa PULSE_SAFETY_MARGIN,
        pensata per il jog manuale libero, dove non si sa in anticipo
        dove si sta andando e la soglia deve restare prudente. Chi
        invece sta avvicinandosi a un punto GIA' VERIFICATO come sicuro
        (es. il rientro automatico verso la posa di riposo) puo'
        passare una soglia piu' bassa: non sta esplorando alla cieca,
        sta tornando verso un bersaglio noto.

        Ritorna (eseguito, motivo).
        """
        if direction not in (1, -1):
            raise ValueError("direction deve essere +1 o -1")

        ammesso, margine_previsto, margine_attuale, previsto = \
            self._valuta_pulse(index, direction, speed, safety_margin)

        if margine_previsto is None:
            return False, "posa non leggibile"

        if not ammesso:
            return False, (
                f"({previsto[0]:.1f}, {previsto[1]:.1f}) margine "
                f"{margine_previsto:.1f}, non migliora "
                f"{margine_attuale:.1f}"
            )

        try:
            with self._lock:
                self._servo.drive_speed(index=index,
                                        speed=int(speed * direction))
        except Exception as e:
            self._stop_speed(index)
            return False, f"drive_speed fallita: {e}"

        time.sleep(duration)
        self._stop_speed(index)
        return True, "ok"

    def _stop_speed(self, index, tentativi=3):
        """Arresto del moto in velocita'. Insiste: e' l'unica cosa che
        ferma il giunto, quindi un solo tentativo non basta."""
        for _ in range(tentativi):
            try:
                with self._lock:
                    self._servo.drive_speed(index=index, speed=0)
                return True
            except Exception:
                time.sleep(0.05)
        log.error("ARRESTO DEL GIUNTO %d FALLITO", index)
        return False

    def allowed_directions(self, speed=PULSE_SPEED):
        """Quali dei quattro movimenti sono ammessi dalla posa attuale.

        Ritorna un dizionario {(index, direction): (ammesso, margine)}.
        Serve alla teleoperazione per dire all'utente dove puo' andare
        invece di limitarsi a rifiutare.
        """
        try:
            s1, s2 = self.angles()
        except ArmError:
            return {}

        attuale = self.region.margin_to_boundary(s1, s2)
        esito = {}
        for index in (SHOULDER, ELBOW):
            for direction in (+1, -1):
                d = abs(speed) * EXCURSION_PER_UNIT * direction
                p = (s1 + d, s2) if index == SHOULDER else (s1, s2 + d)
                m = self.region.margin_to_boundary(*p)
                ammesso = (m >= PULSE_SAFETY_MARGIN
                           or m > attuale + 0.5)
                esito[(index, direction)] = (ammesso, m)
        return esito

    # ------------------------------------------------------------------
    # Posa di riposo
    # ------------------------------------------------------------------
    def get_rest_pose(self):
        """Posa di riposo memorizzata, o None se non e' stata scelta."""
        if not os.path.exists(REST_POSE_PATH):
            return None
        try:
            with open(REST_POSE_PATH) as f:
                d = json.load(f)
            return (float(d["s1"]), float(d["s2"]))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            log.warning("posa di riposo non leggibile: %s", e)
            return None

    def set_rest_pose(self, s1=None, s2=None):
        """Memorizza la posa attuale (o quella indicata) come riposo.

        Viene scritta su disco, quindi vale anche nelle sessioni
        successive: la posa di riposo deve restare identica per tutta la
        raccolta dati.
        """
        if s1 is None or s2 is None:
            s1, s2 = self.angles()

        if not self.region.is_safe(s1, s2):
            raise UnsafePose(
                f"({s1:.1f}, {s2:.1f}) non e' dentro la regione sicura: "
                "non ha senso come posa di riposo"
            )

        with open(REST_POSE_PATH, "w") as f:
            json.dump({
                "s1": round(s1, 2), "s2": round(s2, 2),
                "nota": "posa del braccio durante navigazione e "
                        "avvicinamento; decide l'inquadratura della camera",
            }, f, indent=2, ensure_ascii=False)
        log.info("posa di riposo impostata a (%.1f, %.1f)", s1, s2)
        return (s1, s2)

    def go_rest(self):
        """Porta il braccio alla posa di riposo.

        Se non ne e' stata memorizzata nessuna, ripiega sul centro della
        regione sicura, che e' sicuro ma non e' detto che inquadri bene.
        """
        posa = self.get_rest_pose()
        if posa is None:
            return self.recover()
        self._move_joint_raw(SHOULDER, posa[0])
        time.sleep(SETTLE)
        self._move_joint_raw(ELBOW, posa[1])
        time.sleep(SETTLE)
        return self.angles()

    # ------------------------------------------------------------------
    # Bersaglio del tasto B
    # ------------------------------------------------------------------
    def get_waypoint(self):
        """Bersaglio salvato per il tasto B.

        Se non e' mai stato salvato nulla, ritorna DEFAULT_WAYPOINT
        invece di None: a differenza della posa di riposo, qui si vuole
        un bersaglio utilizzabile subito, senza dover prima premere V.
        """
        if not os.path.exists(WAYPOINT_PATH):
            return DEFAULT_WAYPOINT
        try:
            with open(WAYPOINT_PATH) as f:
                d = json.load(f)
            return (float(d["s1"]), float(d["s2"]))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            log.warning("bersaglio salvato non leggibile: %s", e)
            return DEFAULT_WAYPOINT

    def set_waypoint(self, s1=None, s2=None):
        """Memorizza la posa attuale (o quella indicata) come bersaglio
        del tasto B. Scritta su disco: vale anche nelle sessioni dopo.

        A differenza di set_rest_pose(), qui NON si richiede il margine
        pieno di is_safe() (region.margin, di norma 3.0 gradi): un
        bersaglio per "afferra" e' per natura vicino all'oggetto, quindi
        vicino al bordo della regione verificata, e pretendere lo stesso
        margine della posa di riposo lo renderebbe impossibile da
        salvare pur essendo una posa in cui l'utente si trova davvero,
        con il robot sotto controllo. Si richiede solo che la posa sia
        DENTRO la regione (margine positivo), non fuori.
        """
        if s1 is None or s2 is None:
            s1, s2 = self.angles()

        margine = self.region.margin_to_boundary(s1, s2)
        if margine <= 0:
            raise UnsafePose(
                f"({s1:.1f}, {s2:.1f}) e' fuori dalla regione sicura "
                f"(margine {margine:.1f}): non ha senso come bersaglio"
            )

        with open(WAYPOINT_PATH, "w") as f:
            json.dump({
                "s1": round(s1, 2), "s2": round(s2, 2),
                "nota": "bersaglio del tasto B, per tornare a una posa "
                        "scelta senza rifare il jog manuale ogni volta",
            }, f, indent=2, ensure_ascii=False)
        log.info("bersaglio impostato a (%.1f, %.1f)", s1, s2)
        return (s1, s2)

    def max_excursion(self, speed=PULSE_SPEED):
        """Spostamento massimo di un impulso, nel caso peggiore."""
        return abs(speed) * EXCURSION_PER_UNIT

    # ------------------------------------------------------------------
    # Sicurezza
    # ------------------------------------------------------------------
    def stop(self):
        """Ferma immediatamente entrambi i giunti.

        Manda sia drive_speed(0), che ferma il moto in velocita', sia
        pause(), che interrompe un'eventuale azione moveto. Servono
        entrambi: sono due meccanismi diversi.
        """
        for index in (SHOULDER, ELBOW):
            self._stop_speed(index)
            try:
                with self._lock:
                    self._servo.pause(index=index)
            except Exception as e:
                log.debug("pause(index=%d) fallita: %s", index, e)

    # ------------------------------------------------------------------
    # Posa di riposo
    # ------------------------------------------------------------------
    def get_home(self):
        """Posa di riposo salvata, o il centro della regione."""
        if os.path.exists(HOME_PATH):
            try:
                with open(HOME_PATH) as f:
                    d = json.load(f)
                return (float(d["s1"]), float(d["s2"]))
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
                log.warning("posa di riposo illeggibile (%s)", e)
        return self.region.center

    def set_home(self, s1=None, s2=None):
        """Salva la posa di riposo. Senza argomenti usa quella attuale.

        Rifiuta pose non sicure: una posa di riposo fuori regione
        renderebbe pericoloso ogni rientro.
        """
        if s1 is None or s2 is None:
            s1, s2 = self.angles()
        if not self.region.is_safe(s1, s2):
            raise UnsafePose(
                f"({s1:.1f}, {s2:.1f}) non e' dentro la regione sicura: "
                "premi N per registrarla come sicura, poi riprova"
            )
        with open(HOME_PATH, "w") as f:
            json.dump({
                "s1": round(s1, 2), "s2": round(s2, 2),
                "nota": "posa di riposo: decide cosa inquadra la camera",
            }, f, indent=2)
        log.info("posa di riposo impostata a (%.1f, %.1f)", s1, s2)
        return (s1, s2)

    def go_home(self):
        """Porta il braccio alla posa di riposo."""
        s1, s2 = self.get_home()
        log.info("rientro alla posa di riposo (%.1f, %.1f)", s1, s2)
        self._move_joint_raw(SHOULDER, s1)
        time.sleep(SETTLE)
        self._move_joint_raw(ELBOW, s2)
        time.sleep(SETTLE)
        return self.angles()

    def recover(self):
        """Riporta il braccio al centro della regione sicura.

        Da usare quando si parte, o si finisce, fuori regione. Il centro
        e' sicuro per costruzione: e' il baricentro di pose verificate.
        """
        c1, c2 = self.region.center
        log.info("rientro verso il centro (%.1f, %.1f)", c1, c2)
        self._move_joint_raw(SHOULDER, c1)
        time.sleep(SETTLE)
        self._move_joint_raw(ELBOW, c2)
        time.sleep(SETTLE)
        return self.angles()

    def learn_pose(self):
        """Registra la posa attuale come sicura, allargando la regione.

        Da chiamare SOLO dopo aver verificato a occhio che il braccio
        non stia toccando niente.
        """
        s1, s2 = self.angles()
        self.region.add_point(s1, s2)
        log.info("posa (%.1f, %.1f) aggiunta alla regione sicura", s1, s2)
        return s1, s2


class Watchdog:
    """Sorveglia la posa e ferma i giunti se esce dalla regione sicura."""

    def __init__(self, arm, hz=20, on_trip=None):
        self.arm = arm
        self.period = 1.0 / hz
        self.on_trip = on_trip
        self._stop = threading.Event()
        self._thread = None
        self.tripped = False
        self.last_pose = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                # angles() legge la cache del poller: nessun traffico
                # aggiuntivo sul canale.
                pose = self.arm.angles()
            except ArmError:
                time.sleep(self.period)
                continue
            self.last_pose = pose
            if not self.arm.region.is_safe(*pose):
                self.tripped = True
                log.error("watchdog: posa (%.1f, %.1f) fuori regione", *pose)
                self.arm.stop()
                if self.on_trip:
                    self.on_trip(pose)
                self._stop.set()
                return
            time.sleep(self.period)

    def start(self):
        self.tripped = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


class Gripper:
    """Pinza.

    Nessun feedback di presa: sub_status non distingue chiusura a vuoto
    da chiusura su oggetto. Il successo va verificato altrimenti.
    """

    def __init__(self, ep_gripper):
        self._g = ep_gripper

    def open(self, power=50, wait=1.5):
        self._g.open(power=power)
        if wait:
            time.sleep(wait)
            self._g.pause()

    def close(self, power=50, wait=1.5):
        self._g.close(power=power)
        if wait:
            time.sleep(wait)
            self._g.pause()

    def pause(self):
        self._g.pause()