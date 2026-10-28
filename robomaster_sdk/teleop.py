"""Teleoperazione da tastiera.

La qualita' di questa interfaccia determina la qualita' del dataset: se
teleoperare e' scomodo, le dimostrazioni vengono sporche e il modello
impara male. Vale la pena curarla.

Scelte di progetto
------------------
- La tastiera non da' eventi di rilascio, solo di pressione. Il chassis
  usa quindi una velocita' che DECADE: ogni pressione rinnova un
  bersaglio, e se non arrivano pressioni per HOLD_TIME secondi il
  bersaglio torna a zero. Con l'auto-ripetizione del sistema operativo
  si ottiene un controllo continuo tenendo premuto.

  Il valore comandato pero' non salta di colpo al bersaglio: lo
  rincorre con una rampa lineare (RAMP_TIME secondi da fermo a pieno
  regime). Senza rampa il canale, campionato a 10 Hz, vede solo la
  coppia (0, vmax): e' esattamente quello che ha reso vx binaria nel
  primo dataset (tenere premuto piu' a lungo non aiuta, perche' ogni
  pressione imposta comunque lo stesso valore fisso). Con la rampa,
  ogni accelerazione e ogni frenata attraversano qualche campione a
  velocita' intermedia, e nel dataset compare finalmente una vx (e
  vyaw) continua invece che solo "fermo" o "tutta".

- Il braccio si comanda per INCREMENTI di angolo, non per velocita': i
  servo accettano solo posizioni. Ogni comando viene saturato dentro la
  regione sicura PRIMA di partire, e i comandi sono limitati in
  frequenza per non intasare il canale.

- Il ciclo gira a RECORD_HZ, la stessa frequenza a cui verranno
  registrati gli episodi, cosi' il comportamento in teleoperazione e
  quello in registrazione coincidono.

- Modalita' fase fissa (parametro fase_fissa): si puo' bloccare la
  teleoperazione su UNA sola fase per tutta la sessione, per registrare
  episodi brevi e mirati (es. solo "afferra") invece dell'intero
  compito in sequenza. TAB/BACKSPACE restano disabilitati e ogni nuovo
  episodio riparte dalla fase bloccata invece che dalla prima. Serve a
  ridurre i frame di transizione/fermi che si accumulano quando si
  registra il compito intero in un solo episodio.

Hook di registrazione
---------------------
`on_tick(state, action, flags)` viene chiamato a ogni ciclo. Il
registratore si aggancia li' senza che la teleoperazione sappia niente
di dataset o formati.

Comandi
-------
  W/S    chassis avanti / indietro
  A/D    traslazione sinistra / destra
  Q/E    rotazione antioraria / oraria
  F      cambia velocita': preciso -> normale -> veloce
  + / -  alza / abbassa di un livello
  I/K    spalla su / giu'   (un impulso per pressione, ~5 gradi)
  J/L    gomito su / giu'   (rifiutato se porterebbe fuori regione)
  O/P    pinza apre / chiude
  SPAZIO arresto immediato di tutto
  R      avvia / ferma la registrazione
  TAB    passa alla fase successiva del compito (disabilitato se fase fissa)
  BKSP   torna alla fase precedente (disabilitato se fase fissa)
  1 / 2  marca l'episodio come riuscito / fallito
  N      registra la posa attuale del braccio come sicura
  H      rientro automatico alla posa di riposo, a impulsi (registrato
         come il resto: vedi sotto)
  M      salva la posa attuale COME posa di riposo (guarda il video!)
  B      rientro automatico al bersaglio salvato (stesso meccanismo di H,
         bersaglio diverso: pensato per un punto di partenza ricorrente,
         es. l'inizio di "afferra", quando si registrano tanti episodi
         della stessa fase di seguito)
  V      salva la posa attuale come bersaglio del tasto B
  X      esci
"""

import logging
import sys
import time

try:
    import cv2
except ImportError:      # pragma: no cover
    cv2 = None

from . import config
from .arm import ELBOW, SHOULDER, ArmError, UnsafePose, UnsafePose

log = logging.getLogger(__name__)

# Quanto dura una pressione prima che il bersaglio di velocita' decada
# a zero.
HOLD_TIME = 0.25

# Tempo di rampa da fermo a pieno regime (e viceversa), in secondi.
# Vedi la nota in cima al file: e' quello che rompe il salto istantaneo
# 0/vmax e fa comparire velocita' intermedie nel dataset campionato a
# 10 Hz.
RAMP_TIME = 0.3

# Livelli di velocita' del chassis, come frazione del massimo.
# Allinearsi a un oggetto per afferrarlo richiede spostamenti di pochi
# centimetri: a velocita' piena ogni pressione ne fa troppi, e il
# posizionamento fine diventa impossibile. Il livello "preciso" esiste
# per quello.
LIVELLI = [
    ("preciso", 0.12),
    ("veloce", 1),
]
LIVELLO_INIZIALE = 1

# In modalita' precisa anche la durata dell'impulso si accorcia: la
# combinazione di velocita' bassa e impulso breve da' lo spostamento
# minimo utile, circa un centimetro.
HOLD_TIME_PRECISO = 0.12

# --------------------------------------------------------------------------
# Fasi del compito
# --------------------------------------------------------------------------
# Ogni campione registrato porta con se' la fase in corso, e con essa
# l'istruzione in linguaggio naturale da dare al modello.
#
# Perche' non basta un'unica istruzione per tutto l'episodio: nella fase
# di ricerca girare a sinistra o a destra vanno bene uguale, e un
# modello che impara la media di comportamenti opposti produce "fermo".
# Con l'istruzione della fase, "cerca" e "afferra" diventano compiti
# diversi con azioni diverse, e l'ambiguita' sparisce.
#
# Etichettare non costringe a niente: si puo' addestrare un modello
# unico su tutto, oppure modelli separati per fase, oppure confrontare
# le due cose. Senza etichette nessuna delle tre e' possibile.
# Le fasi sono simmetriche fra oggetto e cestino, e non per eleganza:
# "cerca" e "avvicina" sono comportamenti diversi, perche' nel primo il
# bersaglio non e' ancora in vista e nel secondo si'. Tenerli insieme
# riporta la stessa ambiguita' che le fasi servono a togliere.
#
# QUESTA E' L'UNICA COPIA: run_policy.py la importa da qui invece di
# tenerne una seconda, per non doverle tenere sincronizzate a mano.
FASI = [
    ("cerca_oggetto",    "cerca l'oggetto"),
    ("avvicina_oggetto", "avvicinati all'oggetto"),
    ("allinea",          "allineati per afferrare"),
    ("afferra",          "afferra l'oggetto"),
    ("cerca_cestino",    "cerca il cestino"),
    ("avvicina_cestino", "avvicinati al cestino"),
    ("deposita",         "appoggia l'oggetto nel cestino"),
]

# Velocita' degli impulsi al braccio, in unita' dell'SDK.
# NON alzarla per andare piu' veloce: lo spostamento e' proporzionale
# alla velocita', ed e' cosi' che gli impulsi a 20 e 30 hanno colpito il
# sensore IR. Per muoversi di piu' si preme piu' volte.
ARM_PULSE_SPEED = 5

# Frequenza massima degli impulsi al braccio. Ogni impulso costa una
# chiamata di avvio e una di arresto sul canale sincrono.
ARM_CMD_HZ = 4.0

# Tolleranza di arrivo per il rientro automatico (tasto H). Sotto
# questa soglia l'isteresi/ripetibilita' del servo (~1-1.2 gradi,
# vedi arm.py) potrebbe non permettere di azzerare l'errore del
# tutto, quindi ci si ferma un po' prima invece di oscillare.
AUTO_RETURN_TOLERANCE = 3.0


# ---------------------------------------------------------------------------
# Lettura tastiera non bloccante
# ---------------------------------------------------------------------------
class _KeyReader:
    """Lettura non bloccante, Windows (msvcrt) o POSIX (termios)."""

    def __init__(self):
        self.windows = sys.platform.startswith("win")
        self._old = None
        if self.windows:
            import msvcrt
            self._msvcrt = msvcrt
        else:
            import termios
            import tty
            self._termios = termios
            self._tty = tty

    def __enter__(self):
        if not self.windows:
            self._fd = sys.stdin.fileno()
            self._old = self._termios.tcgetattr(self._fd)
            self._tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if not self.windows and self._old is not None:
            self._termios.tcsetattr(
                self._fd, self._termios.TCSADRAIN, self._old)
        return False

    def get_keys(self):
        """Tutti i tasti in coda, in minuscolo. Lista vuota se nessuno."""
        keys = []
        if self.windows:
            while self._msvcrt.kbhit():
                ch = self._msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):    # prefisso tasti speciali
                    self._msvcrt.getch()
                    continue
                try:
                    keys.append(ch.decode("ascii", "ignore").lower())
                except Exception:
                    pass
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                keys.append(ch.lower())
        return [k for k in keys if k]


# ---------------------------------------------------------------------------
# Velocita' con decadimento
# ---------------------------------------------------------------------------
class _DecayingAxis:
    """Un asse di velocita' rinnovato dalle pressioni, azzerato dal tempo.

    Non insegue un bersaglio di colpo: lo rincorre con una rampa
    lineare (RAMP_TIME secondi da fermo a vmax pieno), alla stessa
    velocita' angolare/lineare indipendentemente dal livello di
    velocita' scelto. E' quello che rompe la coppia (0, vmax) e fa
    comparire valori intermedi nei campioni registrati.
    """

    def __init__(self, vmax, ramp_time=None):
        self.vmax = vmax
        self.value = 0.0
        self.target = 0.0
        self.hold = HOLD_TIME
        self.ramp_time = RAMP_TIME if ramp_time is None else ramp_time
        self._last = 0.0
        self._last_update = None

    def push(self, direction, scala=1.0):
        self.target = direction * self.vmax * scala
        self._last = time.time()

    def set_target(self, value):
        """Imposta direttamente il bersaglio (retromarcia: limite diverso
        dal livello di velocita' corrente, ma stessa rampa)."""
        self.target = value
        self._last = time.time()

    def update(self, now):
        if now - self._last > self.hold:
            self.target = 0.0

        dt = 0.0 if self._last_update is None else now - self._last_update
        self._last_update = now

        if self.ramp_time <= 0 or dt <= 0:
            self.value = self.target
        else:
            passo_max = abs(self.vmax) * dt / self.ramp_time
            diff = self.target - self.value
            if abs(diff) <= passo_max:
                self.value = self.target
            else:
                self.value += passo_max if diff > 0 else -passo_max
        return self.value

    def zero(self):
        self.value = 0.0
        self.target = 0.0


# ---------------------------------------------------------------------------
class Teleop:
    """Teleoperazione da tastiera di chassis, braccio e pinza."""

    def __init__(self, robot, on_tick=None, hz=None, camera=None,
                 show_video=False, fase_fissa=None):
        self.robot = robot
        self.camera = camera
        # La finestra video e' utile per inquadrare, ma i tasti restano
        # letti dalla CONSOLE: tienila come seconda finestra e continua
        # a digitare nel terminale.
        self.show_video = show_video and camera is not None and cv2 is not None
        self.on_tick = on_tick
        self.hz = hz or config.RECORD_HZ
        self.period = 1.0 / self.hz

        self.vx = _DecayingAxis(config.CHASSIS_VX_MAX)
        self.vy = _DecayingAxis(config.CHASSIS_VY_MAX)
        self.vyaw = _DecayingAxis(config.CHASSIS_VYAW_MAX)
        self._livello = LIVELLO_INIZIALE
        self._applica_livello()

        # Fase corrente del compito. Avanza col tasto TAB, torna
        # indietro con BACKSPACE, si azzera a ogni nuovo episodio.
        #
        # Se fase_fissa e' impostata, la teleoperazione resta bloccata
        # su quella fase per tutta la sessione: niente TAB/BACKSPACE,
        # e ogni nuovo episodio riparte da li' invece che dalla fase 0.
        # Serve a registrare un'unica azione per volta (es. solo
        # "afferra"), senza dover rifare ogni volta l'intera sequenza
        # delle 7 fasi: episodi piu' corti e mirati, meno frame di
        # transizione/fermi nel dataset.
        self._fase_fissa = None
        if fase_fissa is not None:
            nomi = [n for n, _ in FASI]
            if fase_fissa not in nomi:
                raise ValueError(
                    f"fase '{fase_fissa}' sconosciuta. Valide: {nomi}")
            self._fase_fissa = nomi.index(fase_fissa)
        self._fase = self._fase_fissa if self._fase_fissa is not None else 0

        # Nessun bersaglio accumulato: solo una richiesta di impulso
        # per il ciclo corrente, che scade se non viene eseguita.
        # (index, direction, margine_soglia) oppure None. Il terzo
        # elemento e' quasi sempre None (usa PULSE_SAFETY_MARGIN, la
        # soglia prudente per il jog libero); _step_auto_return() lo
        # valorizza con una soglia piu' permissiva, perche' li' si sta
        # tornando verso un punto gia' verificato come sicuro, non
        # esplorando alla cieca.
        self._arm_request = None
        self._last_arm_cmd = 0.0

        # Rientro automatico (tasto H): (s1, s2) bersaglio, o None se
        # non attivo. Non muove il braccio direttamente: a ogni ciclo
        # _step_auto_return() chiede UN impulso verso il bersaglio,
        # passando dallo stesso meccanismo di I/K/J/L. Cosi' il
        # rientro finisce nei dati registrati come una sequenza di
        # impulsi veri, invece di un salto di stato senza un'azione
        # che lo spieghi.
        self._auto_target = None
        # Impulso effettivamente ESEGUITO nel ciclo corrente, per il
        # registratore: (index, direction) oppure None. Diverso dalla
        # richiesta, che puo' essere rifiutata.
        self._arm_pulse_done = None

        self.gripper_cmd = 0.0          # 0 aperta, 1 chiusa
        self.recording = False
        self.last_mark = None
        self.running = False
        self._messages = []

    # ------------------------------------------------------------------
    def _applica_livello(self):
        nome, scala = LIVELLI[self._livello]
        self.scala = scala
        hold = HOLD_TIME_PRECISO if nome == "preciso" else HOLD_TIME
        for asse in (self.vx, self.vy, self.vyaw):
            asse.hold = hold
        return nome

    @property
    def livello_nome(self):
        return LIVELLI[self._livello][0]

    def _fase_iniziale(self):
        """Fase da cui ripartire a ogni nuovo episodio."""
        return self._fase_fissa if self._fase_fissa is not None else 0

    def _msg(self, text):
        self._messages.append(text)
        if len(self._messages) > 3:
            self._messages.pop(0)

    def _on_watchdog(self, pose):
        self._msg(f"WATCHDOG ({pose[0]:.1f}, {pose[1]:.1f}) - premi H")

    # ------------------------------------------------------------------
    def _handle_key(self, k):
        if k == "w":
            self.vx.push(+1, self.scala)
        elif k == "s":
            # Il retro e' cieco: nessun sensore IR dietro, quindi la
            # retromarcia resta limitata a prescindere dal livello.
            # Stessa rampa della marcia avanti: niente scatto diretto.
            self.vx.set_target(-min(config.CHASSIS_VX_MAX * self.scala,
                                    config.REVERSE_VX_MAX))
        elif k == "a":
            self.vy.push(-1, self.scala)
        elif k == "d":
            self.vy.push(+1, self.scala)
        elif k == "q":
            self.vyaw.push(-1, self.scala)
        elif k == "e":
            self.vyaw.push(+1, self.scala)

        elif k == "\t":
            if self._fase_fissa is not None:
                self._msg("fase fissa attiva: TAB disabilitato")
            elif self._fase < len(FASI) - 1:
                self._fase += 1
                self._msg(f"fase {self._fase+1}/{len(FASI)}: "
                          f"{FASI[self._fase][1]}")
            else:
                self._msg("ultima fase")
        elif k in ("\x08", "\x7f"):
            if self._fase_fissa is not None:
                self._msg("fase fissa attiva: BKSP disabilitato")
            elif self._fase > 0:
                self._fase -= 1
                self._msg(f"fase {self._fase+1}/{len(FASI)}: "
                          f"{FASI[self._fase][1]}")

        elif k == "f":
            # Scorre i livelli. Serve spesso e deve stare su un tasto
            # solo: durante l'allineamento si passa da veloce a preciso
            # e viceversa di continuo.
            self._livello = (self._livello + 1) % len(LIVELLI)
            nome = self._applica_livello()
            self._msg(f"velocita': {nome} ({self.scala:.2f})")
        elif k == "+" or k == "=":
            self._livello = min(len(LIVELLI) - 1, self._livello + 1)
            self._msg(f"velocita': {self._applica_livello()}")
        elif k == "-" or k == "_":
            self._livello = max(0, self._livello - 1)
            self._msg(f"velocita': {self._applica_livello()}")

        elif k in ("i", "k", "j", "l"):
            # Nessun bersaglio accumulato: era il difetto che faceva
            # scappare via il braccio. Ogni pressione chiede UN impulso,
            # che verra' eseguito solo se la posa prevista resta dentro
            # la regione sicura. Se non viene eseguito, non resta
            # niente in sospeso.
            if self._auto_target is not None:
                self._auto_target = None
                self._msg("rientro automatico interrotto (controllo manuale)")
            index = SHOULDER if k in ("i", "k") else ELBOW
            direction = +1 if k in ("i", "j") else -1
            self._arm_request = (index, direction, None)

        elif k == "o":
            self.robot.gripper.open(wait=0)
            self.gripper_cmd = 0.0
            self._msg("pinza: apre")
        elif k == "p":
            self.robot.gripper.close(wait=0)
            self.gripper_cmd = 1.0
            self._msg("pinza: chiude")

        elif k == " ":
            self.stop_all()
            self._msg("ARRESTO")

        elif k == "r":
            self.recording = not self.recording
            if self.recording:
                # Ogni episodio riparte dalla fase iniziale: la prima
                # fase in generale, oppure la fase fissa se impostata.
                self._fase = self._fase_iniziale()
            self._msg("registrazione ON" if self.recording
                      else "registrazione OFF")
        elif k in ("1", "2"):
            if not self.recording:
                self._msg("nessun episodio in corso")
                return
            self.last_mark = "successo" if k == "1" else "fallimento"
            # La marcatura CHIUDE l'episodio, quindi deve spegnere anche
            # la registrazione. Lasciandola accesa il registratore
            # aprirebbe subito un episodio nuovo, che raccoglie qualche
            # frame finche' non si preme R.
            self.recording = False
            self._fase = self._fase_iniziale()
            self._msg(f"episodio {self.last_mark.upper()}, "
                      "premi R per il prossimo")

        elif k == "n":
            try:
                s1, s2 = self.robot.arm.learn_pose()
                self._msg(f"posa ({s1:.1f}, {s2:.1f}) aggiunta come sicura")
            except ArmError as e:
                self._msg(f"learn_pose fallita: {e}")

        elif k == "h":
            # Attiva il rientro automatico verso la posa di riposo:
            # NON muove il braccio qui. _step_auto_return(), chiamata
            # a ogni ciclo del loop principale, chiede un impulso alla
            # volta verso il bersaglio, esattamente come I/K/J/L, cosi'
            # il rientro viene registrato invece di saltare lo stato
            # senza un'azione corrispondente (era il problema della
            # vecchia go_rest(), che spostava i giunti con moveto()
            # diretto, bypassando pulse() e quindi il registratore).
            try:
                target = self.robot.arm.get_rest_pose()
                if target is None:
                    target = self.robot.arm.region.center
                self._auto_target = target
                self._msg(f"rientro automatico verso "
                          f"({target[0]:.1f}, {target[1]:.1f})")
            except ArmError as e:
                self._msg(f"rientro non avviato: {e}")

        elif k == "m":
            # Memorizza la posa attuale come riposo. Si usa una volta,
            # dopo aver sistemato l'inquadratura guardando il video.
            try:
                s1, s2 = self.robot.arm.set_rest_pose()
                self._msg(f"posa di riposo impostata a "
                          f"({s1:.1f}, {s2:.1f})")
            except (ArmError, UnsafePose) as e:
                self._msg(f"non memorizzata: {e}")

        elif k == "b":
            # Stesso meccanismo di H (impulsi veri, registrati), ma
            # verso un bersaglio diverso: il punto di partenza che si
            # vuole raggiungere senza jog manuale ogni volta, tipico
            # quando si registrano tanti episodi della stessa fase di
            # seguito con --fase.
            target = self.robot.arm.get_waypoint()
            self._auto_target = target
            self._msg(f"rientro automatico verso il bersaglio "
                      f"({target[0]:.1f}, {target[1]:.1f})")

        elif k == "v":
            # Memorizza la posa attuale come bersaglio del tasto B.
            try:
                s1, s2 = self.robot.arm.set_waypoint()
                self._msg(f"bersaglio impostato a ({s1:.1f}, {s2:.1f})")
            except (ArmError, UnsafePose) as e:
                self._msg(f"non memorizzato: {e}")

        elif k == "x":
            self.running = False

    # ------------------------------------------------------------------
    def stop_all(self):
        self.vx.zero()
        self.vy.zero()
        self.vyaw.zero()
        try:
            self.robot.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        except Exception:
            pass
        try:
            self.robot.arm.stop()
        except Exception:
            pass
        try:
            self.robot.gripper.pause()
        except Exception:
            pass
        # Nessun bersaglio da riallineare: con gli impulsi la posa
        # comandata coincide sempre con quella reale.
        self._arm_request = None
        self._auto_target = None

    # ------------------------------------------------------------------
    def _send_arm(self, now):
        """Esegue al massimo un impulso per ciclo."""
        if self._arm_request is None:
            return
        if now - self._last_arm_cmd < 1.0 / ARM_CMD_HZ:
            # Troppo presto: la richiesta viene SCARTATA, non accodata.
            # In teleoperazione conta quello che si sta premendo adesso.
            self._arm_request = None
            return

        index, direction, margine_soglia = self._arm_request
        self._arm_request = None

        eseguito, motivo = self.robot.arm.pulse(
            index, direction, speed=ARM_PULSE_SPEED,
            safety_margin=margine_soglia)
        if eseguito:
            self._arm_pulse_done = (index, direction)
        else:
            # Invece di limitarsi a rifiutare, dice quali tasti
            # funzionerebbero: da una posa vicina al bordo e' l'unica
            # informazione davvero utile.
            tasti = {(SHOULDER, +1): "I", (SHOULDER, -1): "K",
                     (ELBOW, +1): "J", (ELBOW, -1): "L"}
            liberi = [t for k, t in tasti.items()
                      if self.robot.arm.allowed_directions().get(
                          k, (False, 0))[0]]
            if liberi:
                self._msg(f"bloccato, puoi usare: {' '.join(liberi)}")
            else:
                self._msg("bloccato in ogni direzione: premi H")
        self._last_arm_cmd = now

    def _step_auto_return(self):
        """Un impulso per ciclo verso self._auto_target, se attivo.

        Usa lo stesso self._arm_request di I/K/J/L: da qui in poi il
        percorso e' identico a un impulso manuale, isteresi/soglie di
        sicurezza comprese, e finisce nel registratore allo stesso
        modo. Se il ciclo ha gia' una richiesta manuale (tasto appena
        premuto), quella ha la precedenza e questo passo non fa nulla.

        Soglia di sicurezza: NON si usa sempre region.margin (3.0
        gradi). Il bersaglio del tasto B (set_waypoint()) puo' avere un
        margine piu' basso apposta: e' una posa vicina all'oggetto,
        scelta con l'utente presente e il robot sotto controllo, non
        una posa "sicura in generale" come il centro della regione o
        la posa di riposo. Se si pretendesse comunque margine 3.0 per
        ogni impulso, gli ultimi passi verso un bersaglio del genere
        verrebbero rifiutati e il rientro si fermerebbe appena prima
        di arrivare (lo stesso sintomo del problema precedente, ma
        causato dal bersaglio invece che dalla soglia fissa).
        Si usa quindi il margine del BERSAGLIO stesso come soglia,
        con un minimo di 0.5 gradi (mai zero: un po' di cuscinetto
        resta sempre) e un massimo di region.margin (non si allarga
        MAI la soglia oltre quella usata per H, anche se il bersaglio
        fosse piu' profondo nella regione del necessario).

        Scelta del giunto: si preferisce quello piu' lontano dal
        bersaglio, ma SOLO se pulse() lo accetterebbe. Insistere sempre
        sullo stesso giunto quando e' lui il bloccato (vicino al bordo,
        nel verso che serve) e' quello che faceva sembrare il rientro
        "completamente immobile": la richiesta veniva rifiutata a ogni
        ciclo e non si provava mai l'altro giunto, che magari era
        libero. Qui invece si prova prima il preferito e, se non e'
        ammesso, si prova l'altro.
        """
        if self._auto_target is None or self._arm_request is not None:
            return
        try:
            s1, s2 = self.robot.arm.angles()
        except ArmError:
            return

        t1, t2 = self._auto_target
        e1, e2 = t1 - s1, t2 - s2
        if abs(e1) <= AUTO_RETURN_TOLERANCE and abs(e2) <= AUTO_RETURN_TOLERANCE:
            self._auto_target = None
            self._msg(f"rientro automatico completato a "
                      f"({s1:.1f}, {s2:.1f})")
            return

        margine_bersaglio = self.robot.arm.region.margin_to_boundary(t1, t2)
        soglia = max(0.5, min(self.robot.arm.region.margin, margine_bersaglio))
        candidati = []
        if abs(e1) > AUTO_RETURN_TOLERANCE:
            candidati.append((SHOULDER, 1 if e1 > 0 else -1, abs(e1)))
        if abs(e2) > AUTO_RETURN_TOLERANCE:
            candidati.append((ELBOW, 1 if e2 > 0 else -1, abs(e2)))
        candidati.sort(key=lambda c: -c[2])  # il piu' lontano prima

        for index, direction, _ in candidati:
            ammesso, _ = self.robot.arm.can_pulse(
                index, direction, speed=ARM_PULSE_SPEED,
                safety_margin=soglia)
            if ammesso:
                self._arm_request = (index, direction, soglia)
                return

        # Ne' il giunto preferito ne' l'altro possono avanzare verso il
        # bersaglio da qui: insistere non servirebbe, e' meglio dirlo
        # chiaramente invece di lasciare il tasto H premuto a vuoto.
        self._auto_target = None
        self._msg("rientro automatico bloccato: nessuna mossa sicura "
                  "verso il bersaglio da qui, usa I/K/J/L")

    def _read_state(self):
        """Stato corrente. Le letture che falliscono diventano None."""
        state = {"t": time.time()}
        try:
            # Cache del poller: non genera traffico.
            state["arm"] = self.robot.arm.angles()
            state["arm_age"] = self.robot.arm.reading_age
        except ArmError:
            state["arm"] = None
            state["arm_age"] = None
        state["gripper_cmd"] = self.gripper_cmd

        # L'immagine viene presa dalla cache del lettore video: non
        # blocca mai il ciclo, e si registra quanto e' vecchia.
        if self.camera is not None:
            img, age = self.camera.read()
            state["image"] = img
            state["image_age"] = age
        else:
            state["image"] = None
            state["image_age"] = None
        return state

    def _render(self, state):
        arm = state.get("arm")
        if arm:
            margine = self.robot.arm.region.margin_to_boundary(*arm)
            arm_s = f"{arm[0]:6.1f} {arm[1]:6.1f} m{margine:+5.1f}"
        else:
            arm_s = "      n/d           "
        rec = "REC" if self.recording else "   "
        age = state.get("image_age")
        if self.camera is None:
            cam_s = ""
        elif age is None:
            cam_s = "| img --- "
        else:
            cam_s = f"| img {age*1000:3.0f}ms "
        fase = FASI[self._fase][0]
        fissa_s = " [FASE FISSA]" if self._fase_fissa is not None else ""
        auto_s = " [RIENTRO AUTO]" if self._auto_target is not None else ""
        line = (f"\r {rec} |{self._fase+1}.{fase:<17}| "
                f"{self.livello_nome:<8}"
                f"{self.vx.value:+5.2f} "
                f"{self.vy.value:+5.2f} {self.vyaw.value:+5.2f} "
                f"| braccio {arm_s} "
                f"| pinza {self.gripper_cmd:.0f} {cam_s}{fissa_s}{auto_s}")
        if self._messages:
            line += f"| {self._messages[-1][:40]}"
        sys.stdout.write(line.ljust(120))
        sys.stdout.flush()

        if self.show_video:
            img = state.get("image")
            if img is not None:
                vista = img.copy()
                h, w = vista.shape[:2]
                lato = min(w, h)
                x0, y0 = (w - lato) // 2, (h - lato) // 2
                # Riquadro di cio' che il modello vede davvero: stesso
                # ritaglio quadrato centrale applicato da to_lerobot.py
                # (training) e policy_server.py (inferenza), tramite
                # image_prep.ritaglia_quadrato(). Qui e' solo
                # un'anteprima, il file non viene salvato ritagliato:
                # il ritaglio avviene a valle, sui due script Python
                # 3.12.
                cv2.rectangle(vista, (x0, y0),
                              (x0 + lato - 1, y0 + lato - 1), (0, 220, 0), 1)
                if self.recording:
                    cv2.circle(vista, (w - 24, 24), 9, (0, 0, 230), -1)
                    cv2.putText(vista, "REC", (w - 70, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 230), 2, cv2.LINE_AA)
                etichetta = (f"{self._fase+1}/{len(FASI)} "
                             f"{FASI[self._fase][1]}")
                cv2.putText(vista, etichetta, (10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vista, etichetta, (10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow("teleop", vista)
                # waitKey serve a far ridisegnare la finestra; i tasti
                # della teleoperazione arrivano dalla console, non da qui.
                cv2.waitKey(1)

    # ------------------------------------------------------------------
    def run(self):
        print(__doc__.split("Comandi\n-------\n")[1])
        if self._fase_fissa is not None:
            nome, istr = FASI[self._fase_fissa]
            print(f"Modalita' FASE FISSA: '{nome}' (\"{istr}\").")
            print("TAB/BACKSPACE disabilitati, ogni episodio riparte da "
                  "questa fase.\n")
        try:
            # Un solo lettore di angoli per tutti: watchdog,
            # teleoperazione e registratore leggono la stessa cache.
            self.robot.arm.start_polling()
            s1, s2 = self.robot.arm.angles()
            margine = self.robot.arm.margin(s1, s2)
            esc = self.robot.arm.max_excursion(ARM_PULSE_SPEED)
            print(f"Braccio a ({s1:.1f}, {s2:.1f}), margine "
                  f"{margine:.1f} gradi.")
            print(f"Ogni impulso sposta al massimo {esc:.1f} gradi.")
            if margine < esc + 4:
                print("Margine basso: premi H per rientrare al centro.")
        except ArmError as e:
            print(f"Braccio non leggibile ({e}): i comandi al braccio "
                  "verranno rifiutati.")

        self.running = True
        watchdog = self.robot.watchdog(on_trip=self._on_watchdog)
        with _KeyReader() as keys, watchdog as wd:
            try:
                while self.running:
                    now = time.time()

                    self._arm_pulse_done = None
                    for k in keys.get_keys():
                        self._handle_key(k)

                    self._step_auto_return()

                    vx = self.vx.update(now)
                    vy = self.vy.update(now)
                    vyaw = self.vyaw.update(now)

                    try:
                        self.robot.chassis.drive_speed(
                            x=vx, y=vy, z=vyaw * 57.2958, timeout=0.5)
                    except Exception as e:
                        log.debug("drive_speed fallita: %s", e)

                    self._send_arm(now)

                    state = self._read_state()
                    pulse = self._arm_pulse_done
                    action = {
                        "vx": vx, "vy": vy, "vyaw": vyaw,
                        # Impulso eseguito: -1, 0 o +1 per ciascun
                        # giunto. E' il comando vero dato al braccio.
                        "pulse_shoulder": (
                            pulse[1] if pulse and pulse[0] == SHOULDER else 0),
                        "pulse_elbow": (
                            pulse[1] if pulse and pulse[0] == ELBOW else 0),
                        "gripper": self.gripper_cmd,
                    }
                    flags = {"recording": self.recording,
                             "mark": self.last_mark,
                             "fase": FASI[self._fase][0],
                             "istruzione": FASI[self._fase][1],
                             "fase_idx": self._fase}
                    self.last_mark = None

                    if self.on_tick:
                        self.on_tick(state, action, flags)

                    # Dopo un intervento il watchdog si ferma da solo.
                    # Lo si riarma solo quando il braccio e' rientrato,
                    # altrimenti scatterebbe di continuo.
                    if wd.tripped:
                        arm = state.get("arm")
                        if arm and self.robot.arm.region.is_safe(*arm):
                            wd.start()
                            self._msg("watchdog riarmato")

                    self._render(state)

                    dt = self.period - (time.time() - now)
                    if dt > 0:
                        time.sleep(dt)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop_all()
                self.robot.arm.stop_polling()
                if self.show_video:
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
                print("\nTeleoperazione terminata.")