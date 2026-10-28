"""Regione sicura del braccio nel piano dei giunti (servo1, servo2).

Perche' un inviluppo e non delle soglie
---------------------------------------
I vincoli sono almeno due e di natura diversa: il braccio disteso in
basso tocca il piano, e il braccio ripiegato all'indietro urta il modulo
sensori sul telaio. Descriverli con soglie su un singolo giunto non
funziona, perche' il limite di un giunto dipende dall'altro.

L'approccio qui e' diverso: si parte da un insieme di pose VERIFICATE A
MANO come sicure, se ne prende l'inviluppo convesso e lo si restringe di
un margine. Tutto cio' che cade dentro e' sicuro per costruzione, perche'
racchiuso da pose realmente provate.

E' conservativo: se la regione sicura vera fosse concava, si perde
qualcosa. In cambio non serve sapere quale vincolo e' attivo dove, e la
regione si allarga da sola aggiungendo punti verificati.

Tutti gli angoli sono in gradi, come li riporta servo.get_angle().
"""

import json
import logging
import math
import os

log = logging.getLogger(__name__)

# Le pose apprese in sessione (tasto N, oppure arm.learn_pose) vengono
# accumulate qui, cosi' non si perdono alla chiusura del programma.
# Il file sta accanto al modulo: e' dati di taratura di QUESTO robot,
# non del progetto.
LEARNED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "safe_points_learned.json")

# ---------------------------------------------------------------------------
# Pose verificate a mano come sicure. Ogni coppia e' una posa in cui il
# braccio e' stato portato senza toccare nulla.
#
# Per allargare la regione: aggiungi qui altre pose verificate.
# BUCO NOTO: fra s1=206 e s1=227 non ci sono campioni, quindi li'
# l'inviluppo taglia dritto e si perde spazio utilizzabile.
# ---------------------------------------------------------------------------
SAFE_POINTS = [
    (180.64, 161.60),
    (181.04, 182.90),
    (193.00, 129.10),   # raggiunta con moveto, nessun contatto
    (195.60, 143.70),
    (195.60, 143.82),
    (196.18, 161.66),
    (196.30, 121.40),
    (206.46, 108.90),
    (208.88, 182.94),
    (215.50, 174.70),
    (215.58, 174.70),
    (227.20,  95.80),
    (229.24, 145.80),
    (235.78,  79.80),
    (236.16, 174.04),
    (236.20, 174.10),
    (238.30, 137.56),
    (248.00, 129.00),   # raggiunta con moveto, nessun contatto
    (254.30, 111.36),
    (254.54,  79.80),
]

# Margine di sicurezza in gradi, applicato verso l'interno della regione.
# La lettura dei servo ha dispersione ~0.2 gradi, quindi il margine serve
# a coprire il gioco meccanico e l'imprecisione del campionamento a mano,
# non il rumore.
SAFETY_MARGIN = 3.0


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Inviluppo convesso, catena monotona di Andrew. Senso antiorario."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def centroid(poly):
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def point_in_polygon(p, poly):
    """Ray casting. I punti sul bordo possono cadere da una parte o
    dall'altra: irrilevante, perche' il margine ci tiene lontani."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xint = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < xint:
                inside = not inside
    return inside


def _dist_point_segment(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_boundary(p, poly):
    """Distanza in gradi dal bordo del poligono. Sempre positiva."""
    return min(
        _dist_point_segment(p, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )


# ---------------------------------------------------------------------------
# Regione sicura
# ---------------------------------------------------------------------------
class SafeRegion:
    """Regione sicura nel piano dei giunti, con verifica e saturazione."""

    def __init__(self, points=None, margin=SAFETY_MARGIN,
                 load_learned=True, path=LEARNED_PATH):
        self.points = list(points or SAFE_POINTS)
        self.margin = margin
        self.path = path
        self.learned = []
        if load_learned:
            self.load()
        self._rebuild()

    # ------------------------------------------------------------------
    def load(self):
        """Carica le pose apprese nelle sessioni precedenti."""
        if not self.path or not os.path.exists(self.path):
            return 0
        try:
            with open(self.path) as f:
                dati = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("pose apprese non leggibili (%s): le ignoro", e)
            return 0
        nuovi = [(float(a), float(b)) for a, b in dati.get("points", [])]
        self.learned = nuovi
        for p in nuovi:
            if p not in self.points:
                self.points.append(p)
        if nuovi:
            log.info("caricate %d pose apprese da %s",
                     len(nuovi), self.path)
        return len(nuovi)

    def save(self):
        """Scrive su disco le pose apprese. Le originali restano nel
        codice: qui finisce solo quello che si e' imparato dopo."""
        if not self.path:
            return False
        try:
            with open(self.path, "w") as f:
                json.dump({
                    "note": "pose verificate sicure, aggiunte in sessione",
                    "points": [[round(a, 2), round(b, 2)]
                               for a, b in self.learned],
                }, f, indent=2)
            return True
        except OSError as e:
            log.error("salvataggio delle pose apprese fallito: %s", e)
            return False

    def _rebuild(self):
        self.hull = convex_hull(self.points)
        self.center = centroid(self.hull)

    def add_point(self, s1, s2, persist=True):
        """Aggiunge una posa verificata sicura e allarga la regione.

        Con l'erosione corretta questa operazione puo' solo ALLARGARE:
        l'inviluppo convesso di piu' punti contiene quello di meno
        punti, e il criterio di sicurezza si appoggia direttamente a
        quello.
        """
        p = (float(s1), float(s2))
        self.points.append(p)
        self.learned.append(p)
        self._rebuild()
        if persist:
            self.save()
        return p

    # ------------------------------------------------------------------
    def is_safe(self, s1, s2):
        """Dentro l'inviluppo e ad almeno `margin` gradi dal bordo."""
        p = (s1, s2)
        if not point_in_polygon(p, self.hull):
            return False
        return distance_to_boundary(p, self.hull) >= self.margin

    def margin_to_boundary(self, s1, s2):
        """Gradi dal bordo dell'inviluppo. Negativo se fuori.

        Attenzione: e' la distanza dal bordo GREZZO, non dalla soglia di
        sicurezza. Un punto con margine 2 e margine richiesto 3 e' dentro
        l'inviluppo ma non e' considerato sicuro.
        """
        p = (s1, s2)
        d = distance_to_boundary(p, self.hull)
        return d if point_in_polygon(p, self.hull) else -d

    def clamp(self, s1, s2):
        """Riporta un bersaglio dentro la regione sicura.

        Se il punto e' gia' dentro, lo lascia com'e'. Altrimenti lo
        proietta lungo la congiungente col centro della regione, con una
        ricerca binaria: semplice e sufficiente su un convesso.

        Ritorna (s1, s2, saturato).
        """
        if self.is_safe(s1, s2):
            return s1, s2, False

        cx, cy = self.center
        lo, hi = 0.0, 1.0        # 0 = centro, 1 = punto richiesto
        for _ in range(40):
            mid = (lo + hi) / 2
            p = (cx + (s1 - cx) * mid, cy + (s2 - cy) * mid)
            if self.is_safe(*p):
                lo = mid
            else:
                hi = mid
        return cx + (s1 - cx) * lo, cy + (s2 - cy) * lo, True

    # ------------------------------------------------------------------
    def describe(self):
        s1s = [p[0] for p in self.hull]
        s2s = [p[1] for p in self.hull]
        return (
            f"SafeRegion: {len(self.points)} pose verificate, "
            f"{len(self.hull)} vertici di inviluppo, margine "
            f"{self.margin:.1f} gradi\n"
            f"  s1 in [{min(s1s):.1f}, {max(s1s):.1f}]  "
            f"s2 in [{min(s2s):.1f}, {max(s2s):.1f}]\n"
            f"  centro: ({self.center[0]:.1f}, {self.center[1]:.1f})"
        )

    def ascii_plot(self, width=56, height=22):
        """Disegno grezzo della regione, per controllarla a occhio."""
        s1s = [p[0] for p in self.hull]
        s2s = [p[1] for p in self.hull]
        x0, x1 = min(s1s) - 4, max(s1s) + 4
        y0, y1 = min(s2s) - 4, max(s2s) + 4

        grid = [[" "] * width for _ in range(height)]
        for r in range(height):
            for c in range(width):
                x = x0 + (x1 - x0) * c / (width - 1)
                y = y1 - (y1 - y0) * r / (height - 1)
                if self.is_safe(x, y):
                    grid[r][c] = "#"
                elif point_in_polygon((x, y), self.hull):
                    grid[r][c] = "."

        for px, py in self.points:
            c = int(round((px - x0) / (x1 - x0) * (width - 1)))
            r = int(round((y1 - py) / (y1 - y0) * (height - 1)))
            if 0 <= r < height and 0 <= c < width:
                grid[r][c] = "o"

        lines = [f"  s2 {y1:6.1f} +" + "-" * width]
        for r, row in enumerate(grid):
            lines.append("            |" + "".join(row))
        lines.append(f"     {y0:6.1f} +" + "-" * width)
        lines.append(f"            {x0:.0f}" +
                     " " * (width - 8) + f"{x1:.0f}  s1")
        lines.append("       # = sicuro   . = margine   o = posa verificata")
        return "\n".join(lines)


# Istanza predefinita, pronta all'uso.
SAFE_REGION = SafeRegion()


def is_safe(s1, s2):
    return SAFE_REGION.is_safe(s1, s2)


def clamp(s1, s2):
    return SAFE_REGION.clamp(s1, s2)


if __name__ == "__main__":
    r = SAFE_REGION
    print(r.describe())
    print()
    print(r.ascii_plot())
    print()

    prove = [
        (215.0, 140.0, "centro della regione"),
        (236.0, 174.0, "posa verificata, vicino al bordo"),
        (236.0,  60.0, "sotto il limite basso"),
        (185.0, 190.0, "troppo indietro"),
        (270.0, 100.0, "fuori dall'arco di s1"),
    ]
    print("  prova                              richiesto -> effettivo")
    for s1, s2, nome in prove:
        cs1, cs2, sat = r.clamp(s1, s2)
        flag = "SATURATO" if sat else "ok"
        print(f"  {nome:<34} ({s1:6.1f},{s2:6.1f}) -> "
              f"({cs1:6.1f},{cs2:6.1f})  {flag}")