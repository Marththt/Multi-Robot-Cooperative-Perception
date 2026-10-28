"""One-off converter: data files written by the old Italian code -> the
English keys and values used by the current code.

It never modifies the old files: it reads from the old project folder and
writes converted copies into the new one.

    python -m tools.convert_data_to_english --old ../statico --new .

What it converts (old path -> new path, both relative to the project root):
    trials/*.jsonl                      -> trials/*.jsonl
    live/*.json                         -> live/*.json
    config/calib_scacchiera_<ID>.json   -> config/checkerboard_calib_<ID>.json
    config/regioni_optitrack.json       -> config/optitrack_regions.json
    config/esempio_aruco_classi.json    -> config/aruco_classes.json
    config/correzione_<ID>.json         -> config/frame_correction_<ID>.json
    any other config/*.json             -> same name, keys converted

Keys are renamed with KEYS below; the string values in VALUES (class
labels, zone names, lighting levels) are translated wherever they appear.
Free text (e.g. the "_leggimi" notes inside config files) is copied as it
is under the English key: rewrite it by hand if you care, or use the
English config files shipped with the code instead.

An existing destination file is never overwritten unless --overwrite is
given.
"""

import argparse
import glob
import json
import os
import sys

KEYS = {
    "classe": "class_name", "confidenza": "confidence", "posizione": "position",
    "distanza": "distance", "angolo_vista_rad": "viewing_angle_rad",
    "posa": "pose", "posa_robot": "robot_pose", "rilevamenti": "detections",
    "zona": "zone", "guasto_iniettato": "fault_injected", "scartato": "dropped",
    "regioni": "regions", "nome": "name", "_leggimi": "_readme",
    "_nota_rotazione": "_rotation_note", "marker_a_classe": "marker_to_class",
    "errore_rms_px": "rms_error_px", "dimensioni_immagine": "image_size",
    "n_pose": "n_poses", "rotazione_rad": "rotation_rad", "punti": "points",
    "vero": "true", "calcolato": "computed", "nota": "note",
    # old focal-length calibration files (arch/)
    "metodo": "method", "focale_px": "focal_px",
    "focale_deviazione_standard_px": "focal_std_px", "n_distanze": "n_distances",
    "n_campioni_totali": "n_samples_total", "distanze_usate_m": "distances_used_m",
    "aruco_marker_size_usato_m": "aruco_marker_size_used_m",
}

VALUES = {
    "scatola": "box", "bottiglia": "bottle", "bicchiere": "cup",
    "nord": "north", "sud": "south", "est": "east", "ovest": "west",
    "centro": "center", "sconosciuta": "unknown",
    "normale": "normal", "scarsa": "poor",
}

RENAMES = [
    ("config/regioni_optitrack.json", "config/optitrack_regions.json"),
    ("config/esempio_aruco_classi.json", "config/aruco_classes.json"),
]
PREFIX_RENAMES = [
    ("config/calib_scacchiera_", "config/checkerboard_calib_"),
    ("config/correzione_", "config/frame_correction_"),
]


def convert(value):
    if isinstance(value, dict):
        return {KEYS.get(k, k): convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert(v) for v in value]
    if isinstance(value, str):
        return VALUES.get(value, value)
    return value


def new_relative_path(rel):
    for old, new in RENAMES:
        if rel == old:
            return new
    for old, new in PREFIX_RENAMES:
        if rel.startswith(old):
            return new + rel[len(old):]
    return rel


def write(dest, text, overwrite):
    if os.path.exists(dest) and not overwrite:
        print(f"  SKIPPED (already exists, use --overwrite): {dest}")
        return False
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", required=True, help="old Italian project folder (statico/)")
    p.add_argument("--new", required=True, help="new English project folder")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if not os.path.isdir(a.old):
        sys.exit(f"{a.old} is not a folder")
    n = 0
    for pattern in ("trials/*.jsonl", "live/*.json", "config/*.json"):
        for src in sorted(glob.glob(os.path.join(a.old, pattern))):
            rel = os.path.relpath(src, a.old).replace(os.sep, "/")
            dest = os.path.join(a.new, new_relative_path(rel))
            with open(src, encoding="utf-8") as f:
                if src.endswith(".jsonl"):
                    lines = [json.dumps(convert(json.loads(line)))
                             for line in f if line.strip()]
                    text = "\n".join(lines) + ("\n" if lines else "")
                else:
                    text = json.dumps(convert(json.load(f)), indent=2) + "\n"
            if write(dest, text, a.overwrite):
                print(f"  {rel} -> {os.path.relpath(dest, a.new)}")
                n += 1
    print(f"{n} file(s) converted into {a.new}")


if __name__ == "__main__":
    main()
