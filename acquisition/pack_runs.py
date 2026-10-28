"""Packs the runs WITHOUT the images: only the text data needed for the analysis (a few hundred KB per run).

Keeps, for every run folder: P*.jsonl, ground_truth.json, obstacles.json (if any), raw/*.jsonl and raw/meta_*.json.
Leaves out frames/ (the big part), sync/ and rounds_live/.

    python -m acquisition.pack_runs survey_rounds/ runs_light.zip
    python -m acquisition.pack_runs survey_rounds/ runs_light.zip --only L4_ L5_ L6_     # only the runs whose folder starts with ...
"""
import argparse
import os
import zipfile

KEEP_ROOT = (".jsonl", ".json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("survey_dir")
    ap.add_argument("out_zip")
    ap.add_argument("--only", nargs="*", default=None, help="prefixes of the run folders to include")
    a = ap.parse_args()
    runs = sorted(d for d in os.listdir(a.survey_dir)
                  if os.path.isdir(os.path.join(a.survey_dir, d)) and "_" in d
                  and (not a.only or any(d.startswith(p) for p in a.only)))
    n_files = 0
    with zipfile.ZipFile(a.out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for run in runs:
            base = os.path.join(a.survey_dir, run)
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if os.path.isfile(p) and name.endswith(KEEP_ROOT):
                    z.write(p, os.path.join(run, name)); n_files += 1
            raw = os.path.join(base, "raw")
            if os.path.isdir(raw):
                for name in sorted(os.listdir(raw)):
                    if name.endswith(KEEP_ROOT):
                        z.write(os.path.join(raw, name), os.path.join(run, "raw", name)); n_files += 1
    size = os.path.getsize(a.out_zip) / 1e6
    print(f"{len(runs)} run(s), {n_files} files -> {a.out_zip} ({size:.1f} MB)")


if __name__ == "__main__":
    main()