"""Build colab_gpu_steps.ipynb and countersign_bundle.zip (code + lexicon outputs).

Upload the zip to Drive MyDrive/countersign/, open the notebook in Colab with a T4
runtime, Run all.  Results persist under MyDrive/countersign/countersign_runs/ and a
rerun resumes from the trial cache.
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CELLS = [
    ("md", "# Countersign pool — {title}\n"
           "Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all.\n"
           "Everything is cached on Drive; if Colab disconnects, just Run all again."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"
             "import os, subprocess\nOKP = '/content/drive/MyDrive/countersign'\n"
             "assert os.path.exists(f'{DRIVE}/countersign_bundle.zip'), 'upload countersign_bundle.zip to MyDrive/countersign'\n"
             "!rm -rf /content/cs && mkdir -p /content/cs && cd /content/cs && unzip -q -o {DRIVE}/countersign_bundle.zip\n"
             "os.makedirs(f'{DRIVE}/countersign_runs', exist_ok=True)\n"
             "# runs/ lives on Drive: caches and results survive disconnects\n"
             "!cd /content/cs && cp -rn runs/. {DRIVE}/countersign_runs/ && rm -rf runs && ln -s {DRIVE}/countersign_runs runs\n"
             "!ls /content/cs /content/cs/runs"),
    # only what s2-s4 import: surface forms are precomputed in universe.csv, so
    # kiwipiepy / python-mecab-ko / g2pk2 are not needed here (mecab has no
    # wheel for Colab's Python and its failed build aborted the whole install)
    ("code", "!pip -q install numba pyyaml openai-whisper edge-tts gTTS librosa soundfile "
             "statsmodels funasr modelscope kaldi-native-fbank\n"
             "# stop Run all here if anything is missing\n"
             "import whisper, funasr, edge_tts, gtts, numba, yaml, librosa, soundfile, statsmodels, kaldi_native_fbank\n"
             "import torch; assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type -> T4 GPU'\n"
             "print('OK, GPU:', torch.cuda.get_device_name(0))"),
    ("code", "# trial/TTS cache on local disk (SQLite is unreliable on Drive), backed up to Drive every 5 min\n"
             "os.environ['CS_CACHE'] = '/content/cache'\n"
             "!mkdir -p /content/cache {DRIVE}/countersign_cache && cp -rn {DRIVE}/countersign_cache/. /content/cache/\n"
             "# a journal copied from Drive would be replayed as a hot journal and could roll back / corrupt the cache\n"
             "!rm -f /content/cache/trials.sqlite-journal /content/cache/trials.sqlite.tmp\n"
             "if os.path.exists('/content/cache/trials.sqlite'):\n"
             "    import sqlite3 as _s; _c = _s.connect('/content/cache/trials.sqlite')\n"
             "    print('cache integrity:', _c.execute('PRAGMA integrity_check').fetchone()[0],\n"
             "          '| cached trials:', _c.execute('SELECT count(*) FROM fc').fetchone()[0]); _c.close()\n"
             "# backup = a consistent SQLite snapshot (small) + only NEW TTS files.  Recopying every TTS\n"
             "# file each time flooded Drive's upload queue and the trial cache never reached Drive.\n"
             "import sqlite3, shutil, threading, time\n"
             "def backup():\n"
             "    if os.path.exists('/content/cache/trials.sqlite'):\n"
             "        src = sqlite3.connect('/content/cache/trials.sqlite'); dst = sqlite3.connect('/content/snap.sqlite')\n"
             "        src.backup(dst); dst.close(); src.close()\n"
             "        shutil.copyfile('/content/snap.sqlite', f'{DRIVE}/countersign_cache/trials.sqlite.tmp')\n"
             "        os.replace(f'{DRIVE}/countersign_cache/trials.sqlite.tmp', f'{DRIVE}/countersign_cache/trials.sqlite')\n"
             "    subprocess.run(['bash', '-c', f'cp -rn /content/cache/tts {DRIVE}/countersign_cache/ 2>/dev/null'])\n"
             "    with open(f'{DRIVE}/countersign_cache/last_backup.txt', 'w') as f:\n"
             "        f.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))\n"
             "def _loop():\n"
             "    while True:\n"
             "        time.sleep(300)\n"
             "        try: backup()\n"
             "        except Exception as e: print('backup failed:', e)\n"
             "threading.Thread(target=_loop, daemon=True).start()\n"
             "def step(script, *args):\n"
             "    # a failing step raises, so Run all stops instead of running the next steps on missing files;\n"
             "    # output is also appended to runs/logs/ (on Drive) so progress can be read remotely\n"
             "    os.makedirs('/content/cs/runs/logs', exist_ok=True)\n"
             "    name = os.path.basename(script)[:-3] + ''.join('_' + a.strip('-') for a in args)\n"
             "    log = open(f'/content/cs/runs/logs/{name}.log', 'a')\n"
             "    p = subprocess.Popen(['python', '-X', 'utf8', script, *args], cwd='/content/cs',\n"
             "                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ)\n"
             "    for line in p.stdout:\n"
             "        if 'it/s' not in line:\n"
             "            print(line, end='', flush=True)\n"
             "            log.write(line); log.flush()\n"
             "    p.wait()\n"
             "    log.close()\n"
             "    backup()\n"
             "    if p.returncode:\n"
             "        raise RuntimeError(f'{script} failed (exit {p.returncode}); see the traceback above')"),
]

# Drive writes are buffered: without an explicit flush the last files
# (result.json etc.) can be lost when the runtime ends
GPU_STEPS = [
    ("code", "step('scripts/s2_calibration.py')"),
    ("code", "step('scripts/s3_distances.py')"),
    ("code", "step('scripts/s4_metric_validation.py')"),
    ("code", "!ls -la /content/cs/runs/calibration /content/cs/runs/metric_validation\n"
             "backup()\n"
             "drive.flush_and_unmount()\n"
             "print('flushed to Drive: safe to close')"),
    ("md", "Done. Results are in `MyDrive/countersign/countersign_runs/` "
           "(`calibration/`, `distances/summary.json`, `metric_validation/`)."),
]

SELECTION_STEPS = [
    ("code", "step('scripts/s2_calibration.py', '--section', 'calibration_v2')"),
    ("code", "step('scripts/s8_metric_selection.py')"),
    ("code", "!ls -la /content/cs/runs/calibration_v2 /content/cs/runs/metric_selection\n"
             "backup()\n"
             "drive.flush_and_unmount()\n"
             "print('flushed to Drive: safe to close')"),
    ("md", "Done. Results are in `MyDrive/countersign/countersign_runs/` "
           "(`calibration_v2/`, `metric_selection/`, `logs/`)."),
]

NOTEBOOKS = {
    "colab_gpu_steps.ipynb": ("GPU steps (calibration → distances → metric validation)", GPU_STEPS),
    "colab_metric_selection.ipynb": ("metric selection (calibration refit → 4 distances compared, "
                                     "original channel only)", SELECTION_STEPS),
}


def notebook(name="colab_metric_selection.ipynb"):
    title, steps = NOTEBOOKS[name]
    cells = []
    for kind, src in CELLS + steps:
        src = src.replace("{title}", title)
        c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        cells.append(c)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    p = ROOT / name
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def bundle():
    p = ROOT / "countersign_bundle.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["config.yaml"] + [str(x.relative_to(ROOT)) for x in (ROOT / "cpool").glob("*.py")] \
                + [str(x.relative_to(ROOT)) for x in (ROOT / "scripts").glob("s*.py")]:
            z.write(ROOT / f, f)
        for f in ["runs/lexicon/universe.csv", "runs/lexicon/summary.json",
                  "runs/pilot_heldout/words.json", "runs/pilot_heldout/choice.json"]:
            if (ROOT / f).exists():
                z.write(ROOT / f, f)
    return p


if __name__ == "__main__":
    print(notebook(), bundle())
