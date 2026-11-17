"""Prüft Eintrittspunkt, X Ø, Z, Tiefe bis X0, B und Anschnitt für viele Indexbolzen-Varianten
(Kopf 45/30/15°, Bohrwinkel 30–90°, verschiedene Lagen). Aufruf: python tests/pruefe_eintritt.py
Benötigt numpy und manifold3d. Soll-Werte werden unabhängig vom Viewer aus der Geometrie gerechnet."""
import sys, os, math, subprocess, itertools, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s3d_core as c
SP = tempfile.mkdtemp(prefix="eintritt_")
print(f"{'Kopf':>5} {'Bohr∠':>6} {'Lage':>5} | {'X Ø soll':>9} {'ist':>8} | {'Z soll':>7} {'ist':>7} | {'X0-Tiefe soll':>13} {'ist':>8} | {'B oben':>6} {'Anschn. soll/ist':>16} | ok")
bad = 0
for FL, BETA, F in itertools.product((45, 30, 15), (30, 45, 60, 90), (0.3, 0.6)):
    fn = os.path.join(SP, f"v_{FL}_{BETA}_{F}.s3d")
    r = subprocess.run([sys.executable, os.path.join(HERE, "indexbolzen_variante.py"), fn, str(FL), str(BETA), str(F)], capture_output=True, text=True)
    if r.returncode:
        print(f"{FL:>4}° {BETA:>5}° {F:>5} | (Testteil nicht erzeugbar – Bohrung läuft aus dem Kegel)"); continue
    TPa, TPr = 5 - 5 * math.sin(math.radians(FL)), 5 * math.cos(math.radians(FL))
    RE = TPr + F * (7 - TPr)
    ze = TPa + (RE - TPr) / math.tan(math.radians(FL))
    dx0 = RE / math.sin(math.radians(BETA))
    hit = 90 - abs(BETA - (90 - FL))     # Winkel Bohrer/Fläche
    m = c.parse_s3d(fn); w = c.WCS(); w.set_lathe(m)
    hs = [h for h in c.find_holes(m) if "Schräg" in h.features[0]]
    worst = 0
    for h in hs:
        e = c.entry_info(m, h, w)
        dev = max(abs(e["dia"] - 2 * RE), abs(e["z"] - ze), abs(e["depth_x0"] - dx0), abs(e["b_top"] - (90 - BETA)),
                  abs((e["hit"] if e["hit"] is not None else -99) - hit))
        worst = max(worst, dev)
    e = c.entry_info(m, hs[0], w)
    ok = worst < 2e-3 and len(hs) == 4
    bad += not ok
    print(f"{FL:>4}° {BETA:>5}° {F:>5} | {2*RE:9.3f} {e['dia']:8.3f} | {ze:7.3f} {e['z']:7.3f} | {dx0:13.3f} {e['depth_x0']:8.3f} | "
          f"{e['b_top']:6.1f} {hit:7.1f}/{(e['hit'] if e['hit'] is not None else -1):7.1f} | {'ja' if ok else 'NEIN %.4f' % worst}")
print("Abweichungen:", bad)
sys.exit(1 if bad else 0)
