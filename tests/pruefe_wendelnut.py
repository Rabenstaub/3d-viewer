"""Prüft den Wendelnut-Generator am Demo-Teil: Mittellinie gegen die Soll-Kurve aus make_wendelnut.py,
Summen W/H und Satzanzahl des EIA-Unterprogramms."""
import sys, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s3d_core as c

m = c.parse_s3d(os.path.join(os.path.dirname(HERE), "demo", "Demo_Wendelnut.s3d"))
faces = [i for i, f in enumerate(m.faces) if f.surf is None or f.surf.kind not in ("CYLINDER", "PLANE")]
cl = c.groove_centerline(m, faces, step=0.2288)
wcs = c.WCS()
wcs.set_lathe(m)
txt, inf = c.groove_eia(m, wcs, cl)
saetze = [z for z in txt.splitlines() if z.startswith("W")]
fehler = []
if not (5.8 < cl["width"] < 6.1):
    fehler.append(f"Breite {cl['width']:.3f}")
if abs(abs(cl["c"][-1] - cl["c"][0]) - 60) > 0.05:
    fehler.append(f"Drehung {cl['c'][-1] - cl['c'][0]:.3f}")
if abs(abs(cl["a"][-1] - cl["a"][0]) - 95) > 0.2:
    fehler.append(f"Länge {cl['a'][-1] - cl['a'][0]:.3f}")
sw = sum(float(z.split()[0][1:]) for z in saetze)
sh = sum(float(z.split()[1][1:]) for z in saetze)
if abs(sw - inf["W"]) > 1e-5 or abs(sh - inf["H"]) > 1e-5:
    fehler.append(f"Summen W {sw} / H {sh}")
if not txt.rstrip().endswith("M99") or "\r\n" not in txt:
    fehler.append("Ende/CRLF")
print(f"Sätze {len(saetze)}  Summe W {sw:.6f}  H {sh:.6f}")
print(f"Punkte {len(cl['a'])}  Breite {cl['width']:.3f}  Z {cl['a'][0]:.3f}..{cl['a'][-1]:.3f}  "
      f"C {cl['c'][0]:.2f}..{cl['c'][-1]:.2f}")
print("Abweichungen:", len(fehler), *fehler)
sys.exit(1 if fehler else 0)
