"""Prüft die Fräs-Vorführung: Ringtasche, Wendelnut, Nut und Schlüsselfläche werden (fast) vollständig ausgeräumt."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s3d_core as c

DEMO = os.path.join(os.path.dirname(HERE), "demo")
fehler = []


def lade(name):
    m = c.parse_s3d(os.path.join(DEMO, name))
    w = c.WCS()
    w.set_lathe(m)
    return m, w


def pruefe(name, st, mind):
    st.cut_to(len(st.poses) - 1)
    p = st.progress_text()
    print(f"{name:16s} Posen {len(st.poses):6d}  ausgeräumt {p:6.2f} %")
    if p < mind:
        fehler.append(f"{name} nur {p:.1f} %")


m, w = lade("Demo_Stirntasche.s3d")
prof = c.ring_pockets(m)[0]
path = c.ring_pocket_path(prof, 3.0, 1.0, 3.6, 2.0, 0, 0, 2.0, "wendel", r_start=17)
pruefe("Ringtasche", c.RingStock(m, w, prof, path, 3.0), 90)
m, w = lade("Demo_Wendelnut.s3d")
faces = [i for i, f in enumerate(m.faces) if f.surf is None or f.surf.kind not in ("CYLINDER", "PLANE")]
pruefe("Wendelnut", c.GrooveStock(m, c.groove_centerline(m, faces, step=0.2288)), 99)
m, w = lade("Demo_Welle.s3d")
for f in c.milling_features(m, w):
    st = c.PlaneStock(m, f, c.plane_floor_face(m, f))
    for D in ((4.0, 8.0, 9.6, 11.0) if f["kind"] == "Nut" else (8.0, 16.0)):
        st.plan(D / 2, 1.0, 0.6 * D)
        pruefe(f"{f['kind']} D{D:g}", st, 99.5)     # kein Steg in der Mitte
print("Abweichungen:", len(fehler), *fehler)
sys.exit(1 if fehler else 0)
