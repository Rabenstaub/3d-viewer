"""Erzeugt das neutrale Demo-Bauteil „Demo-Welle“ im .s3d-Format (frei erfunden, keine echten Daten).
Drehkontur mit Fasen, Hohlkehle, Rundung, 30°-Kegel, Bohrung Ø20, 6× M6 auf LK Ø36, Querbohrung Ø8,
Einstich 3×1,5 am Ø60, M6 mit Senkung 90°, außermittige Bohrung Ø6 (Y8),
Passfedernut 12×5, Schlüsselfläche SW50 und zwei drei Schrägbohrungen Ø5 unter 45° von der Stirnseite.
Aufruf: python make_demo.py <ziel.s3d>   (benötigt: numpy, manifold3d)"""
import math, sys
import numpy as np
import manifold3d as mf

OUT = sys.argv[1]
T30 = math.tan(math.radians(30))
X = np.array([1.0, 0, 0])

# ---------------- Drehkontur (a = x entlang Achse, r) ----------------
c_conc = np.array([38.0, 22.0]); R1 = 2.0                     # Hohlkehle R2
dd = 2 * math.tan(math.radians(15))                          # Rundung R2 zwischen Ø60 und 30°-Kegel
T1 = np.array([79 - dd, 30.0]); T2 = np.array([79 + dd * math.cos(math.radians(30)), 30 - dd * 0.5])
c_conv = np.array([T1[0], 28.0])
r_end = 30 - 6 * T30
prof = [(0, 10), (0, 19), (1, 20), (38, 20)]
arc1 = [(c_conc[0] + R1 * math.sin(t), c_conc[1] - R1 * math.cos(t)) for t in np.linspace(0, math.pi / 2, 33)]
prof += arc1[1:]
GR_A0, GR_A1, GR_R = 43.0, 46.0, 28.5                      # Einstich 3 breit, 1,5 tief
prof += [(40, 29), (41, 30), (GR_A0, 30), (GR_A0, GR_R), (GR_A1, GR_R), (GR_A1, 30), (T1[0], 30)]
a0 = math.pi / 2; a1 = math.atan2(T2[1] - c_conv[1], T2[0] - c_conv[0])
prof += [(c_conv[0] + R1 * math.cos(t), c_conv[1] + R1 * math.sin(t)) for t in np.linspace(a0, a1, 33)][1:]
prof += [(85, r_end), (85, 10)]
body = mf.Manifold.revolve(mf.CrossSection([[(r, a) for a, r in prof]]), 256).rotate((0, 90, 0))

def radial(theta_deg):
    """Radiale Richtung: 0° = +Z (oben), positiv Richtung +Y."""
    t = math.radians(theta_deg)
    return np.array([0.0, math.sin(t), math.cos(t)])

def cyl_along(p0, d, r, length, segs=96):
    """Zylinder Radius r von p0 in Richtung d (Länge length)."""
    d = d / np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); c = float(z @ d)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    M = np.hstack([R, np.asarray(p0, float).reshape(3, 1)])
    return mf.Manifold.cylinder(length, r, r, segs).transform(M)

# ---------------- Bohrungen und Fräskonturen ----------------
LK = 18.0; rh = 2.5; hb = 70.0
SK_D = 0.7                                                # Senkung 90° auf Ø6,4
tip = rh / math.tan(math.radians(59))
cut = None
thread_axes = []
for k in range(6):
    o = np.array([0.0, *(LK * radial(60 * k)[1:])])
    thread_axes.append(o)
    h = cyl_along(o + hb * X, X, rh, 16) + mf.Manifold.cylinder(tip, 0, rh, 96).rotate((0, 90, 0)).translate((hb - tip, o[1], o[2]))
    h = h + mf.Manifold.cylinder(SK_D + 1.0, rh, rh + SK_D + 1.0, 96).rotate((0, 90, 0)).translate((85 - SK_D, o[1], o[2]))
    cut = h if cut is None else cut + h
cut = cut + cyl_along(np.array([60.0, 0, 0]), np.array([0, 0, 1.0]), 4.0, 35, 128)            # Querbohrung Ø8
# Passfedernut 12x5 (DIN 6885) oben auf Ø40, x 6..34, Grund z=15
KW_B, KW_Z, KW_X0, KW_X1 = 6.0, 15.0, 12.0, 28.0
slot = mf.Manifold.hull(cyl_along(np.array([KW_X0, 0, KW_Z]), np.array([0, 0, 1.0]), KW_B, 10, 128)
                        + cyl_along(np.array([KW_X1, 0, KW_Z]), np.array([0, 0, 1.0]), KW_B, 10, 128))
cut = cut + slot
# Schlüsselfläche SW50 am Ø60, x 48..72
FL_Y, FL_X0, FL_X1 = 25.0, 48.0, 72.0
for sg in (1, -1):
    box = mf.Manifold.cube((FL_X1 - FL_X0, 20, 80)).translate((FL_X0, FL_Y if sg > 0 else -FL_Y - 20, -40))
    cut = cut + box
# Schrägbohrungen Ø6 unter 45°: Stirnfläche am Ø40-Absatz mittig (Ø29), schräg nach innen in die Bohrung Ø20
INC_R, INC_RE = 2.5, 14.5
inc = []
for th in (60.0, 180.0, 300.0):
    u = radial(th)
    p_entry = INC_RE * u
    d = (X - u) / math.sqrt(2.0)
    inc.append((p_entry, d, th))
    cut = cut + cyl_along(p_entry - 6.0 * d, d, INC_R, 6.0 + 14.0, 128)
# außermittige Bohrung Ø6 von unten (C180°), Y-Versatz 8, Sackloch mit 118°-Spitze
OF_P = np.array([75.0, 8.0, 0.0]); OF_R = 3.0
OF_ENTRY = -math.sqrt(30.0 ** 2 - 8.0 ** 2)
OF_BOT = OF_ENTRY + 10.0
OF_TIP = OF_R / math.tan(math.radians(59))
cut = cut + cyl_along(np.array([75.0, 8.0, -35.0]), np.array([0, 0, 1.0]), OF_R, OF_BOT + 35.0, 96) + \
    mf.Manifold.cylinder(OF_TIP, OF_R, 0, 96).translate((75.0, 8.0, OF_BOT))
body = body - cut
mesh = body.to_mesh()
V = np.asarray(mesh.vert_properties)[:, :3].astype(float)
Tr = np.asarray(mesh.tri_verts).astype(int)
P = V[Tr]
print("Dreiecke", len(Tr))

# ---------------- Flächen-Definitionen ----------------
faces = []
def add(**k):
    faces.append(k); return k
GREY, ORANGE, LIGHT, BLUE = (102, 102, 102), (230, 120, 20), (170, 170, 175), (70, 120, 200)
def rev(P):
    return P[..., 0], np.hypot(P[..., 1], P[..., 2])
F_front = add(kind="PLANE", x0=0, rlo=10, rhi=19, col=GREY)
F_ch1 = add(kind="CONE", pA=np.array([0, 19.0]), pB=np.array([1, 20.0]), conc="0", col=ORANGE)
F_c20 = add(kind="CYL", R=20, alo=1, ahi=38, conc="0", col=GREY)
F_hk = add(kind="TORUS", c=c_conc, r=R1, alo=38, ahi=40, conc="1", col=ORANGE)
F_sh = add(kind="PLANE", x0=40, rlo=22, rhi=29, col=GREY)
F_ch2 = add(kind="CONE", pA=np.array([40, 29.0]), pB=np.array([41, 30.0]), conc="0", col=ORANGE)
F_c30 = add(kind="CYL", R=30, alo=41, ahi=GR_A0, conc="0", col=GREY)
F_g0 = add(kind="PLANE", x0=GR_A0, rlo=GR_R, rhi=30, col=BLUE)
F_gb = add(kind="CYL", R=GR_R, alo=GR_A0, ahi=GR_A1, conc="0", col=BLUE)
F_g1 = add(kind="PLANE", x0=GR_A1, rlo=GR_R, rhi=30, col=BLUE)
F_c30b = add(kind="CYL", R=30, alo=GR_A1, ahi=T1[0], conc="0", col=GREY)
F_rd = add(kind="TORUS", c=c_conv, r=R1, alo=T1[0], ahi=T2[0], conc="0", col=ORANGE)
F_k30 = add(kind="CONE", pA=T2.copy(), pB=np.array([85, r_end]), conc="0", col=GREY)
F_back = add(kind="PLANE", x0=85, rlo=10, rhi=r_end, col=GREY)
F_bore = add(kind="CYL", R=10, alo=0, ahi=85, conc="1", col=LIGHT)
hole_faces = []
for o in thread_axes:
    hc = add(kind="HCYL", o=o, R=rh, alo=hb, ahi=85 - SK_D, col=ORANGE, conc="1")
    ht = add(kind="HCONE", o=o, R=rh, alo=hb - tip, ahi=hb, col=ORANGE, conc="1")
    hs = add(kind="HSINK", o=o, R=rh, alo=85 - SK_D, ahi=85, col=ORANGE, conc="1")
    hole_faces.append((hc, ht, o, hs))
F_of = add(kind="LCYL", o=OF_P, d=np.array([0, 0, 1.0]), R=OF_R, tlo=OF_ENTRY - 1e-3 - 0.0 - 1.0, thi=OF_BOT, col=ORANGE, conc="1")
F_oft = add(kind="LTIP", o=OF_P, d=np.array([0, 0, 1.0]), R=OF_R, tlo=OF_BOT, thi=OF_BOT + OF_TIP, col=ORANGE, conc="1")
F_rad = add(kind="LCYL", o=np.array([60.0, 0, 0]), d=np.array([0, 0, 1.0]), R=4.0, tlo=0, thi=35, col=ORANGE, conc="1")
# Passfedernut
F_kw_s = [add(kind="PLANE_N", n=np.array([0, sg, 0.0]), off=KW_B, box=((KW_X0, KW_X1), None, (KW_Z, 21)), col=BLUE,
              conc="1") for sg in (1, -1)]
F_kw_b = add(kind="PLANE_N", n=np.array([0, 0, 1.0]), off=KW_Z, box=((KW_X0 - KW_B, KW_X1 + KW_B), (-KW_B, KW_B), None),
             col=BLUE, conc="1")
F_kw_e = [add(kind="LCYL", o=np.array([xc, 0, KW_Z]), d=np.array([0, 0, 1.0]), R=KW_B, tlo=0, thi=6, col=BLUE, conc="1",
              side=sg) for xc, sg in ((KW_X0, -1), (KW_X1, 1))]
# Schlüsselfläche
F_fl = [add(kind="PLANE_N", n=np.array([0, sg, 0.0]), off=FL_Y, box=((FL_X0, FL_X1), None, None), col=BLUE, conc="0")
        for sg in (1, -1)]
F_flw = [add(kind="PLANE_N", n=np.array([-1.0 if xw == FL_X0 else 1.0, 0, 0]), off=(-xw if xw == FL_X0 else xw),
             box=(None, (FL_Y, 31) if sg > 0 else (-31, -FL_Y), None), col=BLUE, conc="1")
         for xw in (FL_X0, FL_X1) for sg in (1, -1)]
# Schrägbohrungen
F_inc = [add(kind="LCYL", o=pe, d=d, R=INC_R, tlo=-6, thi=14, col=ORANGE, conc="1") for pe, d, th in inc]

def residual(f, P):
    a, r = rev(P)
    k = f["kind"]; eps = 1e-3
    if k == "PLANE":
        if (r.min() < f["rlo"] - eps) or (r.max() > f["rhi"] + eps): return np.inf
        return np.abs(a - f["x0"]).max()
    if k == "CYL":
        if a.min() < f["alo"] - eps or a.max() > f["ahi"] + eps: return np.inf
        return np.abs(r - f["R"]).max()
    if k == "CONE":
        pA, pB = f["pA"], f["pB"]; t = (pB - pA) / np.linalg.norm(pB - pA); n = np.array([-t[1], t[0]])
        s = (a - pA[0]) * t[0] + (r - pA[1]) * t[1]
        if s.min() < -eps or s.max() > np.linalg.norm(pB - pA) + eps: return np.inf
        return np.abs((a - pA[0]) * n[0] + (r - pA[1]) * n[1]).max()
    if k == "TORUS":
        if a.min() < f["alo"] - eps or a.max() > f["ahi"] + eps: return np.inf
        return np.abs(np.hypot(a - f["c"][0], r - f["c"][1]) - f["r"]).max()
    if k in ("HCYL", "HCONE"):
        q = P - f["o"]; ah = q[:, 0]; rr = np.hypot(q[:, 1], q[:, 2])
        if ah.min() < f["alo"] - eps or ah.max() > f["ahi"] + eps: return np.inf
        if k == "HCYL": return np.abs(rr - f["R"]).max()
        return np.abs(rr - (ah - f["alo"]) / (f["ahi"] - f["alo"]) * f["R"]).max()
    if k == "LCYL":
        q = P - f["o"]; t = q @ f["d"]
        if t.min() < f["tlo"] - eps or t.max() > f["thi"] + eps: return np.inf
        if "side" in f and (np.sign(q[:, 0] + f["side"] * 1e-6) != f["side"]).any(): return np.inf
        return np.abs(np.linalg.norm(q - np.outer(t, f["d"]), axis=1) - f["R"]).max()
    if k == "HSINK":
        q = P - f["o"]; ah = q[:, 0]; rr = np.hypot(q[:, 1], q[:, 2])
        if ah.min() < f["alo"] - eps or ah.max() > f["ahi"] + eps: return np.inf
        return np.abs(rr - (f["R"] + (ah - f["alo"]))).max() / math.sqrt(2)
    if k == "LTIP":
        q = P - f["o"]; t = q @ f["d"]
        if t.min() < f["tlo"] - eps or t.max() > f["thi"] + eps: return np.inf
        rr = np.linalg.norm(q - np.outer(t, f["d"]), axis=1)
        return np.abs(rr - f["R"] * (f["thi"] - t) / (f["thi"] - f["tlo"])).max()
    if k == "PLANE_N":
        for ax, rng in enumerate(f["box"]):
            if rng is not None and (P[:, ax].min() < rng[0] - eps or P[:, ax].max() > rng[1] + eps): return np.inf
        return np.abs(P @ f["n"] - f["off"]).max()

assign = np.full(len(P), -1); best = np.full(len(P), np.inf)
for fi, f in enumerate(faces):
    res = np.array([residual(f, p) for p in P])
    better = res < best
    assign[better] = fi; best[better] = res[better]
bad = best > 0.02
print("nicht zugeordnet:", bad.sum(), "max Rest", best[~bad].max())
assert bad.sum() == 0, (bad.sum(), P[bad][:5])
for fi, f in enumerate(faces):
    assert (assign == fi).any(), ("Fläche ohne Dreiecke", fi, f["kind"])

# ---------------- Kanten ----------------
edges = []
def circle_edge(center, normal, R):
    center = np.array(center, float); normal = np.array(normal, float)
    perp = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0, 1.0, 0])
    perp = perp - (perp @ normal) * normal; perp /= np.linalg.norm(perp)
    e = dict(id=f"Edge-{len(edges)+1}", type="CIRCLE", length=2 * math.pi * R, radius=R, start=center + R * perp,
             tan=np.cross(normal, perp), center=center, normal=normal)
    edges.append(e); return e
def link(e, *fs):
    for f in fs:
        f.setdefault("edges", []).append(e["id"])
def fidx(f):
    return next(i for i, g in enumerate(faces) if g is f)
circle_pairs = set()
for (a, r), fa, fb in [((0, 10), F_front, F_bore), ((0, 19), F_front, F_ch1), ((1, 20), F_ch1, F_c20),
                      ((38, 20), F_c20, F_hk), ((40, 22), F_hk, F_sh), ((40, 29), F_sh, F_ch2),
                      ((41, 30), F_ch2, F_c30), ((GR_A0, 30), F_c30, F_g0), ((GR_A0, GR_R), F_g0, F_gb), ((GR_A1, GR_R), F_gb, F_g1),
                      ((GR_A1, 30), F_g1, F_c30b), (tuple(T1), F_c30b, F_rd), (tuple(T2), F_rd, F_k30),
                      ((85, r_end), F_k30, F_back), ((85, 10), F_back, F_bore)]:
    link(circle_edge((a, 0, 0), X, r), fa, fb)
    circle_pairs.add(frozenset((fidx(fa), fidx(fb))))
thread_edges = []
for hc, ht, o, hs in hole_faces:
    link(circle_edge(o + 85 * X, X, rh + SK_D), F_back, hs)
    e_in = circle_edge(o + (85 - SK_D) * X, X, rh); link(e_in, hs, hc); thread_edges.append(e_in)
    link(circle_edge(o + hb * X, X, rh), hc, ht)
    circle_pairs.add(frozenset((fidx(F_back), fidx(hs))))
    circle_pairs.add(frozenset((fidx(hs), fidx(hc))))
    circle_pairs.add(frozenset((fidx(hc), fidx(ht))))
# alle übrigen Flächengrenzen (Nut, Schlüsselfläche, Bohrungen) als Freiform-Kanten aus dem Netz
E = np.sort(np.stack([Tr[:, [0, 1]], Tr[:, [1, 2]], Tr[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
ef = np.repeat(assign, 3)
key = E[:, 0] * len(V) + E[:, 1]
order = np.argsort(key)
ks, fs_ = key[order], ef[order]
pairs = {}
for i in range(len(ks) - 1):
    if ks[i] == ks[i + 1] and fs_[i] != fs_[i + 1]:
        pr = frozenset((int(fs_[i]), int(fs_[i + 1])))
        if pr in circle_pairs:
            continue
        a_, b_ = int(ks[i] // len(V)), int(ks[i] % len(V))
        pairs.setdefault(pr, []).append((a_, b_))
for pr, segs in pairs.items():
    adj = {}
    for a_, b_ in segs:
        adj.setdefault(a_, []).append(b_); adj.setdefault(b_, []).append(a_)
    seen = set()
    for v0 in adj:
        if v0 in seen:
            continue
        comp, st = [], [v0]; seen.add(v0)
        while st:
            v = st.pop(); comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w); st.append(w)
        L = sum(np.linalg.norm(V[a_] - V[b_]) for a_, b_ in segs if a_ in set(comp))
        e = dict(id=f"Edge-{len(edges)+1}", type="FREEFORM", length=L, start=V[comp[0]], tan=X)
        edges.append(e)
        for fi in pr:
            faces[fi].setdefault("edges", []).append(e["id"])

# ---------------- IDs, Achsen, Features ----------------
for i, f in enumerate(faces):
    f["id"] = f"Face-{i+1}"
axes = [("Axis-1", (0, 0, 0), (1, 0, 0))]
def new_axis(o, d):
    aid = f"Axis-{len(axes)+1}"; axes.append((aid, tuple(o), tuple(d))); return aid
for hc, ht, o, hs in hole_faces:
    hc["axis"] = ht["axis"] = hs["axis"] = new_axis(o, (1, 0, 0))
F_of["axis"] = F_oft["axis"] = new_axis(OF_P, (0, 0, 1))
F_rad["axis"] = new_axis((60, 0, 0), (0, 0, 1))
for f in F_kw_e:
    f["axis"] = new_axis(f["o"], (0, 0, 1))
for f in F_inc:
    f["axis"] = new_axis(f["o"], f["d"])
for f in faces:
    f.setdefault("axis", "Axis-1" if f["kind"] in ("CYL", "CONE", "TORUS") else "")
features = [dict(id="Feature-1", name="Drehkontur", type="REVOLVE", params={})]
def feat(name, typ, params=None):
    fid = f"Feature-{len(features)+1}"
    features.append(dict(id=fid, name=name, type=typ, params=params or {})); return fid
F_rad["feat"] = feat("Querbohrung Ø8", "SIMPLE HOLE")
F_of["feat"] = F_oft["feat"] = feat("Bohrung Ø6 außermittig (Y8)", "SIMPLE HOLE")
gr = feat("Einstich 3x1,5", "REVOLVE")
for f in (F_g0, F_gb, F_g1):
    f["feat"] = gr
kw = feat("Passfedernut 12x5 (DIN 6885)", "POCKET")
for f in F_kw_s + [F_kw_b] + F_kw_e:
    f["feat"] = kw
fl = feat("Schlüsselfläche SW50", "POCKET")
for f in F_fl + F_flw:
    f["feat"] = fl
for i, f in enumerate(F_inc):
    f["feat"] = feat(f"Schrägbohrung Ø5 45° ({i+1})", "SIMPLE HOLE")
for k, (hc, ht, o, hs) in enumerate(hole_faces):
    fid = feat(f"Gewindebohrung M6 ({k+1})", "SIMPLE HOLE")
    hc["feat"] = ht["feat"] = hs["feat"] = fid
    feat(f"M6x1 Gewinde ({k+1})", "SYMBOLIC_THREAD",
         {"thread-description": "M6x1", "threaded-face": hc["id"], "start-edge": thread_edges[k]["id"],
          "thread-depth": "12", "thread-pitch": "1", "tapped-diameter": "5", "thread-angle": "60",
          "thread-form": "Metric", "thread-rotation": "right", "num-starts": "1", "minor-diameter": "4.917",
          "thread-method": "Cut", "is-tapered": "0", "length-flag": "fixed length"})
for f in faces:
    if not f.get("feat"):
        f["feat"] = "Feature-1"

def v3(p): return "(" + ", ".join(f"{x:.6g}" for x in p) + ")"
def tri_text(Tri):
    return ", ".join(" ".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in p) for p in Tri.reshape(-1, 3))
area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
vol = abs(np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6)
lo, hi = V.min(0), V.max(0)
L = []; w = L.append
w('<?xml version="1.0" encoding="utf-8"?>')
w('<!-- Demo-Bauteil für 3D Viewer – frei erfunden, frei verwendbar, keine echten Konstruktionsdaten -->')
w('<classmate-cad version="2.0">')
w('  <identifier>DEMO-WELLE-003</identifier>\n  <system>3D Viewer Demo</system>\n  <date>2026.11.17 12:00:00</date>')
w('  <configurations><configuration><type>geometry</type><identifier>demo</identifier><views><view>'
  '<name>CLASSMATE</name><type>CLASSMATE</type>'
  '<transform>[(1.0,0.0,0.0,0.0);(0.0,-0.889,-0.458,0.0);(0.0,0.458,-0.889,0.0);(0.0,0.0,0.0,1.0)]</transform>'
  '</view></views></configuration></configurations>')
w('  <models>\n    <model>\n      <id>Model-1</id>\n      <type>part</type>\n      <name>Demo-Welle</name>\n      <units>mm</units>')
w(f'      <length>{hi[0]-lo[0]:.4f}</length>\n      <width>{hi[1]-lo[1]:.4f}</width>\n      <height>{hi[2]-lo[2]:.4f}</height>')
w(f'      <area>{area.sum():.3f}</area>\n      <volume>{vol:.3f}</volume>\n      <mass>{vol*7.85e-6:.4f}</mass>')
w(f'      <center>{v3((lo+hi)/2)}</center>\n      <main-axis>Axis-1</main-axis>')
w('      <parameters>')
for n, v in [("DB_PART_NAME", "Demo-Welle"), ("Material", "C45 (1.0503)"),
             ("Beschreibung", "Demo-Bauteil zum Ausprobieren"), ("Ersteller", "3D Viewer")]:
    w(f'        <parameter><type>ATTR-STRING</type><name>{n}</name><value>{v}</value></parameter>')
w('      </parameters>\n      <directions>')
for aid, o, d_ in axes:
    w(f'        <axis><id>{aid}</id><origin>{v3(o)}</origin><direction>{v3(d_)}</direction></axis>')
w('      </directions>\n      <features>')
for ft in features:
    w(f'        <feature><id>{ft["id"]}</id><name>{ft["name"]}</name><feature-type>{ft["type"]}</feature-type><parameters>')
    for k, v in ft["params"].items():
        w(f'          <feature-parameter><type>FEATURE</type><name>{k}</name><value>{v}</value></feature-parameter>')
    w('        </parameters></feature>')
w('      </features>\n      <edges>')
for e in edges:
    w(f'        <edge><id>{e["id"]}</id><edge-type>{e["type"]}</edge-type><length>{e["length"]:.6f}</length>')
    if e["type"] == "CIRCLE":
        w(f'          <radius1>{e["radius"]:.6g}</radius1>')
    w(f'          <start>{v3(e["start"])}</start><start-tangent>{v3(e["tan"])}</start-tangent>'
      f'<end>{v3(e["start"])}</end><end-tangent>{v3(e["tan"])}</end-tangent>')
    if e["type"] == "CIRCLE":
        w(f'          <center>{v3(e["center"])}</center><normal>{v3(e["normal"])}</normal>')
    w('        </edge>')
w('      </edges>\n      <faces>')
TYPE = {"PLANE": "PLANE", "PLANE_N": "PLANE", "CYL": "CYLINDER", "CONE": "CONE", "TORUS": "TORUS",
        "HCYL": "CYLINDER", "HCONE": "CONE", "LCYL": "CYLINDER", "HSINK": "CONE", "LTIP": "CONE"}
for fi, f in enumerate(faces):
    tri = P[assign == fi]
    w(f'        <face><id>{f["id"]}</id><face-type>{TYPE[f["kind"]]}</face-type>')
    if f["kind"] == "CYL":
        w(f'          <start-position>{f["alo"]:.6g}</start-position><end-position>{f["ahi"]:.6g}</end-position>'
          f'<radius1>{f["R"]:.6g}</radius1><radius2>{f["R"]:.6g}</radius2><length>{f["ahi"]-f["alo"]:.6g}</length>')
    if f["kind"] in ("HCYL", "LCYL"):
        w(f'          <radius1>{f["R"]:.6g}</radius1><radius2>{f["R"]:.6g}</radius2>')
    if f["kind"] == "CONE":
        pA, pB = f["pA"], f["pB"]
        ang = math.degrees(math.atan2(abs(pB[1] - pA[1]), abs(pB[0] - pA[0])))
        w(f'          <start-position>{pA[0]:.6g}</start-position><end-position>{pB[0]:.6g}</end-position>'
          f'<radius1>{pA[1]:.6g}</radius1><radius2>{pB[1]:.6g}</radius2><angle>{ang:.6g}</angle>')
    if f["kind"] in ("HCONE", "LTIP"):
        w(f'          <radius1>{f["R"]:.6g}</radius1><radius2>0</radius2><angle>59</angle>')
    if f["kind"] == "HSINK":
        w(f'          <radius1>{f["R"] + SK_D:.6g}</radius1><radius2>{f["R"]:.6g}</radius2><angle>45</angle>')
    if f["kind"] == "TORUS":
        w(f'          <radius1>{f["c"][1]+f["r"]:.6g}</radius1><radius2>{f["c"][1]-f["r"]:.6g}</radius2>')
    fa = 180 if "side" in f else 360
    w(f'          <face-angle>{fa}</face-angle><area>{area[assign == fi].sum():.4f}</area><concavity>{f.get("conc","0")}</concavity>')
    if f["axis"]:
        w(f'          <face-axis>{f["axis"]}</face-axis>')
    w(f'          <face-feature>{f["feat"]}</face-feature>')
    w(f"          <face-tesselation count='{len(tri)}'><triangles>{tri_text(tri)}</triangles></face-tesselation>")
    c = f["col"]
    w(f'          <parameters><face-parameter><type>CLASSMATE</type><name>DIFFUSECOLOR</name><value>{c[0]};{c[1]};{c[2]}</value></face-parameter></parameters>')
    w('          <contours><contour><external>1</external>')
    for eid in f.get("edges", []):
        w(f'            <contour-item><reverse>0</reverse><edge-ref>{eid}</edge-ref></contour-item>')
    w('          </contour></contours>\n        </face>')
w('      </faces>\n    </model>\n  </models>\n</classmate-cad>')
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("geschrieben", OUT, len(faces), "Flächen", len(edges), "Kanten", len(features), "Features")
