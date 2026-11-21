"""Erzeugt das neutrale Demo-Bauteil „Demo-Stirntasche“ im .s3d-Format (frei erfunden, keine echten Daten).
Buchse Ø80 mit umlaufender Ringtasche in der Stirnseite: Außenwand schräg (60° zur Achse), ebener Grund,
konische Nabe (5°), die über die Stirnseite hinausragt, durchgehende Bohrung Ø16.
Aufruf: python make_stirntasche.py <ziel.s3d>   (benötigt: numpy, manifold3d)"""
import math, sys
import numpy as np
import manifold3d as mf

OUT = sys.argv[1]
X = np.array([1.0, 0, 0])
R_OUT, R_BORE, L_ALL = 40.0, 8.0, 40.0
A_RING = 10.0                          # Stirnfläche des Außenrings (Nabe ragt bis a = 0 vor)
R_RING_IN = 35.0                       # Beginn der schrägen Außenwand
A_FLOOR = 18.0                         # Taschengrund
T60 = math.tan(math.radians(60))
R_FLOOR_OUT = R_RING_IN - (A_FLOOR - A_RING) * T60        # Außenwand am Grund
R_HUB_FLOOR = 14.0
R_HUB_FRONT = R_HUB_FLOOR - A_FLOOR * math.tan(math.radians(5))
prof = [(0.0, R_BORE), (0.0, R_HUB_FRONT), (A_FLOOR, R_HUB_FLOOR), (A_FLOOR, R_FLOOR_OUT), (A_RING, R_RING_IN),
        (A_RING, R_OUT - 1.0), (A_RING + 1.0, R_OUT), (L_ALL - 1.0, R_OUT), (L_ALL, R_OUT - 1.0), (L_ALL, R_BORE)]
body = mf.Manifold.revolve(mf.CrossSection([[(r, a) for a, r in prof]]), 256).rotate((0, 90, 0))
mesh = body.to_mesh()
V = np.asarray(mesh.vert_properties)[:, :3].astype(float)
Tr = np.asarray(mesh.tri_verts).astype(int)
P = V[Tr]
print("Dreiecke", len(Tr), "Außenwand am Grund r", round(R_FLOOR_OUT, 4), "Nabe vorne r", round(R_HUB_FRONT, 4))

faces = []
def add(**k):
    faces.append(k); return k
GREY, ORANGE, LIGHT, BLUE = (102, 102, 102), (230, 120, 20), (170, 170, 175), (70, 120, 200)
F_hubf = add(kind="PLANE", x0=0.0, rlo=R_BORE, rhi=R_HUB_FRONT, col=GREY)
F_hub = add(kind="CONE", pA=np.array([0.0, R_HUB_FRONT]), pB=np.array([A_FLOOR, R_HUB_FLOOR]), conc="0", col=BLUE)
F_floor = add(kind="PLANE", x0=A_FLOOR, rlo=R_HUB_FLOOR, rhi=R_FLOOR_OUT, col=BLUE)
F_wall = add(kind="CONE", pA=np.array([A_FLOOR, R_FLOOR_OUT]), pB=np.array([A_RING, R_RING_IN]), conc="1", col=BLUE)
F_ring = add(kind="PLANE", x0=A_RING, rlo=R_RING_IN, rhi=R_OUT - 1.0, col=GREY)
F_ch1 = add(kind="CONE", pA=np.array([A_RING, R_OUT - 1.0]), pB=np.array([A_RING + 1.0, R_OUT]), conc="0", col=ORANGE)
F_cyl = add(kind="CYL", R=R_OUT, alo=A_RING + 1.0, ahi=L_ALL - 1.0, conc="0", col=GREY)
F_ch2 = add(kind="CONE", pA=np.array([L_ALL - 1.0, R_OUT]), pB=np.array([L_ALL, R_OUT - 1.0]), conc="0", col=ORANGE)
F_back = add(kind="PLANE", x0=L_ALL, rlo=R_BORE, rhi=R_OUT - 1.0, col=GREY)
F_bore = add(kind="CYL", R=R_BORE, alo=0.0, ahi=L_ALL, conc="1", col=LIGHT)

def rev(P):
    return P[..., 0], np.hypot(P[..., 1], P[..., 2])

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
    if k == "SPH":
        if a.min() < f["alo"] - eps or a.max() > f["ahi"] + eps: return np.inf
        return np.abs(np.linalg.norm(P - f["c"], axis=1) - f["r"]).max()
    if k == "LCYL":
        q = P - f["o"]; t = q @ f["d"]
        if t.min() < f["tlo"] - eps or t.max() > f["thi"] + eps: return np.inf
        return np.abs(np.linalg.norm(q - np.outer(t, f["d"]), axis=1) - f["R"]).max()

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
for (a, r), fa, fb in [((0.0, R_BORE), F_hubf, F_bore), ((0.0, R_HUB_FRONT), F_hubf, F_hub),
                      ((A_FLOOR, R_HUB_FLOOR), F_hub, F_floor), ((A_FLOOR, R_FLOOR_OUT), F_floor, F_wall),
                      ((A_RING, R_RING_IN), F_wall, F_ring), ((A_RING, R_OUT - 1.0), F_ring, F_ch1),
                      ((A_RING + 1.0, R_OUT), F_ch1, F_cyl), ((L_ALL - 1.0, R_OUT), F_cyl, F_ch2),
                      ((L_ALL, R_OUT - 1.0), F_ch2, F_back), ((L_ALL, R_BORE), F_back, F_bore)]:
    link(circle_edge((a, 0, 0), X, r), fa, fb)
    circle_pairs.add(frozenset((fidx(fa), fidx(fb))))
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
for f in faces:
    f.setdefault("axis", "Axis-1" if f["kind"] in ("CYL", "CONE") else "")
features = [dict(id="Feature-1", name="Drehkontur", type="REVOLVE", params={}),
            dict(id="Feature-2", name="Ringtasche Stirnseite", type="REVOLVE", params={}),
            dict(id="Feature-3", name="Bohrung Ø16", type="SIMPLE HOLE", params={})]
for f in (F_hub, F_floor, F_wall):
    f["feat"] = "Feature-2"
F_bore["feat"] = "Feature-3"
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
w('<!-- Demo-Stirntasche für 3D Viewer – frei erfunden, frei verwendbar, keine echten Konstruktionsdaten -->')
w('<classmate-cad version="2.0">')
w('  <identifier>DEMO-STIRNTASCHE-001</identifier>\n  <system>3D Viewer Demo</system>\n  <date>2026.11.17 12:00:00</date>')
w('  <configurations><configuration><type>geometry</type><identifier>demo</identifier><views><view>'
  '<name>CLASSMATE</name><type>CLASSMATE</type>'
  '<transform>[(1.0,0.0,0.0,0.0);(0.0,-0.889,-0.458,0.0);(0.0,0.458,-0.889,0.0);(0.0,0.0,0.0,1.0)]</transform>'
  '</view></views></configuration></configurations>')
w('  <models>\n    <model>\n      <id>Model-1</id>\n      <type>part</type>\n      <name>Demo-Stirntasche</name>\n      <units>mm</units>')
w(f'      <length>{hi[0]-lo[0]:.4f}</length>\n      <width>{hi[1]-lo[1]:.4f}</width>\n      <height>{hi[2]-lo[2]:.4f}</height>')
w(f'      <area>{area.sum():.3f}</area>\n      <volume>{vol:.3f}</volume>\n      <mass>{vol*7.85e-6:.4f}</mass>')
w(f'      <center>{v3((lo+hi)/2)}</center>\n      <main-axis>Axis-1</main-axis>')
w('      <parameters>')
for n, v in [("DB_PART_NAME", "Demo-Stirntasche"), ("Material", "C45 (1.0503)"),
             ("Beschreibung", "Demo-Stirntasche: Ringtasche mit Schrägen in der Stirnseite, Bohrung Ø16"), ("Ersteller", "3D Viewer")]:
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
        "HCYL": "CYLINDER", "HCONE": "CONE", "LCYL": "CYLINDER", "SPH": "SPHERE"}
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
    if f["kind"] == "HCONE":
        w(f'          <radius1>{f["R"]:.6g}</radius1><radius2>0</radius2><angle>59</angle>')
    if f["kind"] == "SPH":
        w(f'          <radius1>{f["r"]:.6g}</radius1><radius2>{f["r"]:.6g}</radius2>')
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
