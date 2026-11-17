"""TESTVARIANTE (Aufruf: ziel.s3d Kopfwinkel Bohrwinkel Lage 0..1) – erzeugt das neutrale Demo-Bauteil „Demo-Indexbolzen“ im .s3d-Format (frei erfunden, keine echten Daten).
Spitze R5, die tangential in einen 30°-Kegel übergeht, danach 15°-Kegel bis Ø20, Schaft Ø20,
Bohrung Ø5 von hinten, 4× Ø3 unter 45° (je 90° versetzt) vom 30°-Kegel nach innen in die Bohrung Ø5.
Winkel = Flankenwinkel zur Längsachse.
Aufruf: python make_indexbolzen.py <ziel.s3d>   (benötigt: numpy, manifold3d)"""
import math, sys
import numpy as np
import manifold3d as mf

OUT = sys.argv[1]
FL = float(sys.argv[2]); BETA = float(sys.argv[3]); RE_F = float(sys.argv[4])
X = np.array([1.0, 0, 0])
t30, t15 = math.tan(math.radians(FL)), math.tan(math.radians(15))
RS = 5.0                                              # Spitzenradius R5, Mittelpunkt auf der Achse
CS = np.array([RS, 0.0])
TP = CS + RS * np.array([-math.sin(math.radians(FL)), math.cos(math.radians(FL))])   # Übergang R5 -> 30°
A1 = TP[0] + (7.0 - TP[1]) / t30                      # 30° bis Ø14
A2 = A1 + (10.0 - 7.0) / t15                          # 15° bis Ø20
L = 60.0
BO_R, BO_A = 2.5, 7.0                                 # Bohrung Ø5 von hinten bis x=7 (+ 118°-Spitze)
BO_TIP = BO_R / math.tan(math.radians(59))
prof = [(0.0, 0.0)]
th0 = math.pi; th1 = math.atan2(TP[1] - CS[1], TP[0] - CS[0])
prof += [(CS[0] + RS * math.cos(t), CS[1] + RS * math.sin(t)) for t in np.linspace(th0, th1, 49)][1:]
prof += [(A1, 7.0), (A2, 10.0), (L - 1, 10.0), (L, 9.0), (L, 0.0)]
body = mf.Manifold.revolve(mf.CrossSection([[(r, a) for a, r in prof]]), 256).rotate((0, 90, 0))

def radial(theta_deg):
    t = math.radians(theta_deg)
    return np.array([0.0, math.sin(t), math.cos(t)])

def cyl_along(p0, d, r, length, segs=96):
    d = d / np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); c = float(z @ d)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    return mf.Manifold.cylinder(length, r, r, segs).transform(np.hstack([R, np.asarray(p0, float).reshape(3, 1)]))

bore = cyl_along(np.array([BO_A, 0, 0]), X, BO_R, L - BO_A + 1, 128) + \
    mf.Manifold.cylinder(BO_TIP, 0, BO_R, 128).rotate((0, 90, 0)).translate((BO_A - BO_TIP, 0, 0))
cut = bore
HR = 1.5
RE = TP[1] + RE_F * (7.0 - TP[1])                                     # Ø3, Eintritt auf dem 30°-Kegel bei Ø11,2
AE = TP[0] + (RE - TP[1]) / t30
inc = []
for th in (0.0, 90.0, 180.0, 270.0):
    u = radial(th)
    pe = AE * X + RE * u
    d = math.cos(math.radians(BETA)) * X - math.sin(math.radians(BETA)) * u
    inc.append((pe, d, th))
    cut = cut + cyl_along(pe - 4.0 * d, d, HR, 4.0 + RE / math.sin(math.radians(BETA)) - 0.3, 96)
body = body - cut
mesh = body.to_mesh()
V = np.asarray(mesh.vert_properties)[:, :3].astype(float)
Tr = np.asarray(mesh.tri_verts).astype(int)
P = V[Tr]
print("Dreiecke", len(Tr), "Eintritt x", round(AE, 4), "Ø", 2 * RE)

faces = []
def add(**k):
    faces.append(k); return k
GREY, ORANGE, LIGHT, BLUE = (102, 102, 102), (230, 120, 20), (170, 170, 175), (70, 120, 200)
F_sph = add(kind="SPH", c=np.array([RS, 0, 0]), r=RS, alo=0, ahi=TP[0], conc="0", col=ORANGE)
F_k30 = add(kind="CONE", pA=TP.copy(), pB=np.array([A1, 7.0]), conc="0", col=GREY)
F_k15 = add(kind="CONE", pA=np.array([A1, 7.0]), pB=np.array([A2, 10.0]), conc="0", col=GREY)
F_c20 = add(kind="CYL", R=10, alo=A2, ahi=L - 1, conc="0", col=GREY)
F_ch = add(kind="CONE", pA=np.array([L - 1, 10.0]), pB=np.array([L, 9.0]), conc="0", col=ORANGE)
F_back = add(kind="PLANE", x0=L, rlo=BO_R, rhi=9, col=GREY)
F_bore = add(kind="CYL", R=BO_R, alo=BO_A, ahi=L, conc="1", col=LIGHT)
F_btip = add(kind="CONE", pA=np.array([BO_A - BO_TIP, 0.0]), pB=np.array([BO_A, BO_R]), conc="1", col=LIGHT)
F_inc = [add(kind="LCYL", o=pe, d=d, R=HR, tlo=-4, thi=RE / math.sin(math.radians(BETA)), col=ORANGE, conc="1") for pe, d, th in inc]

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
for (a, r), fa, fb in [(tuple(TP), F_sph, F_k30), ((A1, 7.0), F_k30, F_k15), ((A2, 10.0), F_k15, F_c20),
                      ((L - 1, 10.0), F_c20, F_ch), ((L, 9.0), F_ch, F_back), ((L, BO_R), F_back, F_bore),
                      ((BO_A, BO_R), F_bore, F_btip)]:
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
def new_axis(o, d):
    aid = f"Axis-{len(axes)+1}"; axes.append((aid, tuple(o), tuple(d))); return aid
for f in F_inc:
    f["axis"] = new_axis(f["o"], f["d"])
for f in faces:
    f.setdefault("axis", "Axis-1" if f["kind"] in ("CYL", "CONE", "SPH") else "")
features = [dict(id="Feature-1", name="Drehkontur", type="REVOLVE", params={})]
def feat(name, typ, params=None):
    fid = f"Feature-{len(features)+1}"
    features.append(dict(id=fid, name=name, type=typ, params=params or {})); return fid
fb = feat("Bohrung Ø5", "SIMPLE HOLE")
F_bore["feat"] = F_btip["feat"] = fb
for i, f in enumerate(F_inc):
    f["feat"] = feat(f"Schrägbohrung Ø3 45° ({i+1})", "SIMPLE HOLE")
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
w('<!-- Demo-Indexbolzen für 3D Viewer – frei erfunden, frei verwendbar, keine echten Konstruktionsdaten -->')
w('<classmate-cad version="2.0">')
w('  <identifier>DEMO-INDEXBOLZEN-001</identifier>\n  <system>3D Viewer Demo</system>\n  <date>2026.11.17 12:00:00</date>')
w('  <configurations><configuration><type>geometry</type><identifier>demo</identifier><views><view>'
  '<name>CLASSMATE</name><type>CLASSMATE</type>'
  '<transform>[(1.0,0.0,0.0,0.0);(0.0,-0.889,-0.458,0.0);(0.0,0.458,-0.889,0.0);(0.0,0.0,0.0,1.0)]</transform>'
  '</view></views></configuration></configurations>')
w('  <models>\n    <model>\n      <id>Model-1</id>\n      <type>part</type>\n      <name>Demo-Indexbolzen</name>\n      <units>mm</units>')
w(f'      <length>{hi[0]-lo[0]:.4f}</length>\n      <width>{hi[1]-lo[1]:.4f}</width>\n      <height>{hi[2]-lo[2]:.4f}</height>')
w(f'      <area>{area.sum():.3f}</area>\n      <volume>{vol:.3f}</volume>\n      <mass>{vol*7.85e-6:.4f}</mass>')
w(f'      <center>{v3((lo+hi)/2)}</center>\n      <main-axis>Axis-1</main-axis>')
w('      <parameters>')
for n, v in [("DB_PART_NAME", "Demo-Indexbolzen"), ("Material", "16MnCr5 (1.7131)"),
             ("Beschreibung", "Demo-Indexbolzen: R5 / 30° / 15°, 4x Ø3 unter 45° in Bohrung Ø5"), ("Ersteller", "3D Viewer")]:
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
