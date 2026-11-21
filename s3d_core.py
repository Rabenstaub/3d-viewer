# -*- coding: utf-8 -*-
"""
s3d_core – Geometrie-Kern für den S3D Viewer (ohne GUI-Abhängigkeiten)

  * Parser für .s3d-Dateien (XML)
  * Analytische Flächen (Ebene, Zylinder, Kegel, Torus) aus Achse + Tessellierung
  * Rissfreie Verfeinerung der Tessellierung + exakte Normalen  -> glatte Darstellung
  * Messfunktionen (Abstand, Winkel, Ø) zwischen Punkten, Achsen, Ebenen
  * Werkstück-Nullpunkt (WCS) und Bohrungstabelle
"""

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------

_VEC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_PRINTABLE_RE = re.compile(r"^[\x20-\x7E -ɏ -⃏\t]*$")


def _vec(text):
    if not text:
        return None
    nums = _VEC_RE.findall(text)
    if len(nums) < 3:
        return None
    return np.array([float(n) for n in nums[:3]], dtype=float)


def _float(text, default=None):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _clean(text):
    if text is None:
        return ""
    t = text.strip()
    if not _PRINTABLE_RE.match(t):
        return None
    return t


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


def any_perp(d):
    d = unit(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    return unit(np.cross(d, a))


def fmt(v, digits=3):
    if v is None:
        return "–"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) < 0.5 * 10 ** (-digits):
        v = 0.0
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# ---------------------------------------------------------------------------
# Analytische Flächen
# ---------------------------------------------------------------------------

class PlaneSurf:
    kind = "PLANE"

    def __init__(self, p, n):
        self.p = np.asarray(p, float)
        self.n = unit(n)

    def project(self, P):
        d = (P - self.p) @ self.n
        return P - d[:, None] * self.n[None, :]

    def normals(self, P):
        return np.tile(self.n, (len(P), 1))


class RevSurf:
    """Rotationsfläche um Achse (o, d). Profil in der (a, rho)-Halbebene:
       'line'   : Gerade  (Zylinder, Kegel)   – p0 (a,rho), t (Richtung)
       'circle' : Kreis   (Torus)             – c (a,rho),  r
    """

    def __init__(self, kind, o, d, profile, **kw):
        self.kind = kind
        self.o = np.asarray(o, float)
        self.d = unit(d)
        self.profile = profile
        self.__dict__.update(kw)
        self._perp = any_perp(self.d)

    def _local(self, P):
        v = P - self.o
        a = v @ self.d
        radial = v - a[:, None] * self.d[None, :]
        rho = np.linalg.norm(radial, axis=1)
        u = np.empty_like(radial)
        ok = rho > 1e-12
        u[ok] = radial[ok] / rho[ok, None]
        u[~ok] = self._perp
        return a, rho, u

    def _proj2(self, a, rho):
        if self.profile == "line":
            p0, t = self.p0, self.t
            s = (a - p0[0]) * t[0] + (rho - p0[1]) * t[1]
            a2 = p0[0] + s * t[0]
            r2 = p0[1] + s * t[1]
            na = np.full_like(a, -t[1])
            nr = np.full_like(a, t[0])
        else:
            wa, wr = a - self.c[0], rho - self.c[1]
            L = np.hypot(wa, wr)
            L[L < 1e-12] = 1e-12
            na, nr = wa / L, wr / L
            a2 = self.c[0] + na * self.r
            r2 = self.c[1] + nr * self.r
        return a2, np.maximum(r2, 0.0), na, nr

    def project(self, P):
        a, rho, u = self._local(P)
        a2, r2, _, _ = self._proj2(a, rho)
        return self.o + a2[:, None] * self.d + r2[:, None] * u

    def normals(self, P):
        a, rho, u = self._local(P)
        _, _, na, nr = self._proj2(a, rho)
        n = na[:, None] * self.d + nr[:, None] * u
        return n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)


def _fit_line_2d(x, y):
    pts = np.column_stack([x, y])
    c = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    t = vt[0]
    resid = s[1] / math.sqrt(len(pts)) if len(s) > 1 else 0.0
    return c, t / np.linalg.norm(t), resid


def _fit_circle_2d(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = math.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 0.0))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return np.array([cx, cy]), r, resid


def fit_surface(face, axis, size):
    """Analytische Fläche aus Achse + Tessellierungs-Punkten bestimmen."""
    V = face.triangles.reshape(-1, 3)
    tol = max(size * 2e-4, 1e-3)
    if face.ftype == "PLANE":
        c = V.mean(axis=0)
        _, s, vt = np.linalg.svd(V - c, full_matrices=False)
        n = vt[2]
        if s[2] / math.sqrt(len(V)) > tol:
            return None
        return PlaneSurf(c, n)
    if axis is None or face.ftype not in ("CYLINDER", "CONE", "TORUS", "SPHERE"):
        return None
    o, d = axis
    d = unit(d)
    v = V - o
    a = v @ d
    rho = np.linalg.norm(v - a[:, None] * d, axis=1)
    if face.ftype == "CYLINDER":
        R = face.radius1 if face.radius1 else float(np.median(rho))
        if np.median(np.abs(rho - R)) > tol:
            R = float(np.median(rho))
        s = RevSurf("CYLINDER", o, d, "line", p0=np.array([0.0, R]), t=np.array([1.0, 0.0]),
                    radius=R)
        return s
    if face.ftype == "CONE":
        cand = []
        if None not in (face.radius1, face.radius2, face.start, face.end) and abs(face.end - face.start) > 1e-9:
            p0 = np.array([face.start, face.radius1])
            t = unit(np.array([face.end - face.start, face.radius2 - face.radius1]))
            cand.append((p0, t))
        c, t, _ = _fit_line_2d(a, rho)
        cand.append((c, t))
        best = None
        for p0, t in cand:
            nrm = np.array([-t[1], t[0]])
            res = np.median(np.abs((a - p0[0]) * nrm[0] + (rho - p0[1]) * nrm[1]))
            if best is None or res < best[0]:
                best = (res, p0, t)
        res, p0, t = best
        if res > tol:
            return None
        if t[0] < 0:
            t = -t
        half = math.degrees(math.atan2(abs(t[1]), abs(t[0])))
        if face.angle is not None and abs(face.angle - half) < 2.0:
            half = face.angle            # exakter Wert aus dem CAD
        return RevSurf("CONE", o, d, "line", p0=p0, t=t, half_angle=half)
    if face.ftype in ("TORUS", "SPHERE"):
        keep = np.ones(len(a), bool)
        for _ in range(3):
            c, r, _ = _fit_circle_2d(a[keep], rho[keep])
            err = np.abs(np.hypot(a - c[0], rho - c[1]) - r)
            keep = err <= max(3 * np.median(err), 1e-4)
        e_circ = float(np.percentile(err, 90))
        lc, lt, _ = _fit_line_2d(a, rho)
        nrm = np.array([-lt[1], lt[0]])
        e_line = float(np.percentile(np.abs((a - lc[0]) * nrm[0] + (rho - lc[1]) * nrm[1]), 90))
        n_prof = len(np.unique(np.round(np.c_[a, rho] / max(tol * 10, 0.02)), axis=0))
        if (n_prof < 3 or e_line <= max(e_circ * 1.5, 1e-4)) and e_line < tol * 5:
            if lt[0] < 0:
                lt = -lt
            half = math.degrees(math.atan2(abs(lt[1]), abs(lt[0])))
            return RevSurf("CONE", o, d, "line", p0=lc, t=lt, half_angle=half)
        if e_circ > tol or r <= 0:
            return None
        return RevSurf("TORUS", o, d, "circle", c=c, r=r, major=c[1])
    return None


# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------

@dataclass
class Face:
    id: str
    ftype: str
    color: tuple
    triangles: np.ndarray
    radius1: float = None
    radius2: float = None
    length: float = None
    width: float = None
    area: float = None
    angle: float = None
    face_angle: float = None
    concavity: str = ""
    axis_id: str = ""
    feature: str = ""
    start: float = None
    end: float = None
    edges: list = field(default_factory=list)
    surf: object = None


@dataclass
class Edge:
    id: str
    etype: str
    points: np.ndarray
    length: float = None
    radius: float = None
    center: np.ndarray = None
    normal: np.ndarray = None
    start: np.ndarray = None
    end: np.ndarray = None


@dataclass
class Feature:
    id: str
    name: str
    ftype: str
    params: dict = field(default_factory=dict)


@dataclass
class Model:
    path: str = ""
    views: list = field(default_factory=list)   # [(Name, 3x3-Matrix)] gespeicherte Ansichten
    system: str = ""
    date: str = ""
    name: str = ""
    units: str = "mm"
    info: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    edges: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)
    main_axis: str = ""
    # abgeleitet
    size: float = 1.0
    tris: np.ndarray = None       # Original-Dreiecke (n,3,3)
    tri_face: np.ndarray = None


def _sample_circle(center, normal, start, length, radius, start_tan):
    n = unit(normal)
    r0 = start - center
    r0 = r0 - (r0 @ n) * n
    if radius is None or radius <= 0:
        radius = np.linalg.norm(r0)
    if radius <= 0 or np.linalg.norm(r0) < 1e-12:
        return None
    r0 = unit(r0) * radius
    ang = (length / radius) if length else 2 * math.pi
    ang = min(ang, 2 * math.pi)
    if start_tan is not None and np.dot(np.cross(n, r0), start_tan) < 0:
        ang = -ang
    steps = max(8, int(abs(ang) / (2 * math.pi) * 180))
    t = np.linspace(0.0, ang, steps + 1)
    b = np.cross(n, r0)
    return center[None, :] + np.cos(t)[:, None] * r0[None, :] + np.sin(t)[:, None] * b[None, :]


def parse_s3d(path):
    root = ET.parse(path).getroot()
    if root.tag != "classmate-cad":
        raise ValueError("Keine gültige .s3d-Datei (Wurzel-Element: %s)" % root.tag)
    m = Model(path=path)
    m.system = root.findtext("system", "") or ""
    m.date = root.findtext("date", "") or ""
    mod = root.find(".//models/model")
    if mod is None:
        raise ValueError("Kein <model> in der Datei gefunden.")
    m.name = mod.findtext("name", "") or ""
    m.units = mod.findtext("units", "mm") or "mm"
    m.main_axis = (mod.findtext("main-axis", "") or "").strip()
    for v in root.iter("view"):
        nums = _VEC_RE.findall(v.findtext("transform", "") or "")
        if len(nums) >= 16:
            T = np.array([float(x) for x in nums[:16]]).reshape(4, 4)[:3, :3]
            m.views.append(((v.findtext("name") or v.findtext("type") or "Ansicht").strip(), T))
    for key in ("length", "width", "height", "area", "volume", "mass", "center"):
        v = _clean(mod.findtext(key))
        if v:
            m.info[key] = v
    for p in mod.findall("parameters/parameter"):
        val = _clean(p.findtext("value"))
        if val is None:
            continue
        m.params.append(((p.findtext("type") or "").strip(), (p.findtext("name") or "").strip(), val))

    for a in root.iter("axis"):
        o, d = _vec(a.findtext("origin")), _vec(a.findtext("direction"))
        if o is not None and d is not None and np.linalg.norm(d) > 0:
            m.axes[a.findtext("id", "")] = (o, unit(d))

    for e in mod.findall("edges/edge"):
        eid = e.findtext("id", "")
        et = (e.findtext("edge-type") or "").upper()
        s, en = _vec(e.findtext("start")), _vec(e.findtext("end"))
        L, r = _float(e.findtext("length")), _float(e.findtext("radius1"))
        c, n = _vec(e.findtext("center")), _vec(e.findtext("normal"))
        pts = None
        if et == "CIRCLE" and c is not None and n is not None and s is not None:
            pts = _sample_circle(c, n, s, L, r, _vec(e.findtext("start-tangent")))
        if pts is None and s is not None and en is not None:
            pts = np.array([s, en])
        if pts is not None:
            m.edges[eid] = Edge(eid, et, pts, L, r, c, unit(n) if n is not None else None, s, en)

    for f in mod.findall("features/feature"):
        fid = f.findtext("id", "")
        params = {}
        for fp in f.findall("parameters/feature-parameter"):
            k = (fp.findtext("name") or fp.findtext("type") or "").strip()
            v = _clean(fp.findtext("value"))
            if k and v:
                params[k] = v
        m.features[fid] = Feature(fid, f.findtext("name", fid) or fid, f.findtext("feature-type", "") or "", params)

    for f in mod.findall("faces/face"):
        txt = f.findtext("face-tesselation/triangles") or ""
        try:
            nums = np.array(txt.replace(",", " ").split(), dtype=float)
        except ValueError:
            continue
        nums = nums[: (len(nums) // 9) * 9]
        if nums.size == 0:
            continue
        col = (0.75, 0.75, 0.75)
        for fp in f.findall("parameters/face-parameter"):
            name = (fp.findtext("name") or fp.findtext("type") or "").upper()
            if "DIFFUSE" in name:
                try:
                    rgb = [int(x) / 255.0 for x in (fp.findtext("value") or "").split(";")[:3]]
                    if len(rgb) == 3:
                        col = tuple(rgb)
                except ValueError:
                    pass
        edges = [ci.findtext("edge-ref") for ci in f.iter("contour-item")]
        m.faces.append(Face(
            id=f.findtext("id", ""), ftype=(f.findtext("face-type") or "").upper(), color=col,
            triangles=nums.reshape(-1, 3, 3),
            radius1=_float(f.findtext("radius1")), radius2=_float(f.findtext("radius2")),
            length=_float(f.findtext("length")), width=_float(f.findtext("width")),
            area=_float(f.findtext("area")), angle=_float(f.findtext("angle")),
            face_angle=_float(f.findtext("face-angle")),
            concavity=(f.findtext("concavity", "") or "").strip(),
            axis_id=f.findtext("face-axis", "") or "", feature=f.findtext("face-feature", "") or "",
            start=_float(f.findtext("start-position")), end=_float(f.findtext("end-position")),
            edges=[x for x in edges if x]))
    if not m.faces:
        raise ValueError("Keine Flächen mit Tessellierung gefunden.")

    m.tris = np.concatenate([f.triangles for f in m.faces])
    m.tri_face = np.concatenate([np.full(len(f.triangles), i) for i, f in enumerate(m.faces)])
    V = m.tris.reshape(-1, 3)
    m.size = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0
    for f in m.faces:
        try:
            f.surf = fit_surface(f, m.axes.get(f.axis_id), m.size)
        except Exception:
            f.surf = None
    return m


# ---------------------------------------------------------------------------
# Verfeinerung (rissfrei: Entscheidung pro Kante, neue Punkte auf exakter Geometrie)
# ---------------------------------------------------------------------------

def _project_multi(P, surfs, iters=8):
    """Punkte auf den Schnitt mehrerer Flächen projizieren (alternierend)."""
    surfs = [s for s in surfs if s is not None]
    if not surfs:
        return P
    if len(surfs) == 1:
        return surfs[0].project(P)
    Q = P.copy()
    for _ in range(iters):
        for s in surfs:
            Q = s.project(Q)
    return Q


def refine_mesh(model, tol=None, max_levels=5, max_tris=600_000):
    """Liefert (V, T, F): Vertices, Dreiecke (Indizes), Flächen-Index je Dreieck."""
    tris = model.tris.reshape(-1, 3)
    key = np.round(tris / 1e-5).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = tris[first].copy()
    T = inv.reshape(-1, 3).astype(np.int64)
    F = model.tri_face.astype(np.int64).copy()
    surfs = [f.surf for f in model.faces]
    if tol is None:
        tol = max(model.size * 4e-5, 5e-4)
    off_max = max(model.size * 3e-3, 0.05)

    for _ in range(max_levels):
        if len(T) * 2 > max_tris:
            break
        E = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1).reshape(-1, 2)
        Es = np.sort(E, axis=1)
        nv = np.int64(len(V))
        ukey, einv = np.unique(Es[:, 0] * nv + Es[:, 1], return_inverse=True)
        uE = np.stack([ukey // nv, ukey % nv], axis=1)
        einv = einv.reshape(-1)
        nE = len(uE)
        # Flächen je Kante (max. 2 verschiedene merken)
        ef = np.repeat(F, 3)
        fa = np.full(nE, -1, np.int64)
        fb = np.full(nE, -1, np.int64)
        fa[einv] = ef
        other = ef != fa[einv]
        fb[einv[other]] = ef[other]
        mid = (V[uE[:, 0]] + V[uE[:, 1]]) * 0.5
        proj = mid.copy()
        nf = np.int64(len(surfs) + 1)
        pkey = np.minimum(fa, np.where(fb < 0, fa, fb)) * nf + np.maximum(fa, fb)
        upk, pinv = np.unique(pkey, return_inverse=True)
        upair = np.stack([upk // nf, upk % nf], axis=1)
        pinv = pinv.reshape(-1)
        order = np.argsort(pinv, kind="stable")
        bounds = np.searchsorted(pinv[order], np.arange(len(upair) + 1))
        for gi, (f1, f2) in enumerate(upair):
            idx = order[bounds[gi]:bounds[gi + 1]]
            ss = [surfs[f1]] if f1 == f2 or f2 < 0 else [surfs[f1], surfs[f2]]
            ss = [x for x in ss if x is not None]
            if not ss or all(isinstance(x, PlaneSurf) for x in ss):
                continue
            # Endpunkte, die nicht exakt auf der Fläche liegen (Export-Ungenauigkeit),
            # werden über ihren Restfehler mitgeführt -> glatter Verlauf ohne Beulen
            A, B = V[uE[idx, 0]], V[uE[idx, 1]]
            rA = A - _project_multi(A, ss)
            rB = B - _project_multi(B, ss)
            good = (np.linalg.norm(rA, axis=1) < off_max) & (np.linalg.norm(rB, axis=1) < off_max)
            if not good.any():
                continue
            idx = idx[good]
            proj[idx] = _project_multi(mid[idx], ss) + 0.5 * (rA[good] + rB[good])
        dev = np.linalg.norm(proj - mid, axis=1)
        elen = np.linalg.norm(V[uE[:, 0]] - V[uE[:, 1]], axis=1)
        split = (dev > tol) & (elen > tol * 4) & (dev < elen)  # Ausreißer verwerfen
        if not split.any():
            break
        new_idx = np.full(nE, -1, np.int64)
        ns = int(split.sum())
        new_idx[split] = np.arange(len(V), len(V) + ns)
        V = np.vstack([V, proj[split]])

        tri_e = einv.reshape(-1, 3)          # Kante 01, 12, 20
        s = split[tri_e]                     # (m,3)
        mids = new_idx[tri_e]
        cnt = s.sum(axis=1)
        outT, outF = [T[cnt == 0]], [F[cnt == 0]]

        # 3 geteilte Kanten -> 4 Dreiecke
        k = cnt == 3
        if k.any():
            a, b, c = T[k, 0], T[k, 1], T[k, 2]
            m01, m12, m20 = mids[k, 0], mids[k, 1], mids[k, 2]
            outT += [np.stack(x, 1) for x in ((a, m01, m20), (m01, b, m12), (m20, m12, c), (m01, m12, m20))]
            outF += [F[k]] * 4
        # 1 geteilte Kante
        k1 = cnt == 1
        if k1.any():
            for r in range(3):
                kk = k1 & s[:, r]
                if not kk.any():
                    continue
                a, b, c = T[kk, r], T[kk, (r + 1) % 3], T[kk, (r + 2) % 3]
                mm = mids[kk, r]
                outT += [np.stack((a, mm, c), 1), np.stack((mm, b, c), 1)]
                outF += [F[kk]] * 2
        # 2 geteilte Kanten
        k2 = cnt == 2
        if k2.any():
            for r in range(3):  # r = nicht geteilte Kante -> rotieren, sodass Kante 20 frei ist
                kk = k2 & ~s[:, r]
                if not kk.any():
                    continue
                # nicht geteilte Kante r verbindet Ecke r und r+1; umbenennen: c=r, a=r+1, b=r+2
                a, b, c = T[kk, (r + 1) % 3], T[kk, (r + 2) % 3], T[kk, r]
                mab, mbc = mids[kk, (r + 1) % 3], mids[kk, (r + 2) % 3]
                outT += [np.stack((mab, b, mbc), 1), np.stack((a, mab, mbc), 1), np.stack((a, mbc, c), 1)]
                outF += [F[kk]] * 3
        T = np.concatenate(outT)
        F = np.concatenate(outF)
    return V, T, F


def vertex_normals(model, V, T, F):
    """Normalen je Dreiecks-Ecke (n,3,3): exakt aus der Fläche, sonst Dreiecksnormale."""
    P = V[T]                                   # (n,3,3)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-15)
    N = np.repeat(fn[:, None, :], 3, axis=1)
    for fi, f in enumerate(model.faces):
        if f.surf is None:
            continue
        k = np.where(F == fi)[0]
        if len(k) == 0:
            continue
        pts = P[k].reshape(-1, 3)
        n = f.surf.normals(pts).reshape(-1, 3, 3)
        # an der Dreiecksnormale ausrichten (gleiche Seite)
        sgn = np.sign(np.einsum("ijk,ik->ij", n, fn[k]))
        sgn[sgn == 0] = 1
        N[k] = n * sgn[:, :, None]
    return N


def feature_edges(V, T, F):
    """Kanten zwischen verschiedenen Flächen (und offene Ränder) als Liniensegmente."""
    E = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1).reshape(-1, 2)
    Es = np.sort(E, axis=1)
    ef = np.repeat(F, 3)
    nv = np.int64(len(V))
    ukey, einv, cnt = np.unique(Es[:, 0] * nv + Es[:, 1], return_inverse=True, return_counts=True)
    uE = np.stack([ukey // nv, ukey % nv], axis=1)
    einv = einv.reshape(-1)
    fmin = np.full(len(uE), np.iinfo(np.int64).max)
    fmax = np.full(len(uE), -1)
    np.minimum.at(fmin, einv, ef)
    np.maximum.at(fmax, einv, ef)
    keep = (fmin != fmax) | (cnt == 1)
    seg = V[uE[keep]]
    return seg.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Punkt-in-Körper (Strahl-Parität)
# ---------------------------------------------------------------------------

def ray_hits(tris, o, d):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tv = o - v0
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return hit, t


_RAY_DIRS = [unit(np.array(x)) for x in ((0.5773, 0.6123, 0.5401), (-0.3311, 0.8123, -0.4801), (0.7011, -0.2133, -0.6801))]


def inside(model, p):
    votes = 0
    for d in _RAY_DIRS:
        hit, _ = ray_hits(model.tris, np.asarray(p, float), d)
        votes += int(hit.sum()) % 2
    return votes >= 2


# ---------------------------------------------------------------------------
# Werkstück-Koordinatensystem
# ---------------------------------------------------------------------------

class WCS:
    def __init__(self):
        self.reset()

    def reset(self):
        self.origin = np.zeros(3)
        self.R = np.eye(3)       # Zeilen = X, Y, Z (in Modellkoordinaten)
        self.name = "Modell-Nullpunkt"

    def set_lathe(self, model):
        """Drehmaschinen-Konvention: Z = Längsachse (Hauptachse), X = radial nach oben, Y quer."""
        o, d = main_axis(model)
        foot = o + ((np.zeros(3) - o) @ d) * d       # Modell-Ursprung auf die Achse fallen lassen
        up = None
        for cand in (np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])):
            x = cand - (cand @ d) * d
            if np.linalg.norm(x) > 1e-6:
                up = unit(x)
                break
        self.origin = foot
        self.R = np.vstack([up, np.cross(d, up), d])
        self.name = "Drehmaschine (Z = Längsachse)"

    def is_model(self):
        return np.allclose(self.origin, 0) and np.allclose(self.R, np.eye(3))

    def to_wcs(self, p):
        return self.R @ (np.asarray(p, float) - self.origin)

    def dir_to_wcs(self, v):
        return self.R @ np.asarray(v, float)

    def set(self, origin, z, x_hint=None, name="Benutzer-Nullpunkt"):
        z = unit(z)
        if x_hint is None:
            x_hint = self.R[0]
        x = np.asarray(x_hint, float) - (np.asarray(x_hint, float) @ z) * z
        if np.linalg.norm(x) < 1e-6:
            for cand in (self.R[1], np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
                x = cand - (cand @ z) * z
                if np.linalg.norm(x) > 1e-6:
                    break
        x = unit(x)
        y = np.cross(z, x)
        self.origin = np.asarray(origin, float)
        self.R = np.vstack([x, y, z])
        self.name = name

    def flip_z(self):
        self.R = np.vstack([self.R[0], -self.R[1], -self.R[2]])

    def rotate_z(self, deg):
        a = math.radians(deg)
        x, y, z = self.R
        x2 = math.cos(a) * x + math.sin(a) * y
        self.R = np.vstack([x2, np.cross(z, x2), z])

    def align_x(self, direction):
        z = self.R[2]
        x = np.asarray(direction, float) - (np.asarray(direction, float) @ z) * z
        if np.linalg.norm(x) < 1e-9:
            return False
        x = unit(x)
        self.R = np.vstack([x, np.cross(z, x), z])
        return True

    def shift(self, dx, dy, dz):
        self.origin = self.origin + self.R.T @ np.array([dx, dy, dz], float)


# ---------------------------------------------------------------------------
# Auswahl-Objekte und Messen
# ---------------------------------------------------------------------------

@dataclass
class Prim:
    kind: str                 # 'point' | 'line' | 'plane'
    p: np.ndarray
    d: np.ndarray = None      # Richtung (line) bzw. Normale (plane)


class Entity:
    """Auswahl-Element: Fläche, Kante oder Punkt."""

    def __init__(self, model, kind, ref, point=None):
        self.model, self.kind, self.ref = model, kind, ref
        self.point = None if point is None else np.asarray(point, float)

    def key(self):
        if self.kind == "point":
            return ("point", tuple(np.round(self.point, 6)))
        return (self.kind, self.ref)

    # --- Beschreibung
    def label(self):
        if self.kind == "face":
            f = self.model.faces[self.ref]
            return f"{f.id} ({FACE_TYPES_DE.get(f.ftype, f.ftype)})"
        if self.kind == "edge":
            e = self.model.edges[self.ref]
            return f"{e.id} ({EDGE_TYPES_DE.get(e.etype, e.etype)})"
        return "Eckpunkt" if getattr(self, "is_vertex", False) else "Punkt"

    @property
    def face(self):
        return self.model.faces[self.ref] if self.kind == "face" else None

    @property
    def edge(self):
        return self.model.edges[self.ref] if self.kind == "edge" else None

    def axis(self):
        """(Punkt, Richtung) einer Achse, falls vorhanden."""
        if self.kind == "face":
            s = self.face.surf
            if isinstance(s, RevSurf):
                return s.o, s.d
        if self.kind == "edge" and self.edge.etype == "CIRCLE" and self.edge.center is not None:
            return self.edge.center, self.edge.normal
        return None

    def primary(self):
        if self.kind == "point":
            return Prim("point", self.point)
        if self.kind == "edge":
            e = self.edge
            if e.etype == "CIRCLE" and e.center is not None:
                return Prim("point", e.center, e.normal)
            if e.start is not None and e.end is not None and np.linalg.norm(e.end - e.start) > 1e-9:
                return Prim("line", e.start, unit(e.end - e.start))
            return Prim("point", e.points[0])
        f = self.face
        s = f.surf
        if isinstance(s, PlaneSurf):
            return Prim("plane", s.p, s.n)
        if isinstance(s, RevSurf):
            # Achspunkt auf Höhe der Flächenmitte
            V = f.triangles.reshape(-1, 3)
            a = ((V - s.o) @ s.d).mean()
            return Prim("line", s.o + a * s.d, s.d)
        return Prim("point", f.triangles.reshape(-1, 3).mean(axis=0))

    def diameter(self):
        if self.kind == "face":
            s = self.face.surf
            if isinstance(s, RevSurf) and s.kind == "CYLINDER":
                return 2 * s.radius
            if self.face.ftype == "CYLINDER" and self.face.radius1:
                return 2 * self.face.radius1
        if self.kind == "edge" and self.edge.etype == "CIRCLE" and self.edge.radius:
            return 2 * self.edge.radius
        return None


FACE_TYPES_DE = {"PLANE": "Ebene", "CYLINDER": "Zylinder", "CONE": "Kegel",
                 "TORUS": "Torus", "SPHERE": "Kugel", "FREEFORM": "Freiform"}
EDGE_TYPES_DE = {"CIRCLE": "Kreis", "LINE": "Gerade", "FREEFORM": "Freiform"}


def _angle(u, v):
    c = abs(float(np.clip(unit(u) @ unit(v), -1, 1)))
    return math.degrees(math.acos(min(1.0, c)))


def _closest_line_line(p1, d1, p2, d2):
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if den < 1e-12:
        return None
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return p1 + s * d1, p2 + t * d2


def _rev_info(e):
    """Für Drehflächen/Kreiskanten: (o, d, kind, R, alpha, a_lo, a_hi)."""
    if e.kind == "face" and isinstance(e.face.surf, RevSurf):
        s = e.face.surf
        V = e.face.triangles.reshape(-1, 3)
        a = (V - s.o) @ s.d
        if s.kind == "CYLINDER":
            return dict(o=s.o, d=s.d, kind="CYL", R=s.radius, alpha=0.0, lo=a.min(), hi=a.max(), e=e)
        if s.kind == "CONE":
            rho = np.linalg.norm(V - s.o - np.outer(a, s.d), axis=1)
            f = e.face
            ax_len = abs(f.end - f.start) if None not in (f.start, f.end) else a.max() - a.min()
            if None not in (f.radius1, f.radius2):
                rmin, rmax = min(f.radius1, f.radius2), max(f.radius1, f.radius2)
            else:
                rmin, rmax = rho.min(), rho.max()
            return dict(o=s.o, d=s.d, kind="CONE", R=None, alpha=s.half_angle, lo=a.min(), hi=a.min() + ax_len,
                        rmin=rmin, rmax=rmax, e=e)
        if s.kind == "TORUS":
            return dict(o=s.o, d=s.d, kind="TORUS", R=None, alpha=None, lo=a.min(), hi=a.max(), r=s.r, e=e)
    if e.kind == "edge" and e.edge.etype == "CIRCLE" and e.edge.center is not None:
        ed = e.edge
        return dict(o=ed.center, d=ed.normal, kind="CIRCLE", R=ed.radius, alpha=None, lo=0.0, hi=0.0, e=e)
    return None


def _coaxial(i1, i2, tol_ang=0.05, tol_d=2e-3):
    if _angle(i1["d"], i2["d"]) > tol_ang:
        return False
    v = i2["o"] - i1["o"]
    d = unit(i1["d"])
    return np.linalg.norm(v - (v @ d) * d) < tol_d


def _plane_of(e):
    if e.kind == "face" and isinstance(e.face.surf, PlaneSurf):
        return e.face.surf
    return None


def _torus_measure(e1, e2, wcs):
    """Radien/Einstiche (Torus): axiale Abstände über den Radius-Mittelpunkt."""
    def tor(e):
        if e.kind == "face" and e.face.ftype in ("TORUS", "SPHERE") and isinstance(e.face.surf, RevSurf):
            try:
                return arc_profile(e.model, e.ref)
            except Exception:
                return None
        return None
    t1, t2 = tor(e1), tor(e2)
    if t1 is None and t2 is None:
        return None
    if t1 is None:
        e1, e2, t1, t2 = e2, e1, t2, t1
    o, d = t1["o"], unit(t1["d"])
    u = wcs.R[0] - (wcs.R[0] @ d) * d
    if np.linalg.norm(u) < 1e-6:
        u = any_perp(d)
    u = unit(u)
    a1, rho1, r1 = float(t1["c"][0]), float(t1["c"][1]), t1["r"]
    V1 = e1.face.triangles.reshape(-1, 3)
    ax1 = (V1 - o) @ d
    lo1, hi1 = float(ax1.min()), float(ax1.max())
    rows = []
    P = lambda a, rho: o + a * d + rho * u
    if t2 is not None:
        o2, d2 = t2["o"], unit(t2["d"])
        if _angle(d, d2) > 0.05:
            return None
        v = o2 - o
        if np.linalg.norm(v - (v @ d) * d) > 2e-3:
            return None
        sg = 1.0 if d2 @ d > 0 else -1.0
        a2 = float((o2 - o) @ d + sg * t2["c"][0])
        V2 = e2.face.triangles.reshape(-1, 3)
        ax2 = (V2 - o) @ d
        gap = max(float(ax2.min()) - hi1, lo1 - float(ax2.max()), 0.0)
        rows.append(("Abstand Mitte–Mitte (axial)", f"<b>{fmt(abs(a2 - a1))} mm</b>"))
        rows.append(("Abstand zwischen den Radien", f"{fmt(gap)} mm"))
        rows.append(("Radien", f"R {fmt(r1)}  /  R {fmt(t2['r'])}"))
        rows.append(("Ø der Radiusmitten", f"{fmt(2 * rho1)}  /  {fmt(2 * t2['c'][1])}"))
        return rows, (P(a1, rho1), P(a2, rho1))
    pl = _plane_of(e2)
    if pl is not None:
        if abs(abs(unit(pl.n) @ d) - 1) > 1e-4:
            return None
        ap = float((pl.p - o) @ d)
        near = min(abs(ap - lo1), abs(ap - hi1))
        rows.append(("Abstand Fläche – Radiusmitte (axial)", f"<b>{fmt(abs(ap - a1))} mm</b>"))
        rows.append(("Abstand Fläche – nächste Radiuskante", f"{fmt(near)} mm"))
        rows.append(("Radius", f"R {fmt(r1)}  ·  Ø Mitte {fmt(2 * rho1)}"))
        return rows, (P(a1, rho1), P(ap, rho1))
    if e2.kind == "edge" and e2.edge.etype == "CIRCLE" and e2.edge.center is not None:
        ed = e2.edge
        if _angle(ed.normal, d) > 0.05:
            return None
        ac = float((ed.center - o) @ d)
        rows.append(("Abstand Kante – Radiusmitte (axial)", f"<b>{fmt(abs(ac - a1))} mm</b>"))
        rows.append(("Radius", f"R {fmt(r1)}  ·  Ø Mitte {fmt(2 * rho1)}"))
        return rows, (P(a1, rho1), P(ac, rho1))
    i2 = _rev_info(e2)
    if i2 is not None and i2["kind"] == "CYL" and _angle(i2["d"], d) < 0.05:
        R = i2["R"]
        rows.append(("Radiusmitte – Zylinder (radial)", f"<b>{fmt(abs(R - rho1))} mm</b>"))
        rows.append(("Ø Zylinder / Ø Radiusmitte", f"{fmt(2 * R)}  /  {fmt(2 * rho1)}"))
        rows.append(("Radius", f"R {fmt(r1)}"))
        return rows, (P(a1, rho1), P(a1, R))
    return None


def measure_special(e1, e2, wcs):
    """Drehteil-Messungen: Wandstärke, Fasenwinkel, axiale Abstände. None, wenn nicht zutreffend."""
    tm = _torus_measure(e1, e2, wcs)
    if tm is not None:
        return tm
    i1, i2 = _rev_info(e1), _rev_info(e2)
    p1, p2 = _plane_of(e1), _plane_of(e2)
    rows, seg = [], None
    # --- zwei koaxiale Drehelemente
    if i1 and i2 and _coaxial(i1, i2):
        d = unit(i1["d"])
        o = i1["o"]
        u = any_perp(d)
        # axiale Lage der zweiten Achse relativ zur ersten
        shift = (i2["o"] - o) @ d
        lo2, hi2 = i2["lo"] + shift, i2["hi"] + shift
        k1, k2 = i1["kind"], i2["kind"]
        if k1 in ("CYL", "CIRCLE") and k2 in ("CYL", "CIRCLE"):
            R1, R2 = i1["R"], i2["R"]
            if k1 == "CIRCLE" and k2 == "CIRCLE" and abs(R1 - R2) < 1e-4:
                ax = abs(shift)
                rows.append(("Abstand (axial)", f"{fmt(ax)} mm"))
                rows.append(("Ø", f"{fmt(2 * R1)} mm"))
                seg = (o + R1 * u, o + shift * d + R1 * u)
                return rows, seg
            if abs(R1 - R2) > 1e-4:
                cc = [x["e"].face.concavity if x["e"].kind == "face" else "" for x in (i1, i2)]
                label = "Wandstärke" if set(cc) == {"0", "1"} else "Absatz (radial)"
                rows.append((label, f"<b>{fmt(abs(R1 - R2))} mm</b>"))
                a_mid = (max(i1["lo"], lo2) + min(i1["hi"], hi2)) / 2
                if max(i1["lo"], lo2) > min(i1["hi"], hi2):
                    a_mid = (i1["lo"] + i1["hi"]) / 2
                    gap = max(i1["lo"], lo2) - min(i1["hi"], hi2)
                    # z. B. Fase/Kegel zwischen zwei Kreiskanten: Länge in Z und Winkel – so wie die Steuerung
                    # sie haben will (Startpunkt Z, Länge, Winkel)
                    rows.append(("Länge (axial)", f"<b>{fmt(gap)} mm</b>"))
                    if gap > 1e-9 and abs(R1 - R2) > 1e-9:
                        ang = math.degrees(math.atan2(abs(R1 - R2), gap))
                        rows.append(("Winkel zur Achse", f"<b>{fmt(ang, 3)}°</b>  (zur Planfläche {fmt(90 - ang, 3)}°)"))
                rows.append(("Ø groß", f"{fmt(2 * max(R1, R2))} mm"))
                rows.append(("Ø klein", f"{fmt(2 * min(R1, R2))} mm"))
                rows.append(("Ø-Differenz", f"{fmt(2 * abs(R1 - R2))} mm"))
                seg = (o + a_mid * d + R1 * u, o + a_mid * d + R2 * u)
            else:
                rows.append(("Gleicher Ø", f"{fmt(2 * R1)} mm"))
                gap = max(i1["lo"], lo2) - min(i1["hi"], hi2)
                if k1 == "CIRCLE" or k2 == "CIRCLE":
                    gap = abs(shift) if k1 == k2 else gap
                if gap > 0:
                    rows.insert(0, ("Abstand (axial)", f"{fmt(gap)} mm"))
            return rows, seg
        # Kegel beteiligt -> Winkel
        if "CONE" in (k1, k2):
            cone = i1 if k1 == "CONE" else i2
            other = i2 if cone is i1 else i1
            al = cone["alpha"]
            if other["kind"] == "CONE":
                diff = abs(al - other["alpha"])
                rows.append(("Winkel zwischen Kegeln", f"{fmt(diff, 3)}°"))
                rows.append(("Kegel 1 zur Achse", f"{fmt(i1['alpha'], 3)}°"))
                rows.append(("Kegel 2 zur Achse", f"{fmt(i2['alpha'], 3)}°"))
            else:
                rows.append(("Fasenwinkel (zur Mantellinie)", f"<b>{fmt(al, 3)}°</b>"))
                rows.append(("Winkel zur Planfläche", f"{fmt(90 - al, 3)}°"))
                rows.append(("Kegel-Spitzenwinkel", f"{fmt(2 * al, 3)}°"))
            ax = cone["hi"] - cone["lo"]
            rad = cone["rmax"] - cone["rmin"]
            rows.append(("Fase axial × radial", f"{fmt(ax)} × {fmt(rad)} mm"))
            rows.append(("Fase schräge Länge", f"{fmt(math.hypot(ax, rad))} mm"))
            if other["kind"] in ("CYL", "CIRCLE"):
                rows.append(("Ø Zylinder", f"{fmt(2 * other['R'])} mm"))
            return rows, None
        if "TORUS" in (k1, k2):
            tor = i1 if k1 == "TORUS" else i2
            rows.append(("Radius (Rundung)", f"R {fmt(tor['r'])} mm"))
            return rows, None
    # --- zwei parallele, nicht koaxiale Bohrungen/Kreise -> Lochkreis + Winkel um die Hauptachse
    if i1 and i2 and _angle(i1["d"], i2["d"]) < 0.05 and not _coaxial(i1, i2):
        m = e1.model
        o, d = main_axis(m)
        if _angle(d, i1["d"]) < 0.5:
            ea, eb = angle_frame(d, wcs)
            r1, t1 = polar(i1["o"], o, d, ea, eb)
            r2, t2 = polar(i2["o"], o, d, ea, eb)
            v = i2["o"] - i1["o"]
            dd = unit(i1["d"])
            par = np.linalg.norm(v - (v @ dd) * dd)
            diff = abs(t2 - t1) % 360.0
            diff = min(diff, 360.0 - diff)
            rows.append(("Abstand (parallel)", f"{fmt(par)} mm"))
            if abs(r1 - r2) < 0.01:
                rows.append(("Lochkreis", f"<b>Ø {fmt(r1 + r2)} mm</b>"))
            else:
                rows.append(("Abstand zur Mittelachse 1", f"{fmt(r1)} mm"))
                rows.append(("Abstand zur Mittelachse 2", f"{fmt(r2)} mm"))
            rows.append(("Winkel zwischen Bohrungen", f"<b>{fmt(diff, 3)}°</b>  (Rest {fmt(360 - diff, 3)}°)"))
            nm = frame_name(ea, wcs)
            rows.append(("Lage Bohrung 1", f"{fmt(t1, 3)}°  (0° = {nm})"))
            rows.append(("Lage Bohrung 2", f"{fmt(t2, 3)}°"))
            for x in (i1, i2):
                if x["R"]:
                    rows.append((f"Ø {x['e'].label()}", f"{fmt(2 * x['R'])} mm"))
            pa = o + ((i1["o"] - o) @ d) * d
            return rows, (i1["o"], i1["o"] + (v - (v @ dd) * dd))
    # --- zwei radiale Bohrungen (Achsen senkrecht zur Hauptachse)
    if i1 and i2 and not _coaxial(i1, i2):
        m = e1.model
        o, d = main_axis(m)
        if abs(_angle(d, i1["d"]) - 90) < 0.5 and abs(_angle(d, i2["d"]) - 90) < 0.5:
            ea, eb = angle_frame(d, wcs)
            hs = [hole_for_entity(m, x["e"]) for x in (i1, i2)]
            dirs = [h.axis_d if h is not None else x["d"] for h, x in zip(hs, (i1, i2))]
            t = [math.degrees(math.atan2(v @ eb, v @ ea)) % 360.0 for v in dirs]
            diff = abs(t[1] - t[0]) % 360.0
            diff = min(diff, 360.0 - diff)
            rows.append(("Winkel zwischen Bohrungen (um Mittelachse)", f"<b>{fmt(diff, 3)}°</b>"))
            rows.append(("Lage Bohrung 1", f"{fmt(t[0], 3)}°  (0° = {frame_name(ea, wcs)})"))
            rows.append(("Lage Bohrung 2", f"{fmt(t[1], 3)}°"))
            a1 = (i1["o"] - o) @ d
            a2 = (i2["o"] - o) @ d
            rows.append(("Axialer Versatz", f"{fmt(abs(a2 - a1))} mm"))
            for x in (i1, i2):
                if x["R"]:
                    rows.append((f"Ø {x['e'].label()}", f"{fmt(2 * x['R'])} mm"))
            return rows, None
    # --- Planfläche + Drehelement mit Achse senkrecht zur Fläche
    pl, ri = (p1, i2) if p1 is not None and i2 else ((p2, i1) if p2 is not None and i1 else (None, None))
    if pl is not None and ri is not None and _angle(pl.n, ri["d"]) < 0.05:
        d = unit(ri["d"])
        if ri["kind"] == "CONE":
            al = ri["alpha"]
            rows.append(("Fasenwinkel (zur Planfläche)", f"<b>{fmt(90 - al, 3)}°</b>"))
            rows.append(("Winkel zur Mantellinie / Achse", f"{fmt(al, 3)}°"))
            ax = ri["hi"] - ri["lo"]
            rad = ri["rmax"] - ri["rmin"]
            rows.append(("Fase axial × radial", f"{fmt(ax)} × {fmt(rad)} mm"))
            return rows, None
        if ri["kind"] == "CIRCLE":
            dist = (ri["o"] - pl.p) @ pl.n
            rows.append(("Abstand Kreis–Fläche (axial)", f"{fmt(abs(dist))} mm"))
            rows.append(("Ø Kreis", f"{fmt(2 * ri['R'])} mm"))
            u = any_perp(d)
            q = ri["o"] + ri["R"] * u
            return rows, (q, q - dist * pl.n)
        if ri["kind"] == "CYL":
            # Abstand der Fläche zum näheren Zylinderende
            pa = (pl.p - ri["o"]) @ d
            gap = min(abs(pa - ri["lo"]), abs(pa - ri["hi"]))
            rows.append(("Planfläche ⟂ Achse", "Winkel 90°"))
            rows.append(("Abstand Fläche – Zylinderende", f"{fmt(gap)} mm"))
            rows.append(("Ø Zylinder", f"{fmt(2 * ri['R'])} mm"))
            return rows, None
    return None


def measure(e1, e2, wcs, par_tol=0.05):
    """Liefert (zeilen, segment): Messwerte + Linie (2 Punkte) für die Anzeige."""
    sp = measure_special(e1, e2, wcs)
    if sp is not None:
        return sp
    a, b = e1.primary(), e2.primary()
    rows = []
    seg = None
    # sortieren: point < line < plane
    order = {"point": 0, "line": 1, "plane": 2}
    swap = order[a.kind] > order[b.kind]
    ea, eb = e1, e2
    if swap:
        a, b = b, a
        ea, eb = eb, ea

    def centroid(e, prim):
        if e.kind == "face":
            return e.face.triangles.reshape(-1, 3).mean(axis=0)
        if e.kind == "edge":
            return e.edge.points.mean(axis=0)
        return prim.p

    def axis_point(e, prim, target):
        """Punkt auf der (unendlichen) Achse auf Höhe von target."""
        t = (target - prim.p) @ prim.d
        return prim.p + t * prim.d

    def dxyz(p, q):
        d = wcs.dir_to_wcs(q - p)
        return [("dX", fmt(abs(d[0]))), ("dY", fmt(abs(d[1]))), ("dZ", fmt(abs(d[2])))]

    if a.kind == "point" and b.kind == "point":
        dist = np.linalg.norm(b.p - a.p)
        rows.append(("Abstand", f"{fmt(dist)} mm"))
        rows += dxyz(a.p, b.p)
        seg = (a.p, b.p)
        if a.d is not None and b.d is not None:
            rows.append(("Winkel Achsen", f"{fmt(_angle(a.d, b.d), 3)}°"))
            if _angle(a.d, b.d) < par_tol:
                v = b.p - a.p
                perp = v - (v @ unit(a.d)) * unit(a.d)
                rows.append(("Achsabstand", f"{fmt(np.linalg.norm(perp))} mm"))
    elif a.kind == "point" and b.kind == "line":
        v = a.p - b.p
        foot = b.p + (v @ b.d) * b.d
        rows.append(("Abstand Punkt–Achse", f"{fmt(np.linalg.norm(a.p - foot))} mm"))
        seg = (a.p, foot)
        if a.d is not None:
            ang = _angle(a.d, b.d)
            rows.append(("Winkel Achsen", f"{fmt(ang, 3)}°"))
    elif a.kind == "point" and b.kind == "plane":
        dist = (a.p - b.p) @ b.d
        foot = a.p - dist * b.d
        rows.append(("Abstand Punkt–Ebene", f"{fmt(abs(dist))} mm"))
        seg = (a.p, foot)
        if a.d is not None:
            rows.append(("Winkel Achse–Ebene", f"{fmt(90 - _angle(a.d, b.d), 3)}°"))
    elif a.kind == "line" and b.kind == "line":
        ang = _angle(a.d, b.d)
        rows.append(("Winkel", f"{fmt(ang, 3)}°"))
        if ang < par_tol:
            v = b.p - a.p
            perp = v - (v @ a.d) * a.d
            rows.insert(0, ("Abstand (parallel)", f"{fmt(np.linalg.norm(perp))} mm"))
            pa = axis_point(ea, a, axis_point(eb, b, centroid(ea, a)))
            seg = (pa, pa + perp)
        else:
            cl = _closest_line_line(a.p, a.d, b.p, b.d)
            if cl:
                dist = np.linalg.norm(cl[1] - cl[0])
                rows.append(("Kürzester Abstand", f"{fmt(dist)} mm"))
                seg = cl
                if dist < 1e-3:
                    pw = wcs.to_wcs(cl[0])
                    rows.append(("Schnittpunkt", " / ".join(fmt(x) for x in pw)))
    elif a.kind == "line" and b.kind == "plane":
        ang = 90 - _angle(a.d, b.d)
        rows.append(("Winkel Achse–Ebene", f"{fmt(ang, 3)}°"))
        if ang < par_tol:
            dist = (a.p - b.p) @ b.d
            rows.insert(0, ("Abstand Achse–Ebene", f"{fmt(abs(dist))} mm"))
            pa = axis_point(ea, a, centroid(eb, b))
            seg = (pa, pa - dist * b.d)
        else:
            t = ((b.p - a.p) @ b.d) / (a.d @ b.d)
            ip = a.p + t * a.d
            rows.append(("Durchstoßpunkt", " / ".join(fmt(x) for x in wcs.to_wcs(ip))))
    elif a.kind == "plane" and b.kind == "plane":
        ang = _angle(a.d, b.d)
        if ang < par_tol:
            dist = (b.p - a.p) @ a.d
            rows.append(("Abstand (parallel)", f"{fmt(abs(dist))} mm"))
            seg = (a.p, a.p + dist * a.d)
        else:
            rows.append(("Winkel zwischen Flächen", f"{fmt(ang, 3)}°  /  {fmt(180 - ang, 3)}°"))
            rows.append(("Ergänzungswinkel", f"{fmt(90 - ang, 3)}°"))
    for e in (e1, e2):
        dia = e.diameter()
        if dia:
            rows.append((f"Ø {e.label()}", f"{fmt(dia)} mm"))
    return rows, seg


# ---------------------------------------------------------------------------
# Nullpunkt aus Auswahl
# ---------------------------------------------------------------------------

def _fit_circle_3d(points, d):
    u = any_perp(d)
    v = np.cross(d, u)
    x, y = points @ u, points @ v
    if len(points) == 2:
        c2 = np.array([x.mean(), y.mean()])
        r = float(np.hypot(x[0] - c2[0], y[0] - c2[1]))
        res = 0.0
    else:
        c2, r, res = _fit_circle_2d(x, y)
    h = float((points @ d).mean())
    return c2[0] * u + c2[1] * v + h * d, r, res


def _lines_intersection(points, dirs):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(points, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ p
    return np.linalg.lstsq(A, b, rcond=None)[0]


def outward_normal(model, face_idx, n):
    f = model.faces[face_idx]
    T = f.triangles
    c = T.mean(axis=1)
    areas = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    p = c[np.argmax(areas)]
    eps = max(model.size * 1e-4, 0.01)
    return -unit(n) if inside(model, p + eps * unit(n)) else unit(n)


def hole_for_entity(model, e):
    """Bohrung (aus find_holes) zu einer Fläche oder Kreiskante finden."""
    holes = getattr(model, "_holes", None)
    if holes is None:
        holes = model._holes = find_holes(model)
    if e.kind == "face":
        for h in holes:
            if e.ref in h.faces:
                return h
        return None
    ax = e.axis()
    if ax is None:
        return None
    p, d = ax
    for h in holes:
        if _angle(d, h.axis_d) > 0.05:
            continue
        v = p - h.axis_p
        if np.linalg.norm(v - (v @ h.axis_d) * h.axis_d) > 1e-3:
            continue
        a = (p - h.entry) @ h.axis_d
        if -h.depth - 0.05 <= a <= 0.05:
            return h
    return None


def wcs_from_selection(model, ents, wcs):
    """Bestimmt (origin, z, beschreibung) aus der Auswahl. Wirft ValueError bei Unklarheit."""
    axes, planes, points = [], [], []
    for e in ents:
        if e.kind == "point":
            points.append(e.point)
            continue
        if e.kind == "face" and isinstance(e.face.surf, PlaneSurf):
            planes.append((e.ref, e.face.surf))
            continue
        ax = e.axis()
        if ax is not None:
            p = e.primary().p
            axes.append((p, unit(ax[1]), e))
        elif e.kind == "edge":
            points.append(e.primary().p)

    plane = None
    if planes:
        fi, s = planes[0]
        plane = (s.p, outward_normal(model, fi, s.n))

    def on_plane(p, d):
        if plane is None:
            return p
        pp, pn = plane
        den = d @ pn
        if abs(den) < 1e-9:
            return p - ((p - pp) @ pn) * pn
        return p + ((pp - p) @ pn) / den * d

    if len(axes) >= 2:
        dirs = np.array([a[1] for a in axes])
        ref = dirs[0]
        parallel = all(_angle(ref, d) < 0.05 for d in dirs)
        pts = np.array([a[0] for a in axes])
        if parallel:
            c, r, res = _fit_circle_3d(pts, ref)
            hs = [hole_for_entity(model, a[2]) for a in axes]
            hs = [h for h in hs if h is not None]
            if plane is not None and _angle(plane[1], ref) < 1:
                z = plane[1]
                origin = on_plane(c, ref)
            elif hs:
                votes = sum(1 if h.axis_d @ ref > 0 else -1 for h in hs)
                z = ref if votes >= 0 else -ref
                level = max(h.entry @ z for h in hs)
                origin = c + (level - c @ z) * z
            else:
                z = ref if ref @ wcs.R[2] >= 0 else -ref
                origin = c
            txt = (f"Mitte Lochkreis aus {len(axes)} Achsen (Ø {fmt(2 * r)} mm)" if len(axes) >= 3
                   else "Mitte zwischen 2 Achsen")
            if len(axes) >= 3 and res > 0.01:
                txt += f" – Abweichung {fmt(res)} mm"
            return origin, z, txt
        # radiale Bohrungen -> Schnittpunkt der Achsen, Z senkrecht zu allen Richtungen
        c = _lines_intersection(pts, dirs)
        _, _, vt = np.linalg.svd(dirs)
        z = vt[2]
        if plane is not None:
            z = plane[1] if abs(plane[1] @ z) > 0.9 else z
            origin = on_plane(c, z)
        else:
            origin = c
        if z @ wcs.R[2] < 0:
            z = -z
        return origin, z, f"Schnittpunkt von {len(axes)} Achsen (radial)"
    if len(axes) == 1:
        p, d, e = axes[0]
        h = hole_for_entity(model, e)
        if h is not None:
            d = h.axis_d
        if plane is not None:
            z = plane[1] if _angle(plane[1], d) < 1 else d
            return on_plane(p, d), z, "Achse ∩ Ebene"
        if e.kind == "edge":
            n = d if h is not None else (d if d @ wcs.R[2] >= 0 else -d)
            return e.edge.center, n, "Kreismittelpunkt"
        if h is not None:
            return h.entry, h.axis_d, "Bohrungseintritt"
        # nur Zylinder: aktuellen Nullpunkt auf die Achse fallen lassen
        o = wcs.origin
        foot = p + ((o - p) @ d) * d
        return foot, d if d @ wcs.R[2] >= 0 else -d, "Auf Achse"
    if points:
        p = points[0] if len(points) == 1 else np.mean(points, axis=0)
        if plane is not None:
            return p, plane[1], "Punkt + Ebene"
        return p, wcs.R[2], "Punkt" if len(points) == 1 else f"Mittelpunkt aus {len(points)} Punkten"
    if plane is not None:
        pp, pn = plane
        o = wcs.origin
        return o - ((o - pp) @ pn) * pn, pn, "Z auf Ebene"
    raise ValueError("Bitte Bohrungen/Zylinder, Kreiskanten, Ebenen oder Punkte auswählen.")


# ---------------------------------------------------------------------------
# Bohrungstabelle
# ---------------------------------------------------------------------------

HOLE_TYPES = ("HOLE", "CBORE", "CSINK", "THREAD", "TAP")


@dataclass
class Hole:
    faces: list
    axis_p: np.ndarray
    axis_d: np.ndarray          # zeigt aus dem Material (Werkzeugachse)
    entry: np.ndarray
    depth: float
    diameters: list
    through: bool
    features: list
    thread: str = ""
    thread_params: dict = None
    thread_face: int = -1


def _vertex_faces(model):
    """Zuordnung gerundeter Netzpunkt -> Flächenindizes (einmal je Modell)."""
    vf = getattr(model, "_vertex_faces", None)
    if vf is None:
        vf = {}
        for j, f in enumerate(model.faces):
            for k in map(tuple, np.round(f.triangles.reshape(-1, 3), 4)):
                vf.setdefault(k, set()).add(j)
        model._vertex_faces = vf
    return vf


def _line_surface_root(surf, o, d, t0, t1, n=64):
    """Schnitt der Geraden o + t·d mit einer analytischen Fläche im Bereich [t0, t1];
    liefert die Nullstelle, die (t0 + t1) / 2 am nächsten liegt, oder None."""
    t = np.linspace(t0, t1, n + 1)
    P = o + np.outer(t, d)

    def g(P):
        Q = surf.project(P)
        return np.einsum("ij,ij->i", P - Q, surf.normals(Q))
    f = g(P)
    roots = []
    for i in range(n):
        if f[i] == 0.0:
            roots.append(t[i])
        elif f[i] * f[i + 1] < 0:
            a, b, fa = t[i], t[i + 1], f[i]
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = g((o + m * d)[None, :])[0]
                if fa * fm <= 0:
                    b = m
                else:
                    a, fa = m, fm
            roots.append(0.5 * (a + b))
    if not roots:
        return None
    mid = 0.5 * (t0 + t1)
    return float(min(roots, key=lambda r: abs(r - mid)))


def _point_in_face_range(f, p, eps=1e-3):
    """Liegt der Punkt (auf der verlängerten Fläche) im Bereich, den die Fläche wirklich abdeckt?
    Drehflächen: Lage entlang der eigenen Achse; Ebenen: Ausdehnung der Fläche."""
    V = f.triangles.reshape(-1, 3)
    s = f.surf
    if isinstance(s, RevSurf):
        a = (V - s.o) @ s.d
        ap = float((p - s.o) @ s.d)
        return a.min() - eps <= ap <= a.max() + eps
    lo, hi = V.min(0) - eps, V.max(0) + eps
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _axis_surface_end(model, faces, o, d, lo, hi):
    """Durchstoßpunkt der Bohrungsachse durch die angrenzende Fläche (Eintritt/Austritt).
    Bei schräg angeschnittenen Bohrungen (Planfläche, Kegel, Mantel, Radius) liegt der Rand der
    Bohrung nicht senkrecht zur Achse; der Bohrer trifft die Fläche mit der Achse dort, wo die
    (gedachte, durchgehende) Nachbarfläche die Achse schneidet. Ergebnis (lo, hi)."""
    tris = np.concatenate([model.faces[i].triangles.reshape(-1, 3, 3) for i in faces])
    key = np.round(tris.reshape(-1, 3), 4)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    E = np.sort(np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]]), axis=1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = uE[cnt == 1]
    if not len(bnd):
        return lo, hi
    P = np.zeros((inv.max() + 1, 3))
    P[inv.ravel()] = tris.reshape(-1, 3)
    vf = _vertex_faces(model)
    own = set(faces)
    # Randkurven als zusammenhängende Schleifen; die Schleife mit dem tiefsten bzw. höchsten Punkt
    # entlang der Achse ist der Austritt bzw. Eintritt (bei flachen Schrägbohrungen können sich
    # die Bereiche von Ein- und Austritt entlang der Achse überlappen).
    parent = {}

    def root(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for x, y in bnd:
        parent[root(int(x))] = root(int(y))
    comps = {}
    for x, y in bnd:
        comps.setdefault(root(int(x)), []).append((int(x), int(y)))
    hall = (P - o) @ d
    res = [lo, hi]
    for side in (0, 1):
        # Schleife, die den äußersten Punkt dieser Seite enthält
        best_c, best_h = None, None
        for cid, edges_ in comps.items():
            vids = {v for e in edges_ for v in e}
            hv = max(hall[list(vids)]) if side else min(hall[list(vids)])
            if best_h is None or (hv > best_h if side else hv < best_h):
                best_c, best_h = cid, hv
        if best_c is None:
            continue
        ext = hall.max() if side else hall.min()
        if abs(best_h - ext) > 1e-3:
            continue                    # dieses Ende hat keinen Rand (z. B. Bohrspitze im Sackloch)
        ids = np.array(sorted({v for e in comps[best_c] for v in e}))
        if len(ids) < 8:
            continue
        h = hall[ids]
        if h.max() - h.min() < 1e-6:
            continue                    # Rand senkrecht zur Achse – Wert stimmt bereits
        # nur ein Rand, der die Bohrungsachse umschließt (kein Durchbruch einer Querbohrung)
        e1_ = any_perp(d)
        e2_ = np.cross(d, e1_)
        Q = P[ids] - o
        ang = np.degrees(np.arctan2(Q @ e2_, Q @ e1_))
        if len(np.unique(np.floor(ang / 15.0).astype(int))) < 20:
            continue
        R = float(np.linalg.norm((P[ids] - o) - np.outer(h, d), axis=1).max())
        votes = {}
        for k in map(tuple, uk[ids]):
            for j in vf.get(k, ()):
                if j not in own:
                    votes[j] = votes.get(j, 0) + 1
        if not votes:
            continue
        # Reicht die Bohröffnung über eine Kante (z. B. Kegel → Kugel), trifft die Achse nur EINE der
        # Nachbarflächen wirklich: die, auf deren tatsächlichem Bereich der Durchstoßpunkt liegt.
        best = None
        for j in sorted(votes, key=votes.get, reverse=True):
            surf = model.faces[j].surf
            if surf is None:
                continue
            a_ = _line_surface_root(surf, o, d, h.min() - 0.25 * R, h.max() + 0.25 * R)
            if a_ is None or not (h.min() - 1e-3 <= a_ <= h.max() + 1e-3):
                continue
            if not _point_in_face_range(model.faces[j], o + a_ * d):
                continue
            if best is None or votes[j] > best[0]:
                best = (votes[j], a_)
        if best is not None:
            res[side] = best[1]
    return res[0], res[1]


def find_holes(model):
    thread_by_face = {}
    for ft in model.features.values():
        tf = ft.params.get("threaded-face")
        if tf and ft.params.get("thread-description"):
            thread_by_face[tf] = ft.params["thread-description"]
            thread_feat = getattr(model, "_thread_feat", {})
            thread_feat[tf] = ft.params
            model._thread_feat = thread_feat

    cand = []
    for i, f in enumerate(model.faces):
        s = f.surf
        if not isinstance(s, RevSurf):
            continue
        if f.ftype not in ("CYLINDER", "CONE"):
            continue
        if f.concavity != "1":
            continue
        full = f.face_angle is None or f.face_angle >= 359.0
        if f.ftype == "CYLINDER" and not full:
            continue
        cand.append(i)

    # nach Achslinie gruppieren
    groups = []
    for i in cand:
        s = model.faces[i].surf
        for g in groups:
            o, d = g["o"], g["d"]
            if _angle(d, s.d) < 0.05:
                v = s.o - o
                if np.linalg.norm(v - (v @ d) * d) < 1e-3:
                    g["faces"].append(i)
                    break
        else:
            groups.append({"o": s.o, "d": s.d, "faces": [i]})

    holes = []
    for g in groups:
        o, d = g["o"], g["d"]
        # Ausdehnung je Fläche entlang Achse
        spans = []
        for i in g["faces"]:
            V = model.faces[i].triangles.reshape(-1, 3)
            a = (V - o) @ d
            spans.append((a.min(), a.max(), i))
        spans.sort()
        # zusammenhängende Abschnitte bilden
        clusters = []
        for lo, hi, i in spans:
            if clusters and lo <= clusters[-1][1] + 0.02:
                clusters[-1][1] = max(clusters[-1][1], hi)
                clusters[-1][2].append(i)
            else:
                clusters.append([lo, hi, [i]])
        for lo, hi, faces in clusters:
            if not any(model.faces[i].ftype == "CYLINDER" for i in faces):
                continue
            lo, hi = _axis_surface_end(model, faces, o, d, lo, hi)
            eps = max(model.size * 2e-3, 0.2)
            p_lo, p_hi = o + (lo - eps) * d, o + (hi + eps) * d
            out_lo, out_hi = not inside(model, p_lo), not inside(model, p_hi)
            through = out_lo and out_hi
            if out_hi and not out_lo:
                entry_a, tool = hi, d
            elif out_lo and not out_hi:
                entry_a, tool = lo, -d
            else:
                # Durchgangsbohrung: Eintritt dort, wo die größte Stufe (Senkung) liegt,
                # sonst auf der Seite, die weiter von der Teile-Mittelachse entfernt ist
                entry_a, tool = hi, d
                cyl = [i for i in faces if model.faces[i].ftype == "CYLINDER"]
                big = max(cyl, key=lambda i: model.faces[i].surf.radius)
                small = min(cyl, key=lambda i: model.faces[i].surf.radius)
                Vb = model.faces[big].triangles.reshape(-1, 3)
                ab = float(((Vb - o) @ d).mean())
                if model.faces[big].surf.radius - model.faces[small].surf.radius > 1e-3:
                    entry_a, tool = (hi, d) if abs(ab - hi) < abs(ab - lo) else (lo, -d)
                else:
                    mo, md = main_axis(model)

                    def rdist(a):
                        q = o + a * d - mo
                        return np.linalg.norm(q - (q @ md) * md)
                    if rdist(lo) > rdist(hi) + 1e-6:
                        entry_a, tool = lo, -d
            dias = sorted({round(2 * model.faces[i].surf.radius, 4) for i in faces
                           if model.faces[i].ftype == "CYLINDER"}, reverse=True)
            feats = []
            for i in faces:
                ft = model.features.get(model.faces[i].feature)
                if ft and ft.name not in feats:
                    feats.append(ft.name)
            thread, tparams, tface = "", None, -1
            for i in faces:
                if model.faces[i].id in thread_by_face:
                    thread = thread_by_face[model.faces[i].id]
                    tparams = model._thread_feat.get(model.faces[i].id)
                    tface = i
            # Hauptbohrungen (Feature ohne Bohrungs-Typ) trotzdem aufnehmen
            holes.append(Hole(faces, o, tool, o + entry_a * d, hi - lo, dias, through, feats, thread,
                              tparams, tface))
    return holes


# ---------------------------------------------------------------------------
# Hauptachse, Winkelbezug, Bohrbild
# ---------------------------------------------------------------------------

def main_axis(model):
    ax = model.axes.get(model.main_axis)
    if ax is not None:
        return np.array(ax[0], float), unit(ax[1])
    V = model.tris.reshape(-1, 3)
    c = (V.min(axis=0) + V.max(axis=0)) / 2
    d = np.zeros(3)
    d[int(np.argmax(V.max(axis=0) - V.min(axis=0)))] = 1
    return c, d


def angle_frame(d, wcs):
    """Bezugsrichtungen (e1 = 0°, e2 = 90°) in der Ebene senkrecht zu d."""
    d = unit(d)
    for c in (wcs.R[0], wcs.R[1], wcs.R[2]):
        if abs(c @ d) < 0.9:
            e1 = unit(c - (c @ d) * d)
            break
    e2 = np.cross(d, e1)
    return e1, e2


def frame_name(e1, wcs):
    names = ["+X", "+Y", "+Z"]
    v = wcs.dir_to_wcs(e1)
    k = int(np.argmax(np.abs(v)))
    return ("+" if v[k] > 0 else "−") + "XYZ"[k]


def polar(p, o, d, e1, e2):
    v = p - o
    v = v - (v @ d) * d
    r = float(np.linalg.norm(v))
    th = round(math.degrees(math.atan2(v @ e2, v @ e1)), 4) % 360.0
    return r, th


@dataclass
class HoleGroup:
    kind: str                 # 'axial' | 'radial' | 'inclined'
    holes: list               # Indizes in model._holes
    level: float              # Lage entlang der Hauptachse
    radius: float             # Lochkreisradius (axial/schräg: Eintritt) bzw. 0 (radial)
    angles: list              # Winkel je Bohrung (°)
    tool_sign: int = 1
    tilt: float = 0.0         # Schrägbohrung: Winkel Bohrungsachse zur Hauptachse (°)
    offsets: list = None      # radial: Y-Versatz je Bohrung (außermittig, 0 = durch die Mitte)


def hole_pattern(model, wcs):
    holes = getattr(model, "_holes", None)
    if holes is None:
        holes = model._holes = find_holes(model)
    o, d = main_axis(model)
    e1, e2 = angle_frame(d, wcs)
    groups = []
    for i, h in enumerate(holes):
        ang = _angle(h.axis_d, d)
        v = h.axis_p - o
        dist_axis = np.linalg.norm(v - (v @ d) * d)
        if ang < 0.5 and dist_axis > 1e-3:
            r, th = polar(h.entry, o, d, e1, e2)
            lvl = float((h.entry - o) @ d)
            sg = 1 if h.axis_d @ d > 0 else -1
            for g in groups:
                if (g.kind == "axial" and g.tool_sign == sg and abs(g.level - lvl) < 0.5
                        and abs(g.radius - r) < 0.05):
                    g.holes.append(i)
                    g.angles.append(th)
                    break
            else:
                groups.append(HoleGroup("axial", [i], lvl, r, [th], sg))
        elif abs(ang - 90) < 0.5:
            # Lage am Umfang aus der Werkzeugrichtung (C), außermittig mit Y-Versatz
            ei = entry_info(model, h, wcs)
            th = round(ei["c"], 4) % 360.0
            y = round(ei["y"], 4)
            lvl = float((h.entry - o) @ d)
            for g in groups:
                if g.kind == "radial" and abs(g.level - lvl) < 0.5:
                    g.holes.append(i)
                    g.angles.append(th)
                    g.offsets.append(y)
                    break
            else:
                groups.append(HoleGroup("radial", [i], lvl, 0.0, [th], offsets=[y]))
        elif 0.5 <= ang <= 179.5:
            # Schrägbohrung: Lage über Eintrittspunkt (Höhe, Radius, Winkel) + Neigung
            r, th = polar(h.entry, o, d, e1, e2)
            lvl = float((h.entry - o) @ d)
            tilt = round(min(ang, 180.0 - ang), 3)
            for g in groups:
                if (g.kind == "inclined" and abs(g.level - lvl) < 0.5 and abs(g.radius - r) < 0.05
                        and abs(g.tilt - tilt) < 0.05):
                    g.holes.append(i)
                    g.angles.append(th)
                    break
            else:
                groups.append(HoleGroup("inclined", [i], lvl, r, [th], 1, tilt))
    for g in groups:
        order = np.argsort(g.angles)
        g.holes = [g.holes[k] for k in order]
        g.angles = [g.angles[k] for k in order]
        if g.offsets is not None:
            g.offsets = [g.offsets[k] for k in order]
    groups.sort(key=lambda g: (g.kind, g.level))
    return groups, (o, d, e1, e2)


def entry_info(model, h, wcs):
    """Eintrittspunkt einer Bohrung für die Maschine (Dreh-Fräszentrum mit C-, Y- und B-Achse).
    Die C-Lage wird so gewählt, dass die Bohrung von „oben“ (+X der Maschine) angefahren wird:
    X = Ø-Wert in dieser Lage, Y = Versatz außermittiger Bohrungen, Z = Lage auf der Längsachse,
    B = Neigung der Bohrungsachse zur Längsachse (90° = radial, 0° = axial)."""
    o, d = main_axis(model)
    e1, e2 = angle_frame(d, wcs)
    p = np.asarray(h.entry, float)
    q = p - o
    qr = q - (q @ d) * d
    tool = -unit(h.axis_d)              # Werkzeugrichtung: von außen ins Material
    tr = tool - (tool @ d) * d
    if np.linalg.norm(tr) > 1e-6:
        eo = unit(-tr)                  # von dort kommt das Werkzeug
    elif np.linalg.norm(qr) > 1e-6:
        eo = unit(qr)                   # axiale Bohrung: Lage am Lochkreis
    else:
        eo = e1
    ey = np.cross(d, eo)
    xr = float(qr @ eo)
    y = float(qr @ ey)
    if abs(y) < 1e-6:
        y = 0.0
    c = math.degrees(math.atan2(eo @ e2, eo @ e1)) % 360.0
    pw = wcs.to_wcs(p)
    b = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(tool @ d))))))
    # Bohrtiefe der Spitze bis Mitte X0 (Ebene durch die Längsachse senkrecht zur Anfahrrichtung)
    down = -float(tool @ eo)
    depth_x0 = xr / down if down > 1e-6 else None
    d_nom = min(h.diameters) if h.diameters else 0.0
    tip_len = d_nom / 2 / math.tan(math.radians(59)) if d_nom else 0.0     # Spiralbohrer 118°
    # Anschnitt: Winkel zwischen Bohrer und Fläche am Eintritt (90° = senkrecht)
    hit = None
    try:
        _, n = entry_surface(model, h)
        if n is not None:
            hit = 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, abs(float(tool @ unit(n)))))))
    except Exception:
        hit = None
    return {"point": p, "dia": 2 * xr, "y": y, "z": float(pw[2]), "c": c, "b": b, "b_top": 90.0 - b,
            "wcs": pw, "tool": wcs.dir_to_wcs(tool), "depth_x0": depth_x0, "tip118": tip_len, "hit": hit}


def hole_details(model, h):
    """Senkung/Stufe am Eintritt und Bohrspitze am Grund einer Bohrung.
    Ergebnis: dict(sinks=[(art, Ø, Winkel oder None, Tiefe ab Eintritt)], tip=(Winkel, Länge) oder None,
    full=Tiefe bis voller Ø (ohne Spitze), bore=Nenn-Ø)."""
    o, d = h.axis_p, unit(h.axis_d)
    ae = float((h.entry - o) @ d)
    info = []
    for i in h.faces:
        f = model.faces[i]
        V = f.triangles.reshape(-1, 3) - o
        a = V @ d
        rho = np.linalg.norm(V - np.outer(a, d), axis=1)
        info.append((i, f, a.min(), a.max(), rho.min(), rho.max()))
    cyl = [x for x in info if x[1].ftype == "CYLINDER"]
    if not cyl:
        return {"sinks": [], "tip": None, "full": h.depth, "bore": None}
    r_bore = min(x[1].surf.radius for x in cyl)
    sinks, tip = [], None
    for i, f, a0, a1, r0, r1 in info:
        if f.ftype == "CONE":
            half = getattr(f.surf, "half_angle", None)
            if r0 < 0.05 * r_bore + 1e-3:
                L = a1 - a0
                tip = (2 * half if half else 2 * math.degrees(math.atan2(r1, L)), L)
            elif r1 > r_bore + 1e-3 and abs(a1 - ae) < max(r1, 0.5):
                sinks.append(("Senkung", 2 * r1, 2 * half if half else None, ae - a0))
        elif f.ftype == "CYLINDER" and f.surf.radius > r_bore + 1e-3:
            sinks.append(("Stufe", 2 * f.surf.radius, None, ae - a0))
    sinks.sort(key=lambda x: -x[1])
    full = h.depth - (tip[1] if tip else 0.0)
    return {"sinks": sinks, "tip": tip, "full": full, "bore": 2 * r_bore}


def entry_surface(model, h):
    """Fläche, auf der der Bohrungseintritt liegt: (Flächenindex, Normale) oder (None, None)."""
    p = np.asarray(h.entry, float)
    R = max(h.diameters) / 2 if h.diameters else model.size * 0.01
    best = None
    for j, f in enumerate(model.faces):
        if j in h.faces or f.surf is None:
            continue
        V = f.triangles.reshape(-1, 3)
        if np.linalg.norm(V - p, axis=1).min() > 3 * R:
            continue
        q = f.surf.project(p[None, :])[0]
        dist = float(np.linalg.norm(q - p))
        if best is None or dist < best[0]:
            best = (dist, j, f.surf.normals(q[None, :])[0])
    if best is None or best[0] > max(1e-3, R * 1e-2):
        return None, None
    return best[1], best[2]


def pitch_text(angles):
    if len(angles) < 2:
        return ""
    a = sorted(angles)
    diffs = [(a[(k + 1) % len(a)] - a[k]) % 360.0 for k in range(len(a))]
    if max(diffs) - min(diffs) < 0.01:
        return f"{len(a)}× {fmt(diffs[0], 3)}° gleich geteilt"
    return "Teilung: " + " / ".join(fmt(x, 3) + "°" for x in diffs)


def inward_faces(model):
    """Flächen, deren Dreiecks-Umlaufsinn nach innen zeigt (für die Schnittdarstellung)."""
    cached = getattr(model, "_inward", None)
    if cached is not None:
        return cached
    eps = max(model.size * 1e-4, 0.01)
    res = set()
    for i, f in enumerate(model.faces):
        T = f.triangles
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        a = np.linalg.norm(n, axis=1)
        j = int(np.argmax(a))
        if a[j] <= 0:
            continue
        p = T[j].mean(axis=0) + n[j] / a[j] * eps
        if inside(model, p):
            res.add(i)
    model._inward = res
    return res


# ---------------------------------------------------------------------------
# Scharfe Ecke: Übergangsradius / Fase rechnerisch entfernen
# ---------------------------------------------------------------------------

def _edge_faces(model):
    ef = getattr(model, "_edge_faces", None)
    if ef is None:
        ef = {}
        for i, f in enumerate(model.faces):
            for e in f.edges:
                ef.setdefault(e, set()).add(i)
        model._edge_faces = ef
    return ef


def _profile_line(model, j, o, d):
    """Profil-Gerade (a, rho) einer Nachbarfläche im Achssystem (o, d) oder None."""
    s = model.faces[j].surf
    if isinstance(s, PlaneSurf):
        if abs(abs(s.n @ d) - 1) < 1e-4:
            return np.array([(s.p - o) @ d, 0.0]), np.array([0.0, 1.0])
        return None
    if isinstance(s, RevSurf) and s.profile == "line":
        if _angle(s.d, d) > 0.05:
            return None
        v = s.o - o
        if np.linalg.norm(v - (v @ d) * d) > 2e-3:
            return None
        sg = 1.0 if s.d @ d > 0 else -1.0
        p0 = np.array([(s.o - o) @ d + sg * s.p0[0], s.p0[1]])
        t = np.array([sg * s.t[0], s.t[1]])
        return p0, t / np.linalg.norm(t)
    return None


def sharp_corner(model, fi):
    """Theoretische scharfe Ecke ohne die Fläche fi (Radius oder Fase am Drehteil).
    Liefert dict mit a, rho (Profil), o, d (Achse), Nachbarflächen, Kantenpunkten, Winkel."""
    f = model.faces[fi]
    s = f.surf
    if not isinstance(s, RevSurf):
        raise ValueError("Nur bei Radien oder Fasen an Drehflächen möglich.")
    o, d = s.o, s.d
    ef = _edge_faces(model)
    cand = {}
    for eid in f.edges:
        for j in ef.get(eid, ()):
            if j == fi or j in cand:
                continue
            pl = _profile_line(model, j, o, d)
            if pl is None:
                continue
            e = model.edges.get(eid)
            ep = None
            if e is not None:
                pts = e.points
                a = (pts - o) @ d
                r = np.linalg.norm(pts - o - np.outer(a, d), axis=1)
                ep = np.array([a.mean(), r.mean()])
            cand[j] = (pl, ep)
    keys = list(cand)
    best = None
    V = f.triangles.reshape(-1, 3)
    a = (V - o) @ d
    r = np.linalg.norm(V - o - np.outer(a, d), axis=1)
    ref = np.array([a.mean(), r.mean()])
    for i1 in range(len(keys)):
        for i2 in range(i1 + 1, len(keys)):
            (p1, t1), e1 = cand[keys[i1]]
            (p2, t2), e2 = cand[keys[i2]]
            den = t1[0] * t2[1] - t1[1] * t2[0]
            if abs(den) < 1e-9:
                continue
            w = p2 - p1
            u = (w[0] * t2[1] - w[1] * t2[0]) / den
            q = p1 + u * t1
            if q[1] < -1e-6:
                continue
            score = np.linalg.norm(q - ref)
            if best is None or score < best[0]:
                ang = math.degrees(math.acos(min(1.0, abs(float(t1 @ t2)))))
                best = (score, q, keys[i1], keys[i2], e1, e2, ang)
    if best is None:
        if len(keys) >= 2:
            raise ValueError("Die angrenzenden Flächen laufen parallel – es gibt keine scharfe Ecke "
                             "(z. B. Vollradius im Einstich).")
        raise ValueError("Keine passenden Nachbarflächen (Zylinder, Kegel, Planfläche) gefunden.")
    _, q, j1, j2, e1, e2, ang = best
    return {"a": float(q[0]), "rho": float(max(q[1], 0.0)), "o": o, "d": d, "faces": (j1, j2),
            "edge_pts": (e1, e2), "angle": ang, "face": fi}


# ---------------------------------------------------------------------------
# Radius (Torus) im Drehprofil: Mittelpunkt, Radius, Anfangs-/Endpunkt
# ---------------------------------------------------------------------------

def arc_profile(model, fi, wcs=None):
    """Profil-Kreisbogen einer Torusfläche im Achssystem (a, rho).
    Mittelpunkt exakt über die Normalen der tangentialen Nachbarflächen, sonst Ausgleichskreis."""
    f = model.faces[fi]
    s = f.surf
    if not isinstance(s, RevSurf):
        raise ValueError("Keine Drehfläche.")
    o, d = s.o, s.d
    ef = _edge_faces(model)
    ends = []
    for eid in f.edges:
        e = model.edges.get(eid)
        if e is None or e.etype != "CIRCLE" or e.normal is None or _angle(e.normal, d) > 0.05:
            continue
        P = e.points
        a = (P - o) @ d
        r = np.linalg.norm(P - o - np.outer(a, d), axis=1)
        if np.ptp(a) > 1e-3 or np.ptp(r) > 1e-3:
            continue
        p = np.array([a.mean(), r.mean()])
        nrm = None
        for j in ef.get(eid, ()):
            if j == fi:
                continue
            pl = _profile_line(model, j, o, d)
            if pl is not None:
                nrm = np.array([-pl[1][1], pl[1][0]])
                break
        if not any(np.linalg.norm(p - q[0]) < 1e-6 for q in ends):
            ends.append((p, nrm))
    V = f.triangles.reshape(-1, 3)
    av = (V - o) @ d
    rv = np.linalg.norm(V - o - np.outer(av, d), axis=1)
    cfit, rfit, _ = _fit_circle_2d(av, rv)
    exact = False
    c, r = cfit, rfit
    if len(ends) >= 2:
        (p1, n1), (p2, n2) = ends[0], ends[1]
        if n1 is not None and n2 is not None:
            den = n1[0] * n2[1] - n1[1] * n2[0]
            if abs(den) > 1e-9:
                w = p2 - p1
                t = (w[0] * n2[1] - w[1] * n2[0]) / den
                cc = p1 + t * n1
                if abs(np.linalg.norm(cc - p1) - np.linalg.norm(cc - p2)) < 1e-3:
                    c, r, exact = cc, float(np.linalg.norm(cc - p1)), True
            else:
                # parallele Flanken (Vollradius): Mitte zwischen den Endpunkten
                cc = (p1 + p2) / 2
                if np.linalg.norm(cc - cfit) < max(0.05, 0.2 * rfit):
                    c, r, exact = cc, float(np.linalg.norm(p1 - p2) / 2), True
        if not exact:
            # Kreis durch beide Endpunkte mit Ausgleichsradius, Seite wie Ausgleichskreis
            m_ = (p1 + p2) / 2
            h = p2 - p1
            L = np.linalg.norm(h)
            if L > 1e-9 and rfit >= L / 2:
                nn = np.array([-h[1], h[0]]) / L
                k = math.sqrt(max(rfit ** 2 - (L / 2) ** 2, 0.0))
                c1, c2 = m_ + k * nn, m_ - k * nn
                c = c1 if np.linalg.norm(c1 - cfit) < np.linalg.norm(c2 - cfit) else c2
                r = rfit
    if len(ends) == 1 and abs(cfit[1]) < max(1e-2 * rfit, 2e-3):
        # Kugel mit Mittelpunkt auf der Achse (z. B. Spitze R5): zweiter Endpunkt ist der Pol auf der Achse
        p1, n1 = ends[0]
        c = np.array([cfit[0], 0.0])
        if n1 is not None and abs(n1[0]) > 1e-9:
            c = np.array([p1[0] - p1[1] * n1[0] / n1[1], 0.0]) if abs(n1[1]) > 1e-9 else c
        r = float(np.linalg.norm(p1 - c))
        sgn = 1.0 if av.mean() > c[0] else -1.0
        ends.append((np.array([c[0] + sgn * r, 0.0]), None))
        exact = n1 is not None
    pts = sorted([e[0] for e in ends], key=lambda p: -p[0])
    return {"o": o, "d": d, "c": np.asarray(c, float), "r": float(r), "ends": pts, "exact": exact,
            "concave": f.concavity == "1"}


# ---------------------------------------------------------------------------
# Gewindebohrung: Gewinde- und Kernlochtiefe (aus den CAD-Daten)
# ---------------------------------------------------------------------------

def _fnum(x):
    try:
        return float(str(x).replace(",", "."))
    except (TypeError, ValueError):
        return None


def thread_info(model, h):
    """Tiefen einer Gewindebohrung, gemessen ab Bohrungseintritt entlang der Werkzeugachse."""
    if not h.thread or h.thread_face < 0:
        return None
    tp = h.thread_params or {}
    ax = unit(h.axis_d)
    depth = lambda P: (h.entry - P) @ ax                     # >0 = ins Material
    tf = model.faces[h.thread_face]
    tv = depth(tf.triangles.reshape(-1, 3))
    c0, c1 = float(tv.min()), float(tv.max())            # Kernloch-Zylinder (Gewindefläche)
    t0 = c0
    se = model.edges.get(tp.get("start-edge", ""))
    if se is not None and len(se.points):
        t0 = float(depth(se.points).mean())
    L = _fnum(tp.get("thread-depth"))
    if L is None or L <= 0:
        L = c1 - t0
    # Richtung: Gewinde läuft vom Startrand in den Zylinder hinein
    if abs(t0 - c1) < abs(t0 - c0):
        t_from, t_to = t0 - L, t0
    else:
        t_from, t_to = t0, t0 + L
    tip = 0.0
    for i in h.faces:
        tip = max(tip, float(depth(model.faces[i].triangles.reshape(-1, 3)).max()))
    P = _fnum(tp.get("thread-pitch"))
    import re as _re
    mm = _re.match(r"\s*M\s*([\d.,]+)", h.thread)
    major = _fnum(mm.group(1)) if mm else None
    core = _fnum(tp.get("tapped-diameter")) or (2 * tf.surf.radius if isinstance(tf.surf, RevSurf) else None)
    over = c1 - t_to
    return {"thread": h.thread, "pitch": P, "major": major, "core": core,
            "t_from": t_from, "t_to": t_to, "length": t_to - t_from,
            "core_from": c0, "core_end": c1, "tip": tip, "over": over,
            "over_p": (over / P) if P else None, "blind": not h.through,
            "pt": lambda t: h.entry - t * ax}


def split_face_by_plane(V, T, F, fi, p0, n, eps=1e-7):
    """Dreiecke der Fläche fi an der Ebene (p0, n) auftrennen, damit Farbgrenzen scharf liegen."""
    n = unit(np.asarray(n, float))
    k = np.where(F == fi)[0]
    if len(k) == 0:
        return V, T, F
    dist = (V - p0) @ n
    D = dist[T[k]]
    cross = (D.min(axis=1) < -eps) & (D.max(axis=1) > eps)
    if not cross.any():
        return V, T, F
    V = list(V)
    cache = {}

    def cut(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            da, db = dist[a], dist[b]
            t = da / (da - db)
            cache[key] = len(V)
            V.append(np.asarray(V[a]) + t * (np.asarray(V[b]) - np.asarray(V[a])))
        return cache[key]

    newT, newF = [], []
    for ti in k[cross]:
        a, b, c = T[ti]
        d = [dist[a], dist[b], dist[c]]
        idx = [a, b, c]
        # das einzelne Eck auf der einen Seite finden
        s = [1 if x > eps else (-1 if x < -eps else 0) for x in d]
        for r in range(3):
            i0, i1, i2 = idx[r], idx[(r + 1) % 3], idx[(r + 2) % 3]
            s0, s1, s2 = s[r], s[(r + 1) % 3], s[(r + 2) % 3]
            if s0 != 0 and s1 != s0 and s2 != s0:
                break
        if s1 == 0 or s2 == 0:
            # ein Eck liegt genau auf der Ebene: in zwei Dreiecke teilen
            on, other = (i1, i2) if s1 == 0 else (i2, i1)
            m = cut(i0, other)
            if s1 == 0:
                newT += [[i0, i1, m], [i1, i2, m]]
            else:
                newT += [[i0, m, i2], [m, i1, i2]]
            newF += [fi, fi]
        else:
            m1, m2 = cut(i0, i1), cut(i0, i2)
            newT += [[i0, m1, m2], [m1, i1, i2], [m1, i2, m2]]
            newF += [fi, fi, fi]
    keep = np.ones(len(T), bool)
    keep[k[cross]] = False
    V = np.array(V, float)
    T = np.concatenate([T[keep], np.array(newT, dtype=T.dtype)])
    F = np.concatenate([F[keep], np.array(newF, dtype=F.dtype)])
    return V, T, F


# ---------------------------------------------------------------------------
# Drehteil: Profil (Kontur) und Maßliste
# ---------------------------------------------------------------------------

def _coax_main(model, s, mo, md, tol=2e-3):
    if not isinstance(s, RevSurf) or _angle(s.d, md) > 0.05:
        return False
    v = s.o - mo
    return np.linalg.norm(v - (v @ md) * md) < tol


def lathe_segments(model):
    """Profil-Elemente aller koaxialen Drehflächen und Planflächen im (a, rho)-System der Hauptachse."""
    mo, md = main_axis(model)
    u1 = any_perp(md)
    u2 = np.cross(md, u1)
    segs = []
    for i, f in enumerate(model.faces):
        s = f.surf
        V = f.triangles.reshape(-1, 3)
        a = (V - mo) @ md
        R = V - mo - np.outer(a, md)
        rho = np.linalg.norm(R, axis=1)
        # nur Flächen, die (fast) rundum laufen – Nutgründe, Schlüsselflächen usw. gehören nicht zur Drehkontur
        th = np.degrees(np.arctan2(R @ u2, R @ u1))
        if len(np.unique(np.floor(th / 10.0).astype(int))) < 18:
            continue
        if isinstance(s, PlaneSurf):
            if abs(abs(unit(s.n) @ md) - 1) > 1e-4:
                continue
            a0 = float(np.median(a))
            radii = [model.edges[e].radius for e in f.edges
                     if e in model.edges and model.edges[e].etype == "CIRCLE" and model.edges[e].radius
                     and model.edges[e].normal is not None and _angle(model.edges[e].normal, md) < 0.05
                     and np.linalg.norm((model.edges[e].center - mo) - ((model.edges[e].center - mo) @ md) * md) < 2e-3]
            r0, r1 = (min(radii), max(radii)) if len(radii) >= 1 else (rho.min(), rho.max())
            if len(radii) == 1:
                r0, r1 = (rho.min(), radii[0]) if abs(radii[0] - rho.max()) < abs(radii[0] - rho.min()) else (radii[0], rho.max())
            if r0 < 1e-3 + 0 and rho.min() < 1e-3:
                r0 = 0.0
            segs.append(dict(face=i, kind="PLANE", p0=np.array([a0, r0]), p1=np.array([a0, r1]),
                             conc=f.concavity))
            continue
        if not _coax_main(model, s, mo, md):
            continue
        sg = 1.0 if s.d @ md > 0 else -1.0
        off = float((s.o - mo) @ md)
        if s.kind == "CYLINDER":
            segs.append(dict(face=i, kind="CYL", p0=np.array([a.min(), s.radius]), p1=np.array([a.max(), s.radius]),
                             conc=f.concavity))
        elif s.kind == "CONE":
            # Endpunkte auf der exakten Profilgeraden
            p0 = np.array([off + sg * s.p0[0], s.p0[1]])
            t = unit(np.array([sg * s.t[0], s.t[1]]))
            def on(av):
                return p0 + ((av - p0[0]) / t[0]) * t if abs(t[0]) > 1e-9 else np.array([av, p0[1]])
            if abs(t[0]) > 1e-6:
                q0, q1 = on(a.min()), on(a.max())
            else:
                q0, q1 = np.array([a.mean(), rho.min()]), np.array([a.mean(), rho.max()])
            segs.append(dict(face=i, kind="CONE", p0=q0, p1=q1, conc=f.concavity, angle=s.half_angle))
        elif s.kind == "TORUS":
            try:
                ap = arc_profile(model, i)
            except Exception:
                continue
            if len(ap["ends"]) < 2:
                continue
            c = np.array([off + sg * ap["c"][0], ap["c"][1]])
            e0 = np.array([off + sg * ap["ends"][0][0], ap["ends"][0][1]])
            e1 = np.array([off + sg * ap["ends"][1][0], ap["ends"][1][1]])
            # Bogenmitte (für Richtung) aus den Flächenpunkten
            mid = np.array([a.mean(), rho.mean()])
            segs.append(dict(face=i, kind="ARC", p0=e0, p1=e1, c=c, r=ap["r"], mid=mid, conc=f.concavity))
    return segs, (mo, md)


def _chains(segs, tol=5e-3):
    nodes = []
    def node(p):
        for k, q in enumerate(nodes):
            if np.linalg.norm(q - p) < tol:
                return k
        nodes.append(np.asarray(p, float))
        return len(nodes) - 1
    edges, seen = [], set()
    for s in segs:
        n0, n1 = node(s["p0"]), node(s["p1"])
        if n0 == n1:
            continue
        key = (min(n0, n1), max(n0, n1), s["kind"])
        if key in seen:
            continue
        seen.add(key)
        edges.append((n0, n1, s))
    adj = {}
    for k, (n0, n1, s) in enumerate(edges):
        adj.setdefault(n0, []).append(k)
        adj.setdefault(n1, []).append(k)
    used, chains = set(), []
    starts = [n for n, ks in adj.items() if len(ks) == 1] + list(adj)
    for st in starts:
        if all(k in used for k in adj[st]):
            continue
        path, cur = [], st
        while True:
            nxt = [k for k in adj[cur] if k not in used]
            if not nxt:
                break
            k = nxt[0]
            used.add(k)
            n0, n1, s = edges[k]
            fwd = n0 == cur
            path.append((s, fwd))
            cur = n1 if fwd else n0
            if cur == st:
                break
        if path:
            chains.append(path)
    return chains, nodes


def _oriented(s, fwd):
    return (s["p0"], s["p1"]) if fwd else (s["p1"], s["p0"])


def lathe_contours(model):
    """Außen- und Innenkontur als geordnete Elementlisten (a, rho). Liefert dict mit 'outer','inner','ends'."""
    segs, (mo, md) = lathe_segments(model)
    if not segs:
        return None
    chains, nodes = _chains(segs)
    if not chains:
        return None
    path = max(chains, key=len)
    pts = [_oriented(s, f)[0] for s, f in path] + [_oriented(*path[-1])[1]]
    P = np.array(pts)
    amin, amax = P[:, 0].min(), P[:, 0].max()
    n = len(path)
    closed = np.linalg.norm(P[0] - P[-1]) < 5e-3

    def idx_of(cond_pts, pick):
        cand = [k for k in range(len(pts) - (1 if closed else 0)) if cond_pts(pts[k])]
        return pick(cand, key=lambda k: pts[k][1]) if cand else None
    A = idx_of(lambda p: abs(p[0] - amax) < 1e-3, max)        # vorne außen
    B = idx_of(lambda p: abs(p[0] - amin) < 1e-3, max)        # hinten außen
    Ai = idx_of(lambda p: abs(p[0] - amax) < 1e-3, min)
    Bi = idx_of(lambda p: abs(p[0] - amin) < 1e-3, min)

    def sub(i, j):
        """Elemente von Knoten i nach j (in Laufrichtung bzw. rückwärts) – beide Varianten."""
        out = []
        if closed:
            m = n
            fw, k = [], i
            while k != j:
                fw.append((path[k % m][0], path[k % m][1]))
                k = (k + 1) % m
            bw, k = [], i
            while k != j:
                k = (k - 1) % m
                s, f = path[k]
                bw.append((s, not f))
            out = [fw, bw]
        else:
            if i <= j:
                out = [[(path[k][0], path[k][1]) for k in range(i, j)]]
            else:
                out = [[(path[k][0], not path[k][1]) for k in range(i - 1, j - 1, -1)]]
        return out

    def mean_rho(elems):
        if not elems:
            return 0.0
        return float(np.mean([(_oriented(s, f)[0][1] + _oriented(s, f)[1][1]) / 2 for s, f in elems]))

    outer = inner = []
    if A is not None and B is not None:
        cands = [c for c in sub(A, B) if c]
        outer = max(cands, key=mean_rho) if cands else []
    if Ai is not None and Bi is not None and P[Ai][1] > 1e-3:
        cands = [c for c in sub(Ai, Bi) if c]
        inner = min(cands, key=mean_rho) if cands else []
        outer_faces = {id(s) for s, _ in outer}
        inner = [(s, f) for s, f in inner if id(s) not in outer_faces]
    used = {id(s) for s, _ in outer} | {id(s) for s, _ in inner}
    ends = [s for s, _ in path if id(s) not in used]
    return {"outer": outer, "inner": inner, "ends": ends, "axis": (mo, md), "segs": segs}


def contour_rows(model, wcs, elems):
    """Elemente in Zeilen (Z, X als Ø, R, Mittelpunkt, Richtung) im Nullpunkt-KS umrechnen."""
    mo, md = main_axis(model)
    zsign = 1.0 if wcs.dir_to_wcs(md)[2] >= 0 else -1.0
    Z = lambda a: float(wcs.to_wcs(mo + a * md)[2])
    rows = []
    if not elems:
        return rows
    s0, f0 = elems[0]
    p_start = _oriented(s0, f0)[0]
    rows.append(dict(elem="Start", z=Z(p_start[0]), x=2 * p_start[1], face=None))
    for s, f in elems:
        q0, q1 = _oriented(s, f)
        r = dict(z=Z(q1[0]), x=2 * q1[1], face=s["face"])
        k = s["kind"]
        if k == "PLANE":
            r["elem"] = "Plan"
        elif k == "CYL":
            r["elem"] = "Gerade Z"
        elif k == "CONE":
            L = float(np.linalg.norm(q1 - q0))
            r["elem"] = "Fase" if L < 3.0 else "Kegel"
            r["info"] = f"{fmt(s['angle'], 3)}° zur Achse, Länge {fmt(L)}"
            if L >= 3.0:
                r["info"] += " · " + taper_text(taper_info(s["angle"], 2 * q0[1], 2 * q1[1], abs(q1[0] - q0[0])))
        elif k == "ARC":
            r["elem"] = "Radius"
            r["r"] = s["r"]
            r["cz"], r["cx"] = Z(s["c"][0]), 2 * s["c"][1]
            # Drehsinn im Bild „Z nach rechts, X nach oben“
            v0 = np.array([Z(q0[0]) - r["cz"], q0[1] - s["c"][1]])
            vm = np.array([Z(s["mid"][0]) - r["cz"], s["mid"][1] - s["c"][1]])
            v1 = np.array([r["z"] - r["cz"], q1[1] - s["c"][1]])
            cr = v0[0] * v1[1] - v0[1] * v1[0]
            cm = v0[0] * vm[1] - v0[1] * vm[0]
            ccw = (cm > 0) if abs(cr) < 1e-9 else (cr > 0)
            r["dir"] = "gegen Uhrzeiger (G3)" if ccw else "im Uhrzeiger (G2)"
        rows.append(r)
    return rows


def dimension_list(model, wcs):
    """Maßliste: alle Ø, Planflächen, Fasen/Kegel, Radien des Drehteils (Nullpunkt-KS)."""
    segs, (mo, md) = lathe_segments(model)
    Z = lambda a: float(wcs.to_wcs(mo + a * md)[2])
    out = []
    for s in segs:
        f = model.faces[s["face"]]
        z0, z1 = sorted((Z(s["p0"][0]), Z(s["p1"][0])))
        side = "innen" if f.concavity == "1" else "außen"
        if s["kind"] == "CYL":
            out.append(dict(typ="Durchmesser", wert=f"Ø {fmt(2 * s['p0'][1])}", z0=z0, z1=z1, L=z1 - z0,
                            info=side, face=s["face"], key=(0, -2 * s['p0'][1])))
        elif s["kind"] == "PLANE":
            r0, r1 = sorted((s["p0"][1], s["p1"][1]))
            out.append(dict(typ="Planfläche", wert=f"Z {fmt(z0)}", z0=z0, z1=z0, L=0.0,
                            info=f"Ø {fmt(2 * r0)} … Ø {fmt(2 * r1)}", face=s["face"], key=(1, 0)))
        elif s["kind"] == "CONE":
            L = float(np.linalg.norm(s["p1"] - s["p0"]))
            d0, d1 = 2 * s["p0"][1], 2 * s["p1"][1]
            inf = f"Ø {fmt(min(d0, d1))} … Ø {fmt(max(d0, d1))} · {side}"
            if L >= 3.0:
                inf += " · " + taper_text(taper_info(s["angle"], d0, d1, z1 - z0))
            out.append(dict(typ="Fase" if L < 3.0 else "Kegel", wert=f"{fmt(s['angle'], 3)}°", z0=z0, z1=z1, L=z1 - z0,
                            info=inf, face=s["face"], key=(2, 0)))
        elif s["kind"] == "ARC":
            out.append(dict(typ="Radius", wert=f"R {fmt(s['r'])}", z0=z0, z1=z1, L=z1 - z0,
                            info=f"Mitte Z {fmt(Z(s['c'][0]))} · X Ø {fmt(2 * s['c'][1])} · "
                                 + ("Hohlkehle/Einstich" if f.concavity == "1" else "Rundung"),
                            face=s["face"], key=(3, 0)))
    out.sort(key=lambda d: (-d["z1"], d["key"]))
    return out


# ---------------------------------------------------------------------------
# Fertigungsmerkmale: Kegel, Einstiche, Werkzeug-Erreichbarkeit, Fräsmerkmale
# ---------------------------------------------------------------------------

STD_TAPERS = [  # (Name, Verhältnis 1:x, Bezugs-Ø groß oder None)
    ("Morsekegel MK0", 19.212, 9.045), ("Morsekegel MK1", 20.047, 12.065), ("Morsekegel MK2", 20.020, 17.780),
    ("Morsekegel MK3", 19.922, 23.825), ("Morsekegel MK4", 19.254, 31.267), ("Morsekegel MK5", 19.002, 44.399),
    ("Morsekegel MK6", 19.180, 63.348), ("Metrischer Kegel 1:20", 20.0, None), ("Steilkegel 7:24 (SK/ISO/BT)", 24 / 7, None),
    ("HSK 1:10", 10.0, None), ("Kegelstift 1:50", 50.0, None), ("Kegel 1:30", 30.0, None), ("Kegel 1:16 (Rohrgewinde)", 16.0, None),
    ("Kegel 1:5", 5.0, None), ("Kegel 1:3", 3.0, None),
]


def taper_info(half_angle, d0, d1, L):
    """Kegel nach Norm: Verhältnis 1:x, Kegelwinkel, D/d/L und – falls passend – Normkegel."""
    t = math.tan(math.radians(half_angle))
    D, d = max(d0, d1), min(d0, d1)
    out = {"D": D, "d": d, "L": L, "angle": 2 * half_angle, "half": half_angle,
           "x": (1 / (2 * t)) if t > 1e-9 else None, "norm": ""}
    if out["x"] is None:
        return out
    best = None
    for name, x, Dn in STD_TAPERS:
        dev = abs(out["x"] - x) / x
        if dev < 0.006:
            score = dev + (0 if Dn is None else min(abs(D - Dn), abs(d - Dn)) / 50.0)
            if Dn is not None and min(abs(D - Dn), abs(d - Dn)) > 3.0:
                continue
            if best is None or score < best[0]:
                best = (score, name)
    if best:
        out["norm"] = best[1]
    return out


def taper_text(ti):
    if ti.get("x") is None:
        return ""
    x = ti["x"]
    ratio = f"Kegel 1:{fmt(x, 3)} · " if ti["half"] <= 20.0 else ""
    s = f"{ratio}Kegelwinkel {fmt(ti['angle'], 3)}° (α/2 {fmt(ti['half'], 3)}°) · D {fmt(ti['D'])} d {fmt(ti['d'])} L {fmt(ti['L'])}"
    if ti["norm"]:
        s += f" · ≈ {ti['norm']}"
    return s


def _profile_samples(elems, n_arc=12):
    """Kontur als Punktfolge (a, rho, Element-Index)."""
    pts = []
    for k, (s, f) in enumerate(elems):
        q0, q1 = _oriented(s, f)
        if s["kind"] == "ARC":
            c = s["c"]
            a0 = math.atan2(q0[1] - c[1], q0[0] - c[0])
            a1 = math.atan2(q1[1] - c[1], q1[0] - c[0])
            am = math.atan2(s["mid"][1] - c[1], s["mid"][0] - c[0])
            da = (a1 - a0) % (2 * math.pi)
            if (am - a0) % (2 * math.pi) > da:
                da -= 2 * math.pi
            for t in np.linspace(0, 1, n_arc + 1):
                pts.append((c[0] + s["r"] * math.cos(a0 + t * da), c[1] + s["r"] * math.sin(a0 + t * da), k))
        else:
            pts.append((q0[0], q0[1], k))
            pts.append((q1[0], q1[1], k))
    return pts


def grooves(model, wcs, elems, inner=False, tol=0.02):
    """Einstiche/Freistiche: Bereiche, in denen die Kontur unter die „Hüllkontur“ fällt
    (außen tiefer als beide Seiten, innen weiter als beide Seiten)."""
    if not elems:
        return []
    mo, md = main_axis(model)
    Z = lambda a: float(wcs.to_wcs(mo + a * md)[2])
    P = _profile_samples(elems)
    sg = -1.0 if inner else 1.0
    r = np.array([sg * p[1] for p in P])
    n = len(r)
    lm = np.maximum.accumulate(r)
    rm = np.maximum.accumulate(r[::-1])[::-1]
    env = np.minimum(lm, rm)
    below = r < env - tol
    out = []
    i = 0
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1]:
            j += 1
        L, R = i - 1, j + 1
        if L < 0 or R >= n:
            i = j + 1
            continue
        E = min(lm[L], rm[R])
        # Kanten auf Höhe E (linear zwischen den Stützpunkten)
        def cross(k0, k1):
            r0, r1 = r[k0], r[k1]
            t = 0.0 if abs(r1 - r0) < 1e-12 else (E - r0) / (r1 - r0)
            t = min(max(t, 0.0), 1.0)
            return P[k0][0] + t * (P[k1][0] - P[k0][0])
        aL = cross(L, L + 1)
        aR = cross(R, R - 1)
        ks = sorted({P[k][2] for k in range(L, R + 1)})
        ks = [k for k in ks if any(r[q] < E - tol for q in range(n) if P[q][2] == k) or
              elems[k][0]["kind"] == "PLANE"]
        depth = E - r[i:j + 1].min()
        radii, angles = [], []
        for k in ks:
            s_ = elems[k][0]
            if s_["kind"] == "ARC":
                radii.append(s_["r"])
            elif s_["kind"] == "CONE":
                angles.append(s_["angle"])
        step = abs(lm[L] - rm[R]) > 0.1
        kind = "Freistich" if step else "Einstich"
        note = ""
        if step and radii and any(abs(a - 15) < 1.0 for a in angles):
            note = "ähnlich DIN 509 Form " + ("F" if any(abs(a - 8) < 1.0 for a in angles) else "E")
        bottom = sg * r[i:j + 1].min()
        out.append(dict(kind=kind + (" innen" if inner else ""), z0=Z(aL), z1=Z(aR), width=abs(aR - aL),
                        depth=depth, dia=2 * sg * E, bottom=2 * bottom, radii=sorted(set(round(x, 4) for x in radii)),
                        note=note, faces=[elems[k][0]["face"] for k in ks]))
        i = j + 1
    return out


# Wendeschneidplatten für Längs-/Kopierdrehen: (Name, Spitzenwinkel, Einstellwinkel κr)
STD_INSERTS = [("CNMG 80° κ95°", 80, 95), ("TNMG 60° κ93°", 60, 93), ("DNMG 55° κ93°", 55, 93),
               ("VBMT 35° κ93°", 35, 93), ("VBMT 35° κ72,5°", 35, 72.5)]


def insert_clearance(eps, kr):
    """Nebeneinstellwinkel κ' = 180° − κr − ε: so steil darf die Kontur in Vorschubrichtung abfallen."""
    return 180.0 - kr - eps


def tool_check(model, wcs, elems, inner=False, margin=2.0):
    """Für jedes Konturelement: wie steil fällt es in Vorschubrichtung −Z bzw. +Z ab, und welche Platte
    kommt hin (κ' ≥ Abfallwinkel + Sicherheit). Hohlkehlen begrenzen den Schneidenradius."""
    mo, md = main_axis(model)
    Z = lambda a: float(wcs.to_wcs(mo + a * md)[2])
    rows = []
    sg = -1.0 if inner else 1.0
    for k, (s, f) in enumerate(elems):
        pts = _profile_samples([(s, f)], n_arc=24)
        zs = np.array([Z(p[0]) for p in pts])
        rs = np.array([sg * p[1] for p in pts])
        dive = {}
        for dirn, sgn in (("-Z", -1.0), ("+Z", 1.0)):
            worst = 0.0
            for q in range(len(pts) - 1):
                dz, dr = zs[q + 1] - zs[q], rs[q + 1] - rs[q]
                # in Vorschubrichtung laufen: Strecke ggf. umdrehen
                if dz * sgn < 0 or (abs(dz) < 1e-9 and False):
                    dz, dr = -dz, -dr
                if abs(dz) < 1e-9:
                    if abs(dr) > 1e-9:
                        worst = max(worst, 90.0)   # senkrechte Planfläche: nur fallend kritisch
                        if dr > 0:
                            worst = max(worst, 0.0)
                    continue
                if dr < -1e-9:
                    worst = max(worst, math.degrees(math.atan2(-dr, abs(dz))))
            dive[dirn] = worst
        # senkrechte Planflächen: steigend in der einen, fallend in der anderen Richtung
        if s["kind"] == "PLANE":
            q0, q1 = _oriented(s, f)
            # welche Seite liegt höher? Nachbarn entscheiden – einfache Regel: Planfläche ist für das Längsdrehen
            # nur in der Richtung erreichbar, in der sie zur Schneide hin steigt
            prev_r = sg * _oriented(*elems[k - 1])[0][1] if k > 0 else None
            next_r = sg * _oriented(*elems[k + 1])[1][1] if k + 1 < len(elems) else None
            za, zb = Z(q0[0]), Z(q1[0])
            zprev = Z(_oriented(*elems[k - 1])[0][0]) if k > 0 else za
            # Kontur läuft (in Elementreihenfolge) über diese Planfläche nach oben oder unten
            up = sg * q1[1] > sg * q0[1]
            moving_dir = 1.0 if (k + 1 < len(elems) and Z(_oriented(*elems[k + 1])[1][0]) > za) or \
                (k > 0 and zprev < za) else -1.0
            # in Laufrichtung der Elemente steigt/fällt sie; Gegenrichtung umgekehrt
            dive = {"+Z": 0.0, "-Z": 0.0}
            run = "+Z" if moving_dir > 0 else "-Z"
            other = "-Z" if run == "+Z" else "+Z"
            dive[run] = 0.0 if up else 90.0
            dive[other] = 90.0 if up else 0.0
        def ok_tools(a):
            return [nm for nm, e, kr in STD_INSERTS if insert_clearance(e, kr) >= a + margin]
        row = dict(nr=k + 1, elem={"PLANE": "Plan", "CYL": "Gerade Z", "CONE": "Kegel/Fase", "ARC": "Radius"}[s["kind"]],
                   z0=Z(_oriented(s, f)[0][0]), z1=Z(_oriented(s, f)[1][0]), face=s["face"],
                   dive_m=dive["-Z"], dive_p=dive["+Z"], tools_m=ok_tools(dive["-Z"]), tools_p=ok_tools(dive["+Z"]),
                   rmax=None)
        if s["kind"] == "ARC" and model.faces[s["face"]].concavity == "1":
            row["rmax"] = s["r"]
        rows.append(row)
    return rows


def milling_features(model, wcs):
    """Fräsmerkmale am Drehteil: Nuten (Boden + zwei Seitenwände), Schlüsselflächen (zwei parallele
    Flächen, SW) und einzelne Abflachungen. Lage in Z und C, Tiefe ab Außendurchmesser."""
    mo, md = main_axis(model)
    e1, e2 = angle_frame(md, wcs)
    Z = lambda a: float(wcs.to_wcs(mo + a * md)[2])
    vf = _vertex_faces(model)
    def neighbours(i):
        out = set()
        for k in map(tuple, np.round(model.faces[i].triangles.reshape(-1, 3), 4)):
            out |= vf.get(k, set())
        out.discard(i)
        return out
    planes = {}
    for i, f in enumerate(model.faces):
        if isinstance(f.surf, PlaneSurf) and abs(f.surf.n @ md) < 1e-3:
            n = outward_normal(model, i, f.surf.n)
            V = f.triangles.reshape(-1, 3)
            c = V.mean(0)
            planes[i] = dict(n=n, c=c, h=float((c - mo) @ n), V=V)
    try:
        cyl_r = [float(sg["p0"][1]) for sg in lathe_segments(model)[0]
                 if sg["kind"] == "CYL" and model.faces[sg["face"]].concavity != "1"]
    except Exception:
        cyl_r = []

    def snapR(R):
        near = [r for r in cyl_r if abs(r - R) < max(0.05, 2e-3 * R)]
        return min(near, key=lambda r: abs(r - R)) if near else R
    used, feats = set(), []
    for i, pl in planes.items():
        if pl["h"] <= 1e-3:
            continue
        n = pl["n"]
        walls = [j for j in neighbours(i) if j in planes and abs(planes[j]["n"] @ n) < 1e-3]
        pair = None
        for a in walls:
            for b in walls:
                if a < b and planes[a]["n"] @ planes[b]["n"] < -0.999:
                    pair = (a, b)
        ends = [j for j in neighbours(i) if model.faces[j].ftype == "CYLINDER" and model.faces[j].concavity == "1"
                and isinstance(model.faces[j].surf, RevSurf) and abs(abs(model.faces[j].surf.d @ n) - 1) < 1e-3]
        c_ang = math.degrees(math.atan2(n @ e2, n @ e1)) % 360.0
        if pair:
            a, b = pair
            width = abs(float((planes[a]["c"] - planes[b]["c"]) @ planes[a]["n"]))
            fs = [i, a, b] + ends
            V = np.concatenate([model.faces[j].triangles.reshape(-1, 3) for j in fs])
            Wv = np.concatenate([planes[a]["V"], planes[b]["V"]])
            q = Wv - mo
            R = snapR(float(np.linalg.norm(q - np.outer(q @ md, md), axis=1).max()))
            ax = (V - mo) @ md
            feats.append(dict(kind="Nut", width=width, depth=R - pl["h"], dia=2 * R,
                              z0=Z(ax.min()), z1=Z(ax.max()), length=float(ax.max() - ax.min()),
                              c=c_ang, h=pl["h"], ends="rund" if ends else "eckig", faces=fs))
            used |= set(fs)
            continue
        if walls:
            continue
        q = pl["V"] - mo
        R = snapR(float(np.linalg.norm(q - np.outer(q @ md, md), axis=1).max()))
        ax = (pl["V"] - mo) @ md
        feats.append(dict(kind="Abflachung", width=None, depth=R - pl["h"], dia=2 * R, z0=Z(ax.min()), z1=Z(ax.max()),
                          length=float(ax.max() - ax.min()), c=c_ang, h=pl["h"], ends="", faces=[i]))
    # Schlüsselflächen: zwei gegenüberliegende Abflachungen gleicher Lage
    flats = [f for f in feats if f["kind"] == "Abflachung"]
    done = set()
    for x in flats:
        for y in flats:
            if id(x) >= id(y) or id(x) in done or id(y) in done:
                continue
            nx = planes[x["faces"][0]]["n"]; ny = planes[y["faces"][0]]["n"]
            if nx @ ny < -0.999 and abs(x["h"] - y["h"]) < 0.01 and min(x["z1"], y["z1"]) > max(x["z0"], y["z0"]) - 1e-3:
                x["kind"] = "Schlüsselfläche"
                x["width"] = x["h"] + y["h"]
                x["faces"] = x["faces"] + y["faces"]
                x["c2"] = y["c"]
                done |= {id(x), id(y)}
                feats.remove(y)
    for f in feats:
        if f["z0"] > f["z1"]:
            f["z0"], f["z1"] = f["z1"], f["z0"]
    feats.sort(key=lambda f: (f["z0"], f["c"]))
    return feats


def pocket_contour(model, wcs, feat):
    """Fräskontur einer Nut (oder Tasche) in der Ebene des Nutgrunds – so, wie sie für das Fräsen
    beschrieben wird: Übergangspunkte Linie/Bogen (Z, Y), Bögen mit R, Mittelpunkt und Drehsinn.
    Koordinaten: Z entlang der Längsachse (Nullpunkt), Y quer dazu in der Lage C der Nut,
    X = Ø-Wert (Boden bzw. Oberkante am Mantel). Umlauf gegen den Uhrzeigersinn mit Blick von +X
    (Werkzeug innen links der Kontur = G41). Ergebnis: dict mit points, elems, info – oder None."""
    if feat.get("kind") != "Nut":
        return None
    mo, md = main_axis(model)
    floor = feat["faces"][0]
    side = [j for j in feat["faces"][1:]]
    f0 = model.faces[floor]
    n = outward_normal(model, floor, f0.surf.n)
    eo = unit(n - (n @ md) * md)
    ey = np.cross(md, eo)
    h = float(feat["h"])
    z_of = lambda p: float(wcs.to_wcs(mo + float((p - mo) @ md) * md)[2])
    zsign = float(np.sign(wcs.dir_to_wcs(md)[2]) or 1.0)

    def to2(p):
        q = np.asarray(p, float) - mo
        return np.array([float(q @ md), float(q @ ey)])      # (a entlang Achse, y)

    # Randschleife des Nutgrunds und Zuordnung jedes Randpunkts zur angrenzenden Seitenfläche
    tris = f0.triangles.reshape(-1, 3, 3)
    key = np.round(tris.reshape(-1, 3), 4)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    E = np.sort(np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]]), axis=1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = uE[cnt == 1]
    if len(bnd) < 3:
        return None
    adj = {}
    for x, y in bnd:
        adj.setdefault(int(x), []).append(int(y))
        adj.setdefault(int(y), []).append(int(x))
    start = int(bnd[0][0])
    loop, prev, cur = [start], None, start
    for _ in range(len(adj) + 2):
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    vf = _vertex_faces(model)
    sset = set(side)
    fsets = [set(vf.get(tuple(uk[v]), ())) & sset for v in loop]
    lab = []
    for i in range(len(loop)):                 # jede Randstrecke gehört zu der Seitenfläche beider Endpunkte
        fs = fsets[i] & fsets[(i + 1) % len(loop)]
        lab.append(next(iter(fs)) if len(fs) == 1 else None)
    # Elementfolge (zyklisch, ohne Wiederholungen)
    seq = []
    for j in lab:
        if j is not None and (not seq or seq[-1] != j):
            seq.append(j)
    while len(seq) > 1 and seq[0] == seq[-1]:
        seq.pop()
    if len(seq) < 2:
        return None
    pts2 = np.array([to2(uk[v]) for v in loop])
    # Umlaufsinn: gegen den Uhrzeigersinn in (Y, Z) mit Blick von +X
    yz = np.column_stack([pts2[:, 1], zsign * pts2[:, 0]])
    area = 0.5 * np.sum(yz[:, 0] * np.roll(yz[:, 1], -1) - np.roll(yz[:, 0], -1) * yz[:, 1])
    if area < 0:
        seq = seq[::-1]
    elems = []
    for j in seq:
        f = model.faces[j]
        V2 = np.array([to2(p) for p in f.triangles.reshape(-1, 3)])
        if isinstance(f.surf, PlaneSurf):
            nn = f.surf.n
            t3 = np.cross(n, nn)
            t2 = np.array([float(t3 @ md), float(t3 @ ey)])
            t2 /= np.linalg.norm(t2)
            elems.append(dict(kind="LINE", face=j, p=V2.mean(0), t=t2, V=V2))
        elif isinstance(f.surf, RevSurf):
            c2 = to2(f.surf.o)
            r = float(np.linalg.norm(V2 - c2, axis=1).mean()) if f.surf.radius is None else float(f.surf.radius)
            elems.append(dict(kind="ARC", face=j, c=c2, r=r, V=V2))
        else:
            return None

    def meet(a, b, hint):
        if a["kind"] == "LINE" and b["kind"] == "LINE":
            A = np.column_stack([a["t"], -b["t"]])
            if abs(np.linalg.det(A)) < 1e-9:
                return hint
            s_ = np.linalg.solve(A, b["p"] - a["p"])
            return a["p"] + s_[0] * a["t"]
        if a["kind"] == "ARC" and b["kind"] == "LINE":
            a, b = b, a
        if a["kind"] == "LINE" and b["kind"] == "ARC":
            foot = a["p"] + ((b["c"] - a["p"]) @ a["t"]) * a["t"]
            dist = np.linalg.norm(b["c"] - foot)
            if abs(dist - b["r"]) < 1e-3 * max(1, b["r"]):
                return foot                                   # tangentialer Übergang
            if dist > b["r"]:
                return hint
            k = math.sqrt(b["r"] ** 2 - dist ** 2)
            cands = [foot + k * a["t"], foot - k * a["t"]]
            return min(cands, key=lambda q: np.linalg.norm(q - hint))
        dcc = b["c"] - a["c"]
        L = np.linalg.norm(dcc)
        if L < 1e-9:
            return hint
        u = dcc / L
        if abs(L - (a["r"] + b["r"])) < 1e-3 or abs(L - abs(a["r"] - b["r"])) < 1e-3:
            return a["c"] + a["r"] * u * (1 if L > abs(a["r"] - b["r"]) + 1e-3 or a["r"] >= b["r"] else -1)
        x = (L * L + a["r"] ** 2 - b["r"] ** 2) / (2 * L)
        yv = math.sqrt(max(a["r"] ** 2 - x * x, 0.0))
        pm = a["c"] + x * u
        w = np.array([-u[1], u[0]])
        return min([pm + yv * w, pm - yv * w], key=lambda q: np.linalg.norm(q - hint))

    m = len(elems)
    trans = []
    for i in range(m):
        a, b = elems[i], elems[(i + 1) % m]
        Va, Vb = a["V"], b["V"]
        dd = np.linalg.norm(Va[:, None, :] - Vb[None, :, :], axis=2)
        k = np.unravel_index(np.argmin(dd), dd.shape)
        hint = 0.5 * (Va[k[0]] + Vb[k[1]])
        trans.append(meet(a, b, hint))
    R_out = feat["dia"] / 2.0

    def pt(q):
        y = float(q[1])
        xo = 2 * math.sqrt(max(R_out ** 2 - y * y, 0.0))
        return dict(z=z_of(mo + q[0] * md), y=y, x_floor=2 * h, x_top=xo,
                    p3=mo + q[0] * md + y * ey + (xo / 2) * eo,            # auf der Oberkante (Mantel)
                    p3_floor=mo + q[0] * md + y * ey + h * eo)
    # Punkte: P1 = Anfang von Element 1 (= Übergang vom letzten Element), dann Ende jedes Elements
    points = [pt(trans[-1])] + [pt(trans[i]) for i in range(m)]
    rows = []
    for i, e in enumerate(elems):
        p0, p1 = points[i], points[i + 1]
        r = dict(nr=i + 1, kind="Linie" if e["kind"] == "LINE" else "Bogen", z0=p0["z"], y0=p0["y"], z1=p1["z"],
                 y1=p1["y"], face=e["face"], r=None, cz=None, cy=None, g=None)
        if e["kind"] == "ARC":
            c = e["c"]
            mid = e["V"][np.argmax(np.linalg.norm(e["V"] - 0.5 * (trans[i - 1] + trans[i]), axis=1))]
            v0 = np.array([trans[i - 1][1] - c[1], zsign * (trans[i - 1][0] - c[0])])
            vm = np.array([mid[1] - c[1], zsign * (mid[0] - c[0])])
            cr = v0[0] * vm[1] - v0[1] * vm[0]
            r.update(r=e["r"], cz=z_of(mo + c[0] * md), cy=float(c[1]), g="G3" if cr > 0 else "G2")
        rows.append(r)
    return dict(points=points, rows=rows, h=h, c=feat["c"], width=feat.get("width"), depth=feat["depth"],
                dia=feat["dia"], eo=eo, ey=ey)


def pocket_contour_reversed(pc):
    """Gleiche Kontur im Uhrzeigersinn (Werkzeug rechts = G42)."""
    pts = pc["points"][::-1]
    rows = []
    for i, r in enumerate(pc["rows"][::-1]):
        q = dict(r)
        q.update(nr=i + 1, z0=r["z1"], y0=r["y1"], z1=r["z0"], y1=r["y0"])
        if r["g"]:
            q["g"] = "G2" if r["g"] == "G3" else "G3"
        rows.append(q)
    out = dict(pc)
    out.update(points=pts, rows=rows)
    return out


def pocket_points_3d(model, wcs):
    """Alle Übergangspunkte aller Nuten: [(Punkt 3D, Nut-Index, Punkt-Index)] – für die roten Punkte."""
    out = []
    try:
        feats = milling_features(model, wcs)
    except Exception:
        return out
    for k, f in enumerate(feats):
        if f["kind"] != "Nut":
            continue
        try:
            pc = pocket_contour(model, wcs, f)
        except Exception:
            pc = None
        if pc:
            for i, p in enumerate(pc["points"][:-1]):
                out.append((p["p3"], k, i))
    return out


def contour_gcode(rows, dialect="iso", x_radius=False, swap_arcs=False, name="Kontur"):
    """Kontur als neutrales G-Code-Konturprogramm (nur Geometrie, ohne Technologie).
    dialect: 'iso' = Fanuc / Mazak EIA / DIN PLUS (Bogen mit R), 'siemens' = Sinumerik 840D (Bogen mit CR=)."""
    def num(v):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    X = (lambda x: x / 2.0) if x_radius else (lambda x: x)
    cm = (lambda t: f"; {t}") if dialect == "siemens" else (lambda t: f"({t})")
    out = [cm(f"{name} – nur Geometrie, ohne Werkzeug/Vorschub/Schneidenradiuskorrektur"),
           cm("X als " + ("Radius" if x_radius else "Durchmesser") + ", Z im gewählten Nullpunkt"),
           cm("Vor Gebrauch in der Simulation prüfen!")]
    n = 10
    for r in rows:
        x, z = num(X(r["x"])), num(r["z"])
        if r["elem"] == "Start":
            out.append(f"N{n} G0 X{x} Z{z}")
        elif r["elem"] == "Radius":
            g = "G2" if "G2" in r.get("dir", "") else "G3"
            if swap_arcs:
                g = "G3" if g == "G2" else "G2"
            rad = num(r["r"])
            out.append(f"N{n} {g} X{x} Z{z} " + (f"CR={rad}" if dialect == "siemens" else f"R{rad}"))
        else:
            out.append(f"N{n} G1 X{x} Z{z}")
        n += 10
    out.append(cm("Ende Kontur"))
    txt = "\n".join(out) + "\n"
    for k, v in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss", "–": "-",
                 "Ø": "D", "°": " Grad"}.items():
        txt = txt.replace(k, v)          # Steuerungen vertragen meist nur ASCII
    return txt


# ---------------------------------------------------------------------------
# Ringtasche in der Stirnseite: Profil aus dem Modell, Wendel-Fräsbahn, DIN-ISO (EIA) U/W/H
# ---------------------------------------------------------------------------

def _seg_poly(s, n_arc=24):
    """Profil-Element als Punktfolge (a, r)."""
    if s["kind"] == "ARC":
        c = s["c"]
        a0 = math.atan2(s["p0"][1] - c[1], s["p0"][0] - c[0])
        a1 = math.atan2(s["p1"][1] - c[1], s["p1"][0] - c[0])
        am = math.atan2(s["mid"][1] - c[1], s["mid"][0] - c[0])
        da = (a1 - a0) % (2 * math.pi)
        if (am - a0) % (2 * math.pi) > da:
            da -= 2 * math.pi
        return [np.array([c[0] + s["r"] * math.cos(a0 + t * da), c[1] + s["r"] * math.sin(a0 + t * da)])
                for t in np.linspace(0, 1, n_arc + 1)]
    return [np.asarray(s["p0"], float), np.asarray(s["p1"], float)]


def ring_pockets(model):
    """Alle umlaufenden Taschen in Stirnseiten: ebener Grund (senkrecht zur Achse), außen eine Wand mit
    Material außen (konkav), innen eine Wand mit Material innen (Nabe, konvex) oder nichts.
    Ergebnis: Liste von Profilen (dict) – siehe ring_pocket_profile."""
    segs, _ = lathe_segments(model)
    out = []
    for i, s in enumerate(segs):
        if s["kind"] != "PLANE":
            continue
        prof = _ring_profile_from_floor(segs, i)
        if prof is not None:
            out.append(prof)
    return out


def _ring_profile_from_floor(segs, i_floor, tol=2e-3):
    fl = segs[i_floor]
    a_f = float(fl["p0"][0])
    r_lo_f, r_hi_f = sorted([float(fl["p0"][1]), float(fl["p1"][1])])

    def at(pt, skip):
        return [j for j, s in enumerate(segs) if j not in skip and
                (np.linalg.norm(s["p0"] - pt) < tol or np.linalg.norm(s["p1"] - pt) < tol)]

    def chain(pt, want_conc):
        """Wand vom Grund aus verfolgen, bis eine Planfläche (Stirnfläche) erreicht ist."""
        used, polys, faces = {i_floor}, [], []
        top = None
        while True:
            nxt = at(pt, used)
            if not nxt:
                break
            j = nxt[0]
            s = segs[j]
            if s["kind"] == "PLANE":
                top = float(s["p0"][0])
                break
            if s["conc"] != want_conc:
                return None
            used.add(j)
            faces.append(s["face"])
            poly = _seg_poly(s)
            if np.linalg.norm(poly[0] - pt) > tol:
                poly = poly[::-1]
            polys += poly
            pt = poly[-1]
        if not polys:
            return None
        return dict(pts=np.array(polys), faces=faces, top=top if top is not None else float(polys[-1][0]),
                    end=np.array(pt))

    p_hi = np.array([a_f, r_hi_f])
    p_lo = np.array([a_f, r_lo_f])
    outer = chain(p_hi, "1")
    if outer is None:
        return None
    inner = chain(p_lo, "0") if r_lo_f > 1e-6 else None
    if inner is None and r_lo_f > 1e-6:
        return None                                    # Grund endet innen an keiner Nabe – keine Ringtasche
    a_top = outer["top"]
    s_dir = 1.0 if a_f > a_top else -1.0              # Richtung „ins Material“ entlang der Achse
    if abs(a_f - a_top) < 1e-6:
        return None
    # Außenwand muss vom Grund zur Öffnung gehen (keine Hinterschneidung in Gegenrichtung)
    return dict(floor_face=fl["face"], a_floor=a_f, a_top=a_top, s=s_dir, outer=outer["pts"], inner=None if inner is None else inner["pts"],
                inner_top=None if inner is None else inner["top"], faces=[fl["face"]] + outer["faces"] + ([] if inner is None else inner["faces"]),
                r_floor=(r_lo_f, r_hi_f))


def _wall_r(pts, a, mode):
    """Radius der Wand bei Lage a (lineare Interpolation; bei mehrdeutigen Stellen der engste Wert)."""
    if pts is None:
        return None
    vals = []
    for p, q in zip(pts[:-1], pts[1:]):
        lo, hi = min(p[0], q[0]), max(p[0], q[0])
        if lo - 1e-9 <= a <= hi + 1e-9:
            if abs(q[0] - p[0]) < 1e-12:
                vals += [p[1], q[1]]
            else:
                t = (a - p[0]) / (q[0] - p[0])
                vals.append(p[1] + t * (q[1] - p[1]))
    if not vals:
        return None
    return min(vals) if mode == "out" else max(vals)


def ring_pocket_limits(prof, a, Rt, aw):
    """Erlaubter Bereich der Fräsermitte (Radius) bei Spitzenlage a: Fräser (zylindrisch, Radius Rt)
    reicht von a bis über die Öffnung – über diese ganze Höhe darf er die Wände nicht verletzen."""
    s = prof["s"]
    span = np.linspace(prof["a_top"], a, 25) if (a - prof["a_top"]) * s > 0 else np.array([a])
    ro = [_wall_r(prof["outer"], x, "out") for x in span]
    ro = [v for v in ro if v is not None]
    ri = [_wall_r(prof["inner"], x, "in") for x in span] if prof["inner"] is not None else []
    ri = [v for v in ri if v is not None]
    rhi = (min(ro) if ro else 1e9) - Rt - aw
    rlo = (max(ri) if ri else 0.0) + Rt + aw
    return rlo, rhi


def ring_pocket_path(prof, Rt, ap, ae, step_deg=2.0, aw=0.0, af=0.0, safety=2.0, strategy="wendel",
                     r_start=None, c_dir=1.0):
    """Fräsbahn als Punktfolge (a, r, c°): wendelförmiges Eintauchen, C dreht, ohne senkrechtes Einstechen.
    strategy „wendel“: je Bahn eine durchgehende Wendel bis auf den Grund + eine Runde am Grund;
                      Bahnen verlaufen anteilig zwischen Innen- und Außengrenze (Schrägen werden mitgenommen).
    strategy „schicht“: je Umdrehung eine Zustellung (Wendel), dann auf dieser Tiefe alle Bahnen als Kreise.
    Ergebnis: dict(points=[(a, r, c, art)], info=…) – art: „luft“, „wendel“, „kreis“, „quer“."""
    s = prof["s"]
    a_top = prof["a_top"]
    a_end = prof["a_floor"] - s * af
    a_start = a_top - s * safety
    depth = (a_end - a_top) * s
    if depth <= 0:
        raise ValueError("Tasche ist nicht tiefer als das Aufmaß.")
    lim_f = ring_pocket_limits(prof, a_end, Rt, aw)
    if lim_f[1] < lim_f[0] - 1e-9:
        raise ValueError(f"Fräser Ø{fmt(2 * Rt)} passt am Grund nicht in die Tasche "
                         f"(Platz für Fräsermitte von r {fmt(lim_f[0])} bis r {fmt(lim_f[1])}).")
    # Grenzen über die Tiefe (für Prüfung und anteilige Bahnen)
    levels_all = np.linspace(a_top, a_end, 60)
    lims = [ring_pocket_limits(prof, x, Rt, aw) for x in levels_all]
    widest = max(hi - lo for lo, hi in lims)
    n_tr = max(1, int(math.ceil(widest / max(ae, 1e-3) - 1e-9))) + 1 if widest > 1e-6 else 1
    ts = np.linspace(0.0, 1.0, n_tr) if n_tr > 1 else np.array([0.5])
    if r_start is not None and n_tr > 1:
        t0 = (r_start - lim_f[0]) / max(lim_f[1] - lim_f[0], 1e-9)
        k0 = int(np.argmin(np.abs(ts - t0)))
    else:
        k0 = 0
    order = [k0] + list(range(k0 + 1, n_tr)) + list(range(k0 - 1, -1, -1))

    def r_of(t, a):
        lo, hi = ring_pocket_limits(prof, a, Rt, aw)
        if hi < lo:
            return 0.5 * (lo + hi)
        return lo + t * (hi - lo)

    pts = []
    c = 0.0

    def emit(a, r, kind):
        pts.append((float(a), float(r), float(c), kind))

    def helix(t, a0, a1):
        nonlocal c
        nrev = max(1, int(math.ceil(abs(a1 - a0) / ap - 1e-9)))
        nstep = max(1, int(math.ceil(360.0 * nrev / step_deg)))
        for k in range(1, nstep + 1):
            f = k / nstep
            a = a0 + f * (a1 - a0)
            c += c_dir * 360.0 * nrev / nstep
            emit(a, r_of(t, a) if a * s >= a_top * s else r_of(t, a_top), "wendel")

    def circle(t, a):
        nonlocal c
        nstep = max(1, int(math.ceil(360.0 / step_deg)))
        r = r_of(t, a)
        for k in range(1, nstep + 1):
            c += c_dir * 360.0 / nstep
            emit(a, r, "kreis")

    first = order[0]
    emit(a_start, r_of(ts[first], a_top), "start")
    if strategy == "wendel":
        for n, k in enumerate(order):
            t = ts[k]
            if n > 0:
                emit(a_start, pts[-1][1], "luft")                     # zurück über die Öffnung
                emit(a_start, r_of(t, a_top), "luft")               # quer zur nächsten Bahn
            helix(t, a_start, a_end)
            circle(t, a_end)
        emit(a_start, pts[-1][1], "luft")
        emit(a_start, pts[0][1], "luft")                              # zurück auf den Startradius
    else:
        nlev = max(1, int(math.ceil((depth + safety) / ap - 1e-9)))
        a_prev = a_start
        for i in range(1, nlev + 1):
            a_k = a_start + s * min(i * ap, depth + safety)
            if (a_k - a_top) * s < 0:
                continue
            emit(a_prev, r_of(ts[first], max(a_prev, a_top) if s > 0 else min(a_prev, a_top)), "quer")
            helix(ts[first], a_prev, a_k)
            circle(ts[first], a_k)
            for k in order[1:]:
                emit(a_k, r_of(ts[k], a_k), "quer")
                circle(ts[k], a_k)
            a_prev = a_k
        emit(a_start, pts[-1][1], "luft")
        emit(a_start, pts[0][1], "luft")
    # Prüfung: keine Bahn verletzt die Wände
    worst = 0.0
    for a, r, _, kind in pts:
        if (a - a_top) * s <= 0:
            continue
        lo, hi = ring_pocket_limits(prof, a, Rt, 0.0)
        worst = max(worst, lo - r, r - hi)
    info = dict(tracks=n_tr, blocks=len(pts) - 1, r_floor=lim_f, worst=worst, a_start=a_start, a_end=a_end)
    return dict(points=pts, info=info)


def ring_pocket_eia(model, wcs, prof, path, F=100.0, S=1500, decimals=4, w_neg=True, u_diameter=True,
                    f_comp=False, name="RINGTASCHE", Rt=None, params_text=""):
    """DIN-ISO-Unterprogramm (Mazak EIA) wie bei Wendelnuten: G01 mit inkrementellen U (X), W (Z), H (C).
    Rundungsfehler werden mitgeführt, die Summen gehen genau auf."""
    mo, md = main_axis(model)
    zs = float(np.sign(wcs.dir_to_wcs(md)[2]) or 1.0)
    pts = path["points"]

    def Zw(a):
        return float(wcs.to_wcs(mo + a * md)[2])
    a0, r0, c0, _ = pts[0]
    q = 10.0 ** decimals
    acc = np.zeros(3)                  # bereits ausgegeben (gerundet)
    exact = np.zeros(3)
    lines = [f"({name} - INKREMENTELL U/W/H, NUR BAHN OHNE TECHNOLOGIEPRUEFUNG)",
             f"(STARTPUNKT: X{2 * r0:.3f} Z{Zw(a0) * (-1 if w_neg else 1) * zs:.3f} C0 - DORT POSITIONIEREN)",
             "(ZUM TESTEN: STARTPUNKT IN DER LUFT VOR DER STIRNSEITE ANSETZEN)"]
    if params_text:
        lines.append(f"({params_text})")
    lines += [f"S{int(S)}", f"F{fmt(F, 3).replace(',', '.')}", "G01"]
    lastF = F
    L_real = 0.0
    for (a, r, c, kind), (pa, pr, pc, _) in zip(pts[1:], pts[:-1]):
        du = (r - pr) * (2 if u_diameter else 1)
        dw = (Zw(a) - Zw(pa)) * (-1.0 if w_neg else 1.0) * zs
        dh = c - pc
        exact += (du, dw, dh)
        out = np.round(exact * q) / q - acc
        acc += out
        seg = math.sqrt((r - pr) ** 2 + (a - pa) ** 2 + ((r + pr) / 2 * math.radians(dh)) ** 2)
        L_real += seg
        if np.all(np.abs(out) < 0.5 / q):
            continue                                   # Satz ohne Bewegung weglassen
        parts = []
        if abs(out[0]) > 0.5 / q:
            parts.append(f"U{out[0]:.{decimals}f}")
        parts.append(f"W{out[1]:.{decimals}f}")
        parts.append(f"H{out[2]:.{decimals}f}")
        if f_comp and seg > 1e-9:
            L_cmd = math.sqrt(out[0] ** 2 + out[1] ** 2 + out[2] ** 2)
            fb = F * L_cmd / seg
            if abs(fb - lastF) > 0.01 * F:
                parts.append(f"F{fb:.1f}")
                lastF = fb
        lines.append(" ".join(p.replace("-0.", "-0.") for p in parts))
    lines.append("M99")
    tot = acc
    info = dict(blocks=len(lines) - 6, U=tot[0], W=tot[1], H=tot[2], length=L_real,
                minutes=L_real / F if F > 0 else 0.0)
    return "\r\n".join(lines) + "\r\n", info


def ring_path_3d(model, wcs, path):
    """Bahn als 3D-Punkte (für die Vorschau)."""
    mo, md = main_axis(model)
    e1, e2 = angle_frame(md, wcs)
    out = []
    for a, r, c, _ in path["points"]:
        cr = math.radians(c)
        out.append(mo + a * md + r * (math.cos(cr) * e1 + math.sin(cr) * e2))
    return np.array(out)


# ---------------------------------------------------------------------------
# Nut am Mantel (z. B. Wendelnut): Mittellinie Z/C aus den Nutflächen, Ausgabe W/H
# ---------------------------------------------------------------------------

def _ray_hits(p, d, A, B, tmax):
    """Kleinstes t > 0, bei dem der Strahl p + t·d die Strecken A→B trifft (2D), sonst None."""
    e = B - A
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(den) > 1e-12
    w = A - p
    t = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1), np.inf)
    u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1), -1)
    hit = ok & (t > 1e-6) & (t < tmax) & (u >= -1e-9) & (u <= 1 + 1e-9)
    return float(t[hit].min()) if hit.any() else None


def groove_centerline(model, faces, step=0.25, r_step=None):
    """Mittellinie einer Nut am Mantel (Fräser radial). faces: Flächen der Nut (Grund, ggf. Wände).
    Der Nutgrund wird auf den Radius abgewickelt (a entlang der Achse, s = r·C); die Mitte zwischen den
    beiden Rändern wird quer zur Nut bestimmt (exakt für Nuten konstanter Breite).
    Ergebnis: dict(a=[…], c=[…] (Grad, stetig), r_floor, r_top, width, length, …) oder ValueError."""
    mo, md = main_axis(model)
    e1 = any_perp(md)
    e2 = np.cross(md, e1)
    T = np.concatenate([model.faces[i].triangles.reshape(-1, 3, 3) for i in faces])
    C = T.mean(1) - mo
    ca = C @ md
    cr = C - np.outer(ca, md)
    rho_c = np.linalg.norm(cr, axis=1)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None] + 1e-15
    u = cr / (rho_c[:, None] + 1e-15)
    radial = np.abs(np.einsum("ij,ij->i", n, u)) > 0.8
    Vall = T.reshape(-1, 3) - mo
    rho_all = np.linalg.norm(Vall - np.outer(Vall @ md, md), axis=1)
    r_min, r_top = float(rho_all.min()), float(rho_all.max())
    floor = radial & (rho_c < r_min + max(0.35 * (r_top - r_min), 0.5))
    if floor.sum() < 3:
        raise ValueError("Kein Nutgrund gefunden – bitte den Grund der Nut (oder die ganze Nut) anklicken.")
    TF = T[floor]
    key = np.round(TF.reshape(-1, 3), 4)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    q = uk - mo
    qa = q @ md
    qth = np.arctan2(q @ e2, q @ e1)
    ref = math.atan2(float(np.sin(qth).mean()), float(np.cos(qth).mean()))
    qth = (qth - ref + math.pi) % (2 * math.pi) - math.pi
    qr = np.linalg.norm(q - np.outer(qa, md), axis=1)
    r_fl = float(np.percentile(qr, 10))
    Q = np.column_stack([qa, r_fl * qth])                                   # abgewickelt (a, s)
    E = np.sort(np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]]), axis=1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = uE[cnt == 1]
    # nur der äußere Rand zählt (kleine Löcher in der Vernetzung des Grundes ignorieren)
    par = {}

    def root(v):
        par.setdefault(v, v)
        while par[v] != v:
            par[v] = par[par[v]]
            v = par[v]
        return v
    for x, y in bnd:
        par[root(int(x))] = root(int(y))
    lens = {}
    for x, y in bnd:
        r_ = root(int(x))
        lens[r_] = lens.get(r_, 0.0) + float(np.linalg.norm(Q[x] - Q[y]))
    main_r = max(lens, key=lens.get)
    bnd = np.array([e for e in bnd if root(int(e[0])) == main_r])
    A, B = Q[bnd[:, 0]], Q[bnd[:, 1]]
    span = float(np.linalg.norm(Q.max(0) - Q.min(0)))
    # grobe Mitte: von jedem Randpunkt quer (beide Richtungen) zum gegenüberliegenden Rand
    mids, widths = [], []
    for k, (x, y) in enumerate(bnd):
        p = 0.5 * (Q[x] + Q[y])
        t = Q[y] - Q[x]
        L = np.linalg.norm(t)
        if L < 1e-9:
            continue
        nn = np.array([-t[1], t[0]]) / L
        for sgn in (1, -1):
            hit = _ray_hits(p + sgn * nn * 1e-6, sgn * nn, A, B, span)
            if hit is not None:
                mids.append(p + sgn * nn * hit / 2)
                widths.append(hit)
    if len(mids) < 4:
        raise ValueError("Nutränder konnten nicht bestimmt werden.")
    mids, widths = np.array(mids), np.array(widths)
    w0 = float(np.median(widths))
    good = np.abs(widths - w0) < 0.08 * w0
    M = mids[good]
    # Reihenfolge: Hauptrichtung der Punktwolke, dann glätten über Abschnitte
    cen = M.mean(0)
    _, _, vt = np.linalg.svd(M - cen)
    dirv = vt[0]
    M = M[np.argsort((M - cen) @ dirv)]
    # Kette über nächste Nachbarn (auch bei stark gekrümmten Nuten)
    order, used = [0], np.zeros(len(M), bool)
    used[0] = True
    for _ in range(len(M) - 1):
        d = np.linalg.norm(M - M[order[-1]], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        order.append(j)
        used[j] = True
    M = M[order]
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))]
    total = seg[-1]
    nb = max(8, int(total / max(0.5, 0.1 * w0)))
    grid = np.linspace(0, total, nb + 1)
    cc = []
    for g0, g1 in zip(grid[:-1], grid[1:]):
        sel = (seg >= g0) & (seg <= g1)
        if sel.any():
            cc.append(M[sel].mean(0))
    cc = np.array(cc)

    # genau: quer zur Mittellinie auf beide Ränder messen und mitteln (2 Durchgänge)
    def refine(P):
        out, ws = [], []
        for i in range(len(P)):
            t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
            t /= np.linalg.norm(t) + 1e-15
            nn = np.array([-t[1], t[0]])
            h1 = _ray_hits(P[i], nn, A, B, 2 * w0)
            h2 = _ray_hits(P[i], -nn, A, B, 2 * w0)
            if h1 is None or h2 is None or abs(h1 + h2 - w0) > 0.08 * w0:
                out.append(P[i]); ws.append(np.nan)
                continue
            out.append(P[i] + nn * (h1 - h2) / 2)
            ws.append(h1 + h2)
        return np.array(out), np.array(ws)
    cc, _ = refine(cc)
    cc, ws = refine(cc)
    ok = ~np.isnan(ws)
    if ok.sum() < 2:
        raise ValueError("Nutmitte konnte nicht sicher bestimmt werden.")
    cc = cc[ok]
    # Enden: bis zur gedachten Fräsermitte in den Endrundungen bzw. bis zum Rand verlängern
    for end in (0, -1):
        P0 = cc[end]
        t = cc[end] - cc[1 if end == 0 else -2]
        t /= np.linalg.norm(t) + 1e-15
        h = _ray_hits(P0, t, A, B, 3 * w0)
        if h is not None and h > w0 / 2 + 1e-6:
            P1 = P0 + t * (h - w0 / 2)
            cc = np.vstack([P1, cc]) if end == 0 else np.vstack([cc, P1])
    # gleichmäßig neu abtasten: Schrittlänge auf dem Radius r_step (Standard: Mantel, wie bei CAM üblich)
    rs = r_top if r_step is None else r_step
    D = np.column_stack([cc[:, 0], cc[:, 1] / r_fl * rs])
    sl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))]
    nst = max(1, int(round(sl[-1] / step)))
    g = np.linspace(0, sl[-1], nst + 1)
    a_s = np.interp(g, sl, D[:, 0])
    th = np.interp(g, sl, D[:, 1]) / rs + ref                               # rad, stetig
    return dict(a=a_s, c=np.degrees(th), r_floor=r_fl, r_top=r_top, r_step=rs, width=w0, length=float(sl[-1]),
                step=float(sl[-1] / nst), raw=cc, ref=ref)


def groove_eia(model, wcs, cl, F=100.0, S=1500, decimals=6, w_neg=True, reverse=False, h_sign=1.0,
               f_comp=False, name="NUT", absolute=False):
    """Nut-Mittellinie als DIN-ISO-Unterprogramm wie bei Mazak-Wendelnuten: G01 mit W (Z) und H (C) inkrementell
    (Summen gehen genau auf), wahlweise absolut Z/C."""
    mo, md = main_axis(model)
    e1, e2 = angle_frame(md, wcs)
    a, c = np.array(cl["a"], float), np.array(cl["c"], float)
    # C im Nullpunkt-KS: Winkel des Achs-Rahmens berücksichtigen
    u0 = any_perp(md)
    v0 = np.cross(md, u0)
    th = np.radians(c)
    vec = np.outer(np.cos(th), u0) + np.outer(np.sin(th), v0)
    cw = np.degrees(np.unwrap(np.arctan2(vec @ e2, vec @ e1)))
    Z = np.array([float(wcs.to_wcs(mo + x * md)[2]) for x in a])
    if (Z[-1] < Z[0]) != reverse:
        Z, cw = Z[::-1], cw[::-1]
    zf = -1.0 if w_neg else 1.0
    q = 10.0 ** decimals
    lines = [f"({name} - NUTMITTE, G01 {'ABSOLUT Z/C' if absolute else 'INKREMENTELL W/H'}, NUR BAHN)",
             f"(STARTPUNKT: X{2 * cl['r_floor']:.3f} (NUTGRUND) Z{Z[0] * zf:.3f} C{round(cw[0] * h_sign, 3) % 360:.3f})",
             f"(NUTBREITE {cl['width']:.3f} - LAENGE Z {abs(Z[-1] - Z[0]):.3f} - DREHUNG C {(cw[-1] - cw[0]) * h_sign:.3f})",
             f"S{int(S)}", f"F{F:g}", "G01"]
    acc = np.zeros(2)
    lastF = F
    L_real = 0.0
    for i in range(1, len(Z)):
        dz = (Z[i] - Z[i - 1]) * zf
        dh = (cw[i] - cw[i - 1]) * h_sign
        seg = math.hypot(Z[i] - Z[i - 1], cl["r_floor"] * math.radians(cw[i] - cw[i - 1]))
        L_real += seg
        if absolute:
            parts = [f"Z{Z[i] * zf:.{decimals}f}", f"C{(cw[i] * h_sign):.{decimals}f}"]
            cmd = math.hypot(dz, dh)
        else:
            ex = np.array([(Z[i] - Z[0]) * zf, (cw[i] - cw[0]) * h_sign])
            out = np.round(ex * q) / q - acc
            acc += out
            parts = [f"W{out[0]:.{decimals}f}", f"H{out[1]:.{decimals}f}"]
            cmd = math.hypot(out[0], out[1])
        if f_comp and seg > 1e-9:
            fb = F * cmd / seg
            if abs(fb - lastF) > 0.01 * F:
                parts.append(f"F{fb:.1f}")
                lastF = fb
        lines.append(" ".join(parts))
    lines.append("M99")
    info = dict(blocks=len(Z) - 1, W=float((Z[-1] - Z[0]) * zf), H=float((cw[-1] - cw[0]) * h_sign), length=L_real,
                minutes=L_real / F if F > 0 else 0.0, z0=float(Z[0]), c0=float(cw[0] * h_sign))
    return "\r\n".join(lines) + "\r\n", info


def groove_path_3d(model, cl, radius=None):
    mo, md = main_axis(model)
    u0 = any_perp(md)
    v0 = np.cross(md, u0)
    r = cl["r_floor"] if radius is None else radius
    th = np.radians(np.asarray(cl["c"], float))
    return mo + np.outer(cl["a"], md) + r * (np.outer(np.cos(th), u0) + np.outer(np.sin(th), v0))


# ---------------------------------------------------------------------------
# Fräs-Simulation (Vorführung): Restmaterial als Höhenfeld, Fräser fährt die Bahn ab
# ---------------------------------------------------------------------------

def _dense_poses(P, step):
    """Zwischenpunkte, damit der Fräser zwischen zwei Sätzen nichts überspringt. P: (n, k) Parameter,
    Länge je Abschnitt über die ersten drei Spalten (3D-Punkte)."""
    out = [P[0]]
    idx = [0]
    for i in range(1, len(P)):
        L = float(np.linalg.norm(P[i, :3] - P[i - 1, :3]))
        n = max(1, int(math.ceil(L / step)))
        for k in range(1, n + 1):
            out.append(P[i - 1] + (P[i] - P[i - 1]) * (k / n))
            idx.append(i)
    return np.array(out), np.array(idx)


class _StockBase:
    """Gemeinsame Abspiel-Logik: poses = (n, 3+…) mit 3D-Spitze vorne; cut_to(k) fräst bis Pose k."""

    def reset(self):
        self.k = 0
        self._reset_state()

    def cut_to(self, k):
        k = int(min(max(k, 0), len(self.poses) - 1))
        changed = False
        for j in range(self.k + 1, k + 1):
            changed |= self._cut(j)
        self.k = max(self.k, k)
        return changed

    def tool(self, k=None):
        """(Spitze 3D, Richtung der Werkzeugachse vom Teil weg)"""
        k = self.k if k is None else int(k)
        return self.tips[k], self.axes[k]


class RingStock(_StockBase):
    """Ringtasche in der Stirnseite: Restmaterial als Höhenfeld über (r, θ), Fräser axial."""

    def __init__(self, model, wcs, prof, path, Rt, dth=1.5):
        mo, md = main_axis(model)
        e1, e2 = angle_frame(md, wcs)
        self.mo, self.md, self.e1, self.e2 = mo, md, e1, e2
        s, a_top = prof["s"], prof["a_top"]
        D = (prof["a_floor"] - a_top) * s
        self.s, self.a_top, self.D, self.Rt = s, a_top, D, Rt
        lev = np.linspace(0.0, D, 241)
        rows = []
        for d in lev:
            a = a_top + s * min(max(d, 1e-4), D - 1e-4)
            ro = _wall_r(prof["outer"], a, "out")
            ri = _wall_r(prof["inner"], a, "in") if prof["inner"] is not None else 0.0
            if ro is not None:
                rows.append((d, ri if ri is not None else 0.0, ro))
        rows = np.array(rows)
        rmin, rmax = float(rows[:, 1].min()), float(rows[:, 2].max())
        dr = min(0.25, max((rmax - rmin) / 40.0, 0.05))
        R = np.linspace(rmin, rmax, max(3, int(math.ceil((rmax - rmin) / dr)) + 1))
        B = np.full(len(R), -1.0)
        for d, ri, ro in rows:
            inside = (R >= ri - 1e-6) & (R <= ro + 1e-6)
            B[inside] = np.maximum(B[inside], d)
        B[B >= lev[-2]] = D
        T = np.radians(np.arange(0.0, 360.0 + 1e-9, dth))
        self.R, self.T, self.B = R, T, B
        self.X = np.outer(R, np.cos(T))
        self.Y = np.outer(R, np.sin(T))
        self.BB = np.repeat(B[:, None], len(T), axis=1)
        # Bahn → Posen (3D-Spitze, a, r, c)
        pts = np.array([(a, r, c) for a, r, c, _ in path["points"]], float)
        P3 = np.array([mo + a * md + r * (math.cos(math.radians(c)) * e1 + math.sin(math.radians(c)) * e2)
                       for a, r, c in pts])
        dense, self.block = _dense_poses(np.hstack([P3, pts]), max(0.3 * Rt, 0.15))
        self.poses = dense
        self.tips = dense[:, :3]
        self.axes = np.repeat((-s * md)[None, :], len(dense), axis=0)
        self.reset()

    def _reset_state(self):
        self.t = np.zeros_like(self.BB)

    def _cut(self, j):
        a, r, c = self.poses[j, 3:6]
        d = (a - self.a_top) * self.s
        if d <= 0:
            return False
        rows = np.nonzero(np.abs(self.R - r) <= self.Rt + 1e-9)[0]
        if not len(rows):
            return False
        cr = math.radians(c)
        x, y = r * math.cos(cr), r * math.sin(cr)
        sl = slice(rows[0], rows[-1] + 1)
        m = (self.X[sl] - x) ** 2 + (self.Y[sl] - y) ** 2 <= self.Rt ** 2 + 1e-9
        new = np.minimum(d, self.BB[sl])
        upd = m & (new > self.t[sl] + 1e-9)
        if not upd.any():
            return False
        self.t[sl][upd] = new[upd]
        return True

    def mesh(self):
        """(Ecken (n,3), Dreiecke (m,3)) des Restmaterials."""
        BB, t = self.BB, self.t
        outside = BB <= 0
        done = (t >= BB - 1e-3) | outside
        h = np.where(done, BB - 0.01, t)
        h = np.where(outside, 0.0, h)
        a = self.a_top + self.s * h
        V = (self.mo[None, None, :] + a[..., None] * self.md[None, None, :]
             + self.X[..., None] * self.e1[None, None, :] + self.Y[..., None] * self.e2[None, None, :])
        nr, nt = t.shape
        idx = np.arange(nr * nt).reshape(nr, nt)
        q00, q10, q01, q11 = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
        dn, ou = done.ravel(), outside.ravel()
        keep = ~(dn[q00] & dn[q10] & dn[q01] & dn[q11]) & ~(ou[q00] | ou[q10] | ou[q01] | ou[q11])
        q00, q10, q01, q11 = q00[keep], q10[keep], q01[keep], q11[keep]
        F = np.vstack([np.stack([q00, q10, q11], 1), np.stack([q00, q11, q01], 1)])
        return V.reshape(-1, 3), F

    def progress_text(self):
        rest = np.clip(self.BB - self.t, 0, None)[self.BB > 0]
        tot = np.clip(self.BB, 0, None)[self.BB > 0]
        return 100.0 * (1.0 - rest.sum() / max(tot.sum(), 1e-9))


class GrooveStock(_StockBase):
    """Nut am Mantel: Restmaterial als Höhenfeld über (Z, C) im Nutbereich, Fräser radial auf Nutgrund."""

    def __init__(self, model, cl, da=0.2):
        mo, md = main_axis(model)
        u0 = any_perp(md)
        v0 = np.cross(md, u0)
        self.mo, self.md, self.u0, self.v0 = mo, md, u0, v0
        rf, rt, w = cl["r_floor"], cl["r_top"], cl["width"]
        self.rf, self.rtop, self.Rt = rf, rt, 0.5 * w
        a, c = np.asarray(cl["a"], float), np.radians(np.asarray(cl["c"], float))
        marg = self.Rt + 2 * da
        A = np.arange(a.min() - marg, a.max() + marg + da, da)
        dth = da / rf
        T = np.arange(c.min() - marg / rf, c.max() + marg / rf + dth, dth)
        self.A, self.T = A, T
        AA, TT = np.meshgrid(A, T, indexing="ij")
        self.AA, self.SS = AA, TT * rf
        cs = np.column_stack([a, c * rf])
        flat = np.column_stack([AA.ravel(), self.SS.ravel()])
        dmin = np.full(len(flat), np.inf)
        for k in range(0, len(flat), 20000):
            q = flat[k:k + 20000]
            dmin[k:k + 20000] = np.sqrt(((q[:, None, :] - cs[None, :, :]) ** 2).sum(-1)).min(1)
        self.mask = (dmin <= self.Rt + 1e-6).reshape(AA.shape)
        self.cos, self.sin = np.cos(TT), np.sin(TT)
        P3 = np.array([mo + x * md + rf * (math.cos(t) * u0 + math.sin(t) * v0) for x, t in zip(a, c)])
        dense, self.block = _dense_poses(np.hstack([P3, a[:, None], c[:, None]]), max(0.3 * self.Rt, 0.1))
        self.poses = dense
        self.tips = dense[:, :3]
        th = dense[:, 4]
        self.axes = np.cos(th)[:, None] * u0[None, :] + np.sin(th)[:, None] * v0[None, :]
        self.reset()

    def _reset_state(self):
        self.cut = np.zeros_like(self.mask)

    def _cut(self, j):
        a, th = self.poses[j, 3], self.poses[j, 4]
        cols = np.nonzero(np.abs(self.A - a) <= self.Rt + 1e-9)[0]
        if not len(cols):
            return False
        sl = slice(cols[0], cols[-1] + 1)
        m = ((self.AA[sl] - a) ** 2 + (self.SS[sl] - th * self.rf) ** 2 <= (self.Rt + 1e-6) ** 2) & self.mask[sl]
        upd = m & ~self.cut[sl]
        if not upd.any():
            return False
        self.cut[sl][upd] = True
        return True

    def mesh(self):
        done = self.cut | ~self.mask
        r = np.where(self.cut, self.rf + 0.01, self.rtop - 0.003)
        V = (self.mo[None, None, :] + self.AA[..., None] * self.md[None, None, :]
             + (r * self.cos)[..., None] * self.u0[None, None, :] + (r * self.sin)[..., None] * self.v0[None, None, :])
        na, nt = r.shape
        idx = np.arange(na * nt).reshape(na, nt)
        q00, q10, q01, q11 = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
        dn, mk = done.ravel(), self.mask.ravel()
        keep = ~(dn[q00] & dn[q10] & dn[q01] & dn[q11]) & (mk[q00] & mk[q10] & mk[q01] & mk[q11])
        q00, q10, q01, q11 = q00[keep], q10[keep], q01[keep], q11[keep]
        F = np.vstack([np.stack([q00, q10, q11], 1), np.stack([q00, q11, q01], 1)])
        return V.reshape(-1, 3), F

    def progress_text(self):
        return 100.0 * self.cut[self.mask].sum() / max(self.mask.sum(), 1)


def tool_mesh(Rt, L_flute, L_shank, n=40, flutes=4):
    """Schaftfräser entlang +z (Spitze bei z=0): Ecken, Dreiecke, Farbe je Dreieck."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, z: np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(n, z)])
    rs = 0.92 * Rt
    L2 = L_flute + L_shank
    V = np.vstack([ring(Rt, 0), ring(Rt, L_flute), ring(rs, L_flute), ring(rs, L2), [[0, 0, 0]], [[0, 0, L2]]])
    F, C = [], []
    gold, dark, steel = (0.86, 0.70, 0.28, 1), (0.35, 0.28, 0.10, 1), (0.70, 0.72, 0.76, 1)
    for i in range(n):
        j = (i + 1) % n
        stripe = (i * flutes // n) != (j * flutes // n) or (i * flutes) % n < 2
        for base, col in ((0, dark if stripe else gold), (2 * n, steel)):
            F += [[base + i, base + n + i, base + n + j], [base + i, base + n + j, base + j]]
            C += [col, col]
        F += [[4 * n, j, i]]
        C += [dark]
        F += [[4 * n + 1, 3 * n + i, 3 * n + j], [n + i, 2 * n + i, 2 * n + j], [n + i, 2 * n + j, n + j]]
        C += [steel, steel, steel]
    return V, np.array(F), np.array(C, float)


def _outer_radius_fn(model):
    """Radius der gedrehten Außenkontur in Abhängigkeit von a (vektorisiert, 0 außerhalb des Teils)."""
    lc = lathe_contours(model)
    if not lc or not lc["outer"]:
        return lambda a: np.zeros_like(np.asarray(a, float))
    P = np.array([(p[0], p[1]) for p in _profile_samples(lc["outer"])], float)

    def fn(a):
        a = np.asarray(a, float)
        out = np.zeros_like(a)
        for p, q in zip(P[:-1], P[1:]):
            lo, hi = min(p[0], q[0]), max(p[0], q[0])
            if hi - lo < 1e-9:
                m = np.abs(a - lo) < 1e-6
                out[m] = np.maximum(out[m], max(p[1], q[1]))
                continue
            m = (a >= lo - 1e-9) & (a <= hi + 1e-9)
            out[m] = np.maximum(out[m], p[1] + (a[m] - p[0]) / (q[0] - p[0]) * (q[1] - p[1]))
        return out
    return fn


def _trace_boundary(mask):
    """Rand einer zusammenhängenden Zellfläche als geschlossene Zellfolge (Moore-Nachbarschaft)."""
    nz = np.argwhere(mask)
    if not len(nz):
        return np.zeros((0, 2), int)
    start = tuple(nz[np.lexsort((nz[:, 1], nz[:, 0]))][0])
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    H, W = mask.shape
    ok = lambda p: 0 <= p[0] < H and 0 <= p[1] < W and mask[p]
    out = [start]
    cur, back = start, 6                           # von „links“ gekommen
    for _ in range(4 * mask.size):
        found = False
        for k in range(8):
            d = (back + 1 + k) % 8
            q = (cur[0] + nb[d][0], cur[1] + nb[d][1])
            if ok(q):
                back = (d + 4) % 8
                cur = q
                found = True
                break
        if not found or (cur == start and len(out) > 2):
            break
        out.append(cur)
    return np.array(out)


class PlaneStock(_StockBase):
    """Fräsmerkmal mit ebenem Grund am Drehteil (Nut, Schlüsselfläche, Abflachung): Restmaterial als
    Höhenfeld über dem Grund, Fräser senkrecht zum Grund. Bahn: schichtweise Zickzack + Kontur."""

    def __init__(self, model, feat, floor_face, g=None):
        mo, md = main_axis(model)
        f = model.faces[floor_face]
        n = unit(f.surf.n - (f.surf.n @ md) * md)
        cen = f.triangles.reshape(-1, 3).mean(0)
        foot = mo + ((cen - mo) @ md) * md
        if (cen - foot) @ n < 0:
            n = -n
        h = float((cen - foot) @ n)
        vdir = np.cross(n, md)
        self.mo, self.md, self.n, self.vdir, self.h = mo, md, n, vdir, h
        T = f.triangles - mo
        U = T @ md
        Vv = T @ vdir
        span = max(U.max() - U.min(), Vv.max() - Vv.min())
        g = g or float(np.clip(span / 180.0, 0.08, 0.4))
        marg = 0.25 * span + 3 * g
        self.u = np.arange(U.min() - marg, U.max() + marg + g, g)
        self.v = np.arange(Vv.min() - marg, Vv.max() + marg + g, g)
        UU, VV = np.meshgrid(self.u, self.v, indexing="ij")
        self.UU, self.VV, self.g = UU, VV, g
        # Grundfläche rastern
        mask = np.zeros(UU.shape, bool)
        for tu, tv in zip(U, Vv):
            i0, i1 = np.searchsorted(self.u, [tu.min() - g, tu.max() + g])
            j0, j1 = np.searchsorted(self.v, [tv.min() - g, tv.max() + g])
            if i1 <= i0 or j1 <= j0:
                continue
            pu, pv = UU[i0:i1, j0:j1], VV[i0:i1, j0:j1]
            d0 = (tu[1] - tu[0]) * (pv - tv[0]) - (tv[1] - tv[0]) * (pu - tu[0])
            d1 = (tu[2] - tu[1]) * (pv - tv[1]) - (tv[2] - tv[1]) * (pu - tu[1])
            d2 = (tu[0] - tu[2]) * (pv - tv[2]) - (tv[0] - tv[2]) * (pu - tu[2])
            e = 1e-9
            ins = ((d0 >= -e) & (d1 >= -e) & (d2 >= -e)) | ((d0 <= e) & (d1 <= e) & (d2 <= e))
            mask[i0:i1, j0:j1] |= ins
        Rp = _outer_radius_fn(model)(self.u)[:, None]
        top = np.sqrt(np.clip(Rp ** 2 - VV ** 2, 0, None)) - h
        self.mask = mask & (top > 1e-3)
        self.top0 = np.where(self.mask, np.clip(top, 0, None), 0.0)
        self.wall = (~mask) & (top > 0.05)              # Material neben dem Grund (Schulter, Nutwand)
        self.Hmax = float(self.top0.max()) if self.mask.any() else 0.0
        self.feat = feat
        self.poses = np.zeros((1, 6))
        self.tips = np.zeros((1, 3))
        self.axes = n[None, :]
        self.reset()

    def _reset_state(self):
        self.top = self.top0.copy()

    def to3d(self, u, v, z):
        return self.mo + u * self.md + v * self.vdir + (self.h + z) * self.n

    def _allowed(self, Rc):
        """Zellen, auf denen die Fräsermitte stehen darf (Abstand zu Wänden ≥ Rc)."""
        if not hasattr(self, "_wdist"):
            wb = self.wall & ~np.all([np.roll(self.wall, s, ax) for ax in (0, 1) for s in (1, -1)], axis=0)
            W = np.column_stack([self.UU[wb], self.VV[wb]])
            dist = np.full(self.UU.shape, np.inf)
            if len(W):
                q = np.column_stack([self.UU[self.mask], self.VV[self.mask]])
                dd = np.empty(len(q))
                for k in range(0, len(q), 4000):
                    dd[k:k + 4000] = np.sqrt(((q[k:k + 4000, None, :] - W[None, :, :]) ** 2).sum(-1)).min(1)
                dist[self.mask] = dd
            self._wdist = dist
        return self.mask & (self._wdist >= Rc - 0.5 * self.g)

    def plan(self, Rt, ap, ae, safety=2.0, aw=0.0, af=0.0, finish=None):
        """Bahn erzeugen (Fräsermitte u, v, Höhe z über Grund, Werkzeug-Nr.).
        Schruppen mit Fräser Rt, Aufmaß aw an den Wänden und af am Grund; optional
        finish=dict(Rt=…, ap=…, ae=…): Schlichten mit 2. Fräser – erst Grund (Zickzack), dann Wand (Kontur)."""
        allowed = self._allowed(Rt + aw)
        if not allowed.any():
            raise ValueError(f"Fräser Ø{fmt(2 * Rt)} mit Aufmaß {fmt(aw)} passt nicht in dieses Merkmal.")
        if af >= self.Hmax:
            raise ValueError("Aufmaß am Grund ist größer als die Tiefe.")
        z_safe = self.Hmax + safety
        pts = []

        def go(u, v, z, kind, t):
            pts.append((u, v, z, kind, t))

        def approach(u, v, z, t):
            if pts:
                go(pts[-1][0], pts[-1][1], z_safe, "luft", t)
            go(u, v, z_safe, "luft", t)
            go(u, v, z, "tauchen", t)

        def zigzag(alw, z, ae, Rc, t, need_z):
            ae = min(ae, 1.8 * Rc)
            jv = np.nonzero(alw.any(0))[0]
            v_lo, v_hi = self.v[jv[0]], self.v[jv[-1]]
            nrow = max(1, int(math.ceil((v_hi - v_lo) / ae - 1e-9)) + 1)
            vs = np.linspace(v_lo, v_hi, nrow) if nrow > 1 else np.array([0.5 * (v_lo + v_hi)])
            rows = []
            for vv in vs:
                j = int(jv[np.argmin(np.abs(self.v[jv] - vv))])
                if not rows or j != rows[-1]:
                    rows.append(j)
            reach = int(math.ceil(Rc / self.g))
            first, flip = True, False
            for j in rows:
                lo, hi = max(j - reach, 0), min(j + reach + 1, len(self.v))
                idx = np.nonzero(alw[:, j])[0]
                need = (self.top0[:, lo:hi] > need_z + 1e-6).any(1)
                if not len(idx) or not need[idx].any():
                    continue
                runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                if flip:
                    runs = [r[::-1] for r in runs[::-1]]
                for r in runs:
                    u0, u1 = self.u[r[0]], self.u[r[-1]]
                    if first or abs(pts[-1][1] - self.v[j]) > ae * 1.6 + 1e-6 or len(runs) > 1:
                        approach(u0, self.v[j], z, t)
                        first = False
                    else:
                        go(u0, self.v[j], z, "quer", t)
                    go(u1, self.v[j], z, "fraesen", t)
                flip = not flip

        def contour(alw, z, t):
            bnd = _trace_boundary(alw)
            if len(bnd) <= 2:
                return
            B = bnd[::max(1, int(round(0.4 / self.g)))]
            bu, bv = self.u[B[:, 0]], self.v[B[:, 1]]
            approach(bu[0], bv[0], z, t)
            for a, b in zip(bu[1:], bv[1:]):
                go(a, b, z, "kontur", t)
            go(bu[0], bv[0], z, "kontur", t)

        def levels(ap, z_end):
            zs, z = [], self.Hmax
            while z > z_end + 1e-6:
                z = max(z - ap, z_end)
                zs.append(z)
            return zs or [z_end]

        zs = levels(ap, af)
        for z in zs:
            zigzag(allowed, z, ae, Rt, 0, z)
            contour(allowed, z, 0)
        radii = [Rt]
        info = dict(layers=len(zs), finish_layers=0)
        if finish:
            R2 = finish["Rt"]
            alw2 = self._allowed(R2)
            if not alw2.any():
                raise ValueError(f"Schlichtfräser Ø{fmt(2 * R2)} passt nicht in dieses Merkmal.")
            go(pts[-1][0], pts[-1][1], z_safe, "luft", 0)
            go(pts[-1][0], pts[-1][1], z_safe, "wechsel", 1)
            if af > 1e-9:
                zigzag(alw2, 0.0, finish.get("ae", 1.2 * R2), R2, 1, -1.0)
            ap2 = finish.get("ap") or self.Hmax
            zs2 = levels(ap2, 0.0)
            for z in zs2:
                contour(alw2, z, 1)
            radii.append(R2)
            info["finish_layers"] = len(zs2)
        go(pts[-1][0], pts[-1][1], z_safe, "luft", pts[-1][4])
        self.path_pts = pts
        self.radii = radii
        self.Rt = max(radii)
        P = np.array([(u, v, z, t) for u, v, z, _, t in pts], float)
        P3 = np.array([self.to3d(*p[:3]) for p in P])
        dense, self.block = _dense_poses(np.hstack([P3, P]), max(0.3 * min(radii), 0.15))
        self.poses = dense
        self.tips = dense[:, :3]
        self.tool_idx = np.round(dense[:, 6]).astype(int)
        self.axes = np.repeat(self.n[None, :], len(dense), axis=0)
        self.reset()
        Lp = float(np.linalg.norm(np.diff(P3, axis=0), axis=1).sum())
        info.update(blocks=len(pts), length=Lp, depth=self.Hmax)
        return info

    def _cut(self, j):
        u, v, z = self.poses[j, 3:6]
        R = self.radii[self.tool_idx[j]] if hasattr(self, "tool_idx") else self.Rt
        if z >= self.Hmax:
            return False
        i = np.nonzero(np.abs(self.u - u) <= R + 1e-9)[0]
        k = np.nonzero(np.abs(self.v - v) <= R + 1e-9)[0]
        if not len(i) or not len(k):
            return False
        si, sk = slice(i[0], i[-1] + 1), slice(k[0], k[-1] + 1)
        m = ((self.UU[si, sk] - u) ** 2 + (self.VV[si, sk] - v) ** 2 <= R ** 2 + 1e-9) & self.mask[si, sk]
        upd = m & (self.top[si, sk] > z + 1e-9)
        if not upd.any():
            return False
        self.top[si, sk][upd] = max(z, 0.0)
        return True

    def mesh(self):
        done = (self.top <= 1e-3) | ~self.mask
        z = np.where(done, 0.01, self.top)
        V = (self.mo[None, None, :] + self.UU[..., None] * self.md[None, None, :]
             + self.VV[..., None] * self.vdir[None, None, :] + (self.h + z)[..., None] * self.n[None, None, :])
        nu, nv = z.shape
        idx = np.arange(nu * nv).reshape(nu, nv)
        q00, q10, q01, q11 = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
        dn, mk = done.ravel(), self.mask.ravel()
        keep = ~(dn[q00] & dn[q10] & dn[q01] & dn[q11]) & (mk[q00] & mk[q10] & mk[q01] & mk[q11])
        q00, q10, q01, q11 = q00[keep], q10[keep], q01[keep], q11[keep]
        F = np.vstack([np.stack([q00, q10, q11], 1), np.stack([q00, q11, q01], 1)])
        return V.reshape(-1, 3), F

    def progress_text(self):
        return 100.0 * (1.0 - self.top[self.mask].sum() / max(self.top0[self.mask].sum(), 1e-9))


def plane_floor_face(model, feat):
    """Grundfläche eines Fräsmerkmals (Planfläche parallel zur Achse, am nächsten zur Achse)."""
    mo, md = main_axis(model)
    best = None
    for i in feat["faces"]:
        f = model.faces[i]
        if not isinstance(f.surf, PlaneSurf) or abs(f.surf.n @ md) > 1e-3:
            continue
        cen = f.triangles.reshape(-1, 3).mean(0)
        foot = mo + ((cen - mo) @ md) * md
        nn = unit(f.surf.n - (f.surf.n @ md) * md)
        if abs(unit(cen - foot) @ nn) < 0.5:
            continue                                   # Seitenwand, kein Grund
        d = abs((cen - foot) @ nn)
        if best is None or d < best[0]:
            best = (d, i)
    return None if best is None else best[1]
