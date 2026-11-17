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
                rows.append(("Ø groß", f"{fmt(2 * max(R1, R2))} mm"))
                rows.append(("Ø klein", f"{fmt(2 * min(R1, R2))} mm"))
                rows.append(("Ø-Differenz", f"{fmt(2 * abs(R1 - R2))} mm"))
                a_mid = (max(i1["lo"], lo2) + min(i1["hi"], hi2)) / 2
                if max(i1["lo"], lo2) > min(i1["hi"], hi2):
                    a_mid = (i1["lo"] + i1["hi"]) / 2
                    gap = max(i1["lo"], lo2) - min(i1["hi"], hi2)
                    rows.append(("Abstand (axial)", f"{fmt(gap)} mm"))
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
    mid = 0.5 * (lo + hi)
    res = [lo, hi]
    for side in (0, 1):
        sel = bnd[(((P[bnd[:, 0]] - o) @ d) > mid) == bool(side)]
        if len(sel) < 8:
            continue
        ids = np.unique(sel)
        h = (P[ids] - o) @ d
        if h.max() - h.min() < 1e-6:
            continue                    # Rand senkrecht zur Achse – Wert stimmt bereits
        # nur ein einziger geschlossener Rand rund um die Achse zählt (keine Durchbrüche von Querbohrungen)
        parent = {int(v): int(v) for v in ids}

        def root(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for x, y in sel:
            parent[root(int(x))] = root(int(y))
        if len({root(int(v)) for v in ids}) != 1:
            continue
        R = float(np.linalg.norm((P[ids] - o) - np.outer(h, d), axis=1).max())
        votes = {}
        for k in map(tuple, uk[ids]):
            for j in vf.get(k, ()):
                if j not in own:
                    votes[j] = votes.get(j, 0) + 1
        if not votes:
            continue
        j = max(votes, key=votes.get)
        surf = model.faces[j].surf
        if surf is None:
            continue
        a = _line_surface_root(surf, o, d, h.min() - 0.25 * R, h.max() + 0.25 * R)
        if a is not None and h.min() - 1e-3 <= a <= h.max() + 1e-3:
            res[side] = a
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
            cl = _closest_line_line(h.axis_p, h.axis_d, o, d)
            if cl is None or np.linalg.norm(cl[0] - cl[1]) > 0.05:
                continue
            # Lage der Bohrung am Umfang: von der Mittelachse zum Eintrittspunkt
            q = h.entry - o
            q = q - (q @ d) * d
            if np.linalg.norm(q) < 1e-6:
                q = h.axis_d
            th = round(math.degrees(math.atan2(q @ e2, q @ e1)), 4) % 360.0
            lvl = float((cl[1] - o) @ d)
            for g in groups:
                if g.kind == "radial" and abs(g.level - lvl) < 0.5:
                    g.holes.append(i)
                    g.angles.append(th)
                    break
            else:
                groups.append(HoleGroup("radial", [i], lvl, 0.0, [th]))
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
    groups.sort(key=lambda g: (g.kind, g.level))
    return groups, (o, d, e1, e2)


def entry_info(model, h, wcs):
    """Eintrittspunkt einer Bohrung für die Maschine: Lage im Nullpunkt-KS (X als Durchmesser
    um die Längsachse, Z), Winkel am Umfang (C) und Neigung der Bohrungsachse zur Längsachse (B)."""
    o, d = main_axis(model)
    e1, e2 = angle_frame(d, wcs)
    p = np.asarray(h.entry, float)
    q = p - o
    r = float(np.linalg.norm(q - (q @ d) * d))
    c = math.degrees(math.atan2(q @ e2, q @ e1)) % 360.0 if r > 1e-6 else 0.0
    pw = wcs.to_wcs(p)
    tool = -unit(h.axis_d)              # Werkzeugrichtung: von außen ins Material
    b = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(tool @ d))))))
    return {"point": p, "dia": 2 * r, "z": float(pw[2]), "c": c, "b": b,
            "wcs": pw, "tool": wcs.dir_to_wcs(tool)}


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
            out.append(dict(typ="Fase" if L < 3.0 else "Kegel", wert=f"{fmt(s['angle'], 3)}°", z0=z0, z1=z1, L=z1 - z0,
                            info=f"Ø {fmt(min(d0, d1))} … Ø {fmt(max(d0, d1))} · {side}", face=s["face"], key=(2, 0)))
        elif s["kind"] == "ARC":
            out.append(dict(typ="Radius", wert=f"R {fmt(s['r'])}", z0=z0, z1=z1, L=z1 - z0,
                            info=f"Mitte Z {fmt(Z(s['c'][0]))} · X Ø {fmt(2 * s['c'][1])} · "
                                 + ("Hohlkehle/Einstich" if f.concavity == "1" else "Rundung"),
                            face=s["face"], key=(3, 0)))
    out.sort(key=lambda d: (-d["z1"], d["key"]))
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
