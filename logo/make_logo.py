"""Logo für Viewer 3D: Sechseck, isometrischer Würfel, Umlauf-Pfeil. Erzeugt PNG/ICO/SVG und Schriftzug."""
import math, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtGui, QtCore
from PyQt6.QtSvg import QSvgGenerator
QF = QtCore.QPointF
OUT = sys.argv[1]
app = QtGui.QGuiApplication([])

DARK = QtGui.QColor("#2b3446")
B_TOP1, B_TOP2 = QtGui.QColor("#8cc3ff"), QtGui.QColor("#4f97f7")
B_LEFT1, B_LEFT2 = QtGui.QColor("#3d86ee"), QtGui.QColor("#2667d6")
B_RIGHT1, B_RIGHT2 = QtGui.QColor("#2462cf"), QtGui.QColor("#173f9a")
ARC1, ARC2 = QtGui.QColor("#4aa0ff"), QtGui.QColor("#1f5fd0")


def bez(p0, p1, p2, p3, t):
    u = 1 - t
    return (u**3 * p0[0] + 3 * u*u*t * p1[0] + 3 * u*t*t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u*u*t * p1[1] + 3 * u*t*t * p2[1] + t**3 * p3[1])


def swoosh_path():
    """Sich verbreiternder Bogen mit Pfeilspitze (Koordinaten im 256er-Raster)."""
    P = [(40, 172), (66, 220), (178, 212), (212, 118)]
    n = 60
    pts = [bez(*P, i / n) for i in range(n + 1)]
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        j0, j1 = max(i - 1, 0), min(i + 1, n)
        dx, dy = pts[j1][0] - pts[j0][0], pts[j1][1] - pts[j0][1]
        L = math.hypot(dx, dy) or 1
        nx, ny = -dy / L, dx / L
        t = i / n
        w = 1.2 + 9.5 * math.sin(min(t, 0.92) / 0.92 * math.pi / 2)
        left.append((x + nx * w / 2, y + ny * w / 2))
        right.append((x - nx * w / 2, y - ny * w / 2))
    path = QtGui.QPainterPath(QF(*left[0]))
    for p in left[1:]:
        path.lineTo(QF(*p))
    for p in reversed(right):
        path.lineTo(QF(*p))
    path.closeSubpath()
    # Pfeilspitze
    ex, ey = pts[-1]
    dx, dy = pts[-1][0] - pts[-4][0], pts[-1][1] - pts[-4][1]
    L = math.hypot(dx, dy); dx, dy = dx / L, dy / L
    nx, ny = -dy, dx
    tip = (ex + dx * 22, ey + dy * 22)
    a = (ex + nx * 17, ey + ny * 17)
    b = (ex - nx * 17, ey - ny * 17)
    head = QtGui.QPainterPath(QF(*tip)); head.lineTo(QF(*a)); head.lineTo(QF(*b)); head.closeSubpath()
    return path.united(head)


def hexagon(r_out, r_in, cx=128, cy=128):
    def poly(r):
        pp = QtGui.QPainterPath()
        for k in range(6):
            ang = math.radians(-90 + 60 * k)
            p = QF(cx + r * math.cos(ang), cy + r * math.sin(ang))
            pp.moveTo(p) if k == 0 else pp.lineTo(p)
        pp.closeSubpath()
        return pp
    return poly(r_out).subtracted(poly(r_in))


def cube_faces(cx=128, cy=130, s=58):
    h = s * math.sin(math.radians(30))
    c = s * math.cos(math.radians(30))
    T = QF(cx, cy - 2 * h - 0); C = QF(cx, cy); Lp = QF(cx - c, cy - h); R = QF(cx + c, cy - h)
    BL = QF(cx - c, cy - h + s); BR = QF(cx + c, cy - h + s); B = QF(cx, cy + s)
    def poly(*ps):
        p = QtGui.QPainterPath(ps[0])
        for q in ps[1:]:
            p.lineTo(q)
        p.closeSubpath(); return p
    return poly(T, R, C, Lp), poly(Lp, C, B, BL), poly(C, R, BR, B)


def paint(p, size, with_gap=True):
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.scale(size / 256.0, size / 256.0)
    sw = swoosh_path()
    gap = QtGui.QPainterPathStroker(); gap.setWidth(14); gap.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    halo = gap.createStroke(sw).united(sw)
    hexp = hexagon(116, 96)
    if with_gap:
        hexp = hexp.subtracted(halo)
    hg = QtGui.QLinearGradient(QF(0, 10), QF(0, 246))
    hg.setColorAt(0, QtGui.QColor("#45526b")); hg.setColorAt(1, QtGui.QColor("#262e3e"))
    p.fillPath(hexp, QtGui.QBrush(hg))
    # feiner heller Rand, damit das Zeichen auch auf dunklem Hintergrund (Taskleiste) sichtbar bleibt
    rim = QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 2.0); rim.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(rim); p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(hexp); p.setPen(Qt.PenStyle.NoPen)
    top, left, right = cube_faces()
    def grad(path, c1, c2, vertical=True):
        r = path.boundingRect()
        g = QtGui.QLinearGradient(r.topLeft(), r.bottomLeft() if vertical else r.bottomRight())
        g.setColorAt(0, c1); g.setColorAt(1, c2)
        return QtGui.QBrush(g)
    for path, c1, c2 in ((top, B_TOP1, B_TOP2), (left, B_LEFT1, B_LEFT2), (right, B_RIGHT1, B_RIGHT2)):
        pp = path.subtracted(halo) if with_gap else path
        p.fillPath(pp, grad(path, c1, c2))
    # helle Kanten
    pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 2.2)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    for path in (top,):
        p.drawPath(path.subtracted(halo) if with_gap else path)
    r = sw.boundingRect()
    g = QtGui.QLinearGradient(r.bottomLeft(), r.topRight())
    g.setColorAt(0, ARC2); g.setColorAt(1, ARC1)
    p.fillPath(sw, QtGui.QBrush(g))


Qt = QtCore.Qt
os.makedirs(OUT, exist_ok=True)
imgs = {}
for size in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256, 512, 1024):
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img); paint(p, size); p.end()
    imgs[size] = img
    if size in (256, 512, 1024):
        img.save(os.path.join(OUT, f"Viewer3D_{size}.png"))
# SVG
gen = QSvgGenerator(); gen.setFileName(os.path.join(OUT, "Viewer3D.svg"))
gen.setSize(QtCore.QSize(256, 256)); gen.setViewBox(QtCore.QRect(0, 0, 256, 256)); gen.setTitle("Viewer 3D")
p = QtGui.QPainter(gen); paint(p, 256); p.end()
# ICO über Pillow (mehrere Größen)
from PIL import Image
import io
def to_pil(img):
    ba = QtCore.QByteArray(); buf = QtCore.QBuffer(ba); buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG"); return Image.open(io.BytesIO(bytes(ba))).convert("RGBA")
big = to_pil(imgs[256])
sizes = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]
frames = {s[0]: to_pil(imgs[s[0]]) for s in sizes}
big.save(os.path.join(OUT, "Viewer3D.ico"), sizes=sizes, append_images=[frames[s[0]] for s in sizes[:-1]])

# Schriftzug (Logo + "3D VIEWER")
fid = QtGui.QFontDatabase.addApplicationFont("/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf")
fam = QtGui.QFontDatabase.applicationFontFamilies(fid)[0]
for name, W, H, icon, fs in (("Viewer3D_Logo.png", 1400, 360, 300, 150), ("Viewer3D_Logo_klein.png", 700, 180, 150, 75)):
    img = QtGui.QImage(W, H, QtGui.QImage.Format.Format_ARGB32_Premultiplied); img.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.save(); p.translate(20 * W / 1400, (H - icon) / 2); paint(p, icon); p.restore()
    f = QtGui.QFont(fam); f.setWeight(QtGui.QFont.Weight.ExtraBold); f.setPixelSize(fs); f.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 98)
    p.setFont(f); fm = QtGui.QFontMetrics(f)
    x = 20 * W / 1400 + icon + 40 * W / 1400
    y = H / 2 + fm.capHeight() / 2
    p.setPen(QtGui.QColor("#1f6fe5")); p.drawText(QF(x, y), "3D")
    x += fm.horizontalAdvance("3D") + fs * 0.28
    p.setPen(DARK); p.drawText(QF(x, y), "VIEWER")
    p.end()
    img.save(os.path.join(OUT, name))
print("ok", os.listdir(OUT))
