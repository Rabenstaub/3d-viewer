"""Erklär-Grafiken für das Handbuch (hilfe/bilder/d_*.png). Aufruf: python make_hilfe_bilder.py"""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc, FancyArrowPatch, Wedge

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bilder")
os.makedirs(OUT, exist_ok=True)
PART, EDGE, DIM, RED, ORG, GRN, AX = "#d9dde4", "#2b2f36", "#1d4fb8", "#d0021b", "#e07800", "#1a8a3a", "#7a7f88"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def fig(w=9, h=5):
    f, ax = plt.subplots(figsize=(w, h), dpi=110)
    ax.set_aspect("equal")
    ax.axis("off")
    return f, ax


def save(f, name):
    f.tight_layout()
    f.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(f)


def arrow(ax, p, q, col=DIM, lw=1.3, both=True, ms=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms, color=col, lw=lw,
                                 shrinkA=0, shrinkB=0))


def dim(ax, p, q, text, off=(0, 0), col=DIM, ext=True, tpos=0.5, toff=(0, 0.6), rot=None, size=11):
    p, q = np.array(p, float), np.array(q, float)
    o = np.array(off, float)
    a, b = p + o, q + o
    if ext and np.linalg.norm(o) > 0:
        for s, t in ((p, a), (q, b)):
            ax.plot([s[0], t[0] + 0.15 * np.sign(o[0])], [s[1], t[1] + 0.15 * np.sign(o[1])], color=col, lw=0.8)
    arrow(ax, a, b, col)
    m = a + tpos * (b - a) + np.array(toff)
    ax.text(m[0], m[1], text, color=col, ha="center", va="center", fontsize=size, rotation=rot or 0,
            bbox=dict(fc="white", ec="none", pad=1.0))


def label(ax, x, y, t, col=EDGE, size=11, ha="center", va="center", bold=False, box=False):
    ax.text(x, y, t, color=col, fontsize=size, ha=ha, va=va, fontweight="bold" if bold else "normal",
            bbox=dict(fc="white", ec=col if box else "none", pad=2.5, boxstyle="round") if box else None)


def axis_line(ax, x0, x1, y=0):
    ax.plot([x0, x1], [y, y], color=AX, lw=1, ls=(0, (10, 3, 2, 3)))


def half_profile(ax, pts, mirror=True):
    """Drehteil: Profil (z, r) oben und gespiegelt unten."""
    P = np.array(pts, float)
    full = np.vstack([P, P[::-1] * [1, -1]]) if mirror else P
    ax.add_patch(Polygon(full, closed=True, fc=PART, ec=EDGE, lw=1.6))


# 1 Koordinaten an der Drehmaschine
f, ax = fig(10, 5.2)
prof = [(0, 0), (0, 9), (1, 10), (30, 10), (30, 15), (60, 15), (60, 0)]
half_profile(ax, prof)
axis_line(ax, -12, 72)
arrow(ax, (0, 0), (75, 0), RED, both=False, lw=2)
label(ax, 77, 0, "Z", RED, 14, bold=True)
arrow(ax, (0, 0), (0, 22), GRN, both=False, lw=2)
label(ax, 0, 23.8, "X", GRN, 14, bold=True)
ax.plot(0, 0, "o", color=RED, ms=8)
label(ax, -2, -3, "Z0 / Nullpunkt", RED, 10, ha="right")
dim(ax, (45, -15), (45, 15), "X = Ø 30", off=(0, 0), toff=(5.5, 0))
dim(ax, (15, 0), (15, 10), "r = 5", toff=(3.5, 0), col=GRN)
label(ax, 45, -20.5, "X wird als Durchmesser angegeben (Ø), Radius = halber Wert", AX, 10)
# C-Achse als Ansicht von vorne
cx, cy, R = 92, 3, 13
ax.add_patch(Circle((cx, cy), R, fc=PART, ec=EDGE, lw=1.6))
ax.plot([cx, cx + R + 4], [cy, cy], color=RED, lw=1.5)
label(ax, cx + R + 6, cy, "C0°", RED, 11)
ax.add_patch(Arc((cx, cy), 2 * R + 6, 2 * R + 6, theta1=0, theta2=90, color=DIM, lw=1.6))
arrow(ax, (cx + 0.5, cy + R + 3), (cx - 0.5, cy + R + 3), DIM, both=False)
label(ax, cx + 13, cy + 13, "C+", DIM, 11)
label(ax, cx, cy - R - 4, "Blick auf die Stirnseite", AX, 10)
ax.set_xlim(-30, 115); ax.set_ylim(-24, 27)
save(f, "d_koordinaten.png")

# 2 Maßliste: ab links / ab rechts
f, ax = fig(10, 4.6)
prof = [(0, 0), (0, 8), (20, 8), (20, 12), (45, 12), (45, 10), (60, 10), (60, 0)]
half_profile(ax, prof)
axis_line(ax, -5, 66)
dim(ax, (0, 12), (20, 12), "ab links 20", off=(0, 5))
dim(ax, (20, -12), (60, -12), "ab rechts 40", off=(0, -5), toff=(0, -1.4))
dim(ax, (0, 12), (60, 12), "Gesamtlänge 60", off=(0, 9.5))
ax.plot([20, 20], [-12, 12], color=RED, lw=3)
label(ax, 21.5, 0, "Planfläche\nZ 20", RED, 10, ha="left")
ax.set_xlim(-6, 66); ax.set_ylim(-21, 24)
save(f, "d_massliste.png")

# 3 Radius-Mittelpunkt und scharfe Ecke
f, ax = fig(9, 5)
R = 6
prof = [(0, 0), (0, 10), (30 - R, 10)]
t = np.linspace(-math.pi / 2, 0, 30)
prof += [(30 - R + R * math.cos(a), 10 + R + R * math.sin(a)) for a in t][1:]
prof += [(30, 22), (50, 22), (50, 0)]
ax.add_patch(Polygon(prof, closed=True, fc=PART, ec=EDGE, lw=1.6))
axis_line(ax, -4, 56)
M = (30 - R, 10 + R)
ax.plot(*M, "+", color=RED, ms=14, mew=2)
label(ax, M[0] - 1, M[1] + 2.5, "M  (Z / X Ø)", RED, 10, ha="right")
ax.plot([M[0], M[0] + R * math.cos(-math.pi / 4)], [M[1], M[1] + R * math.sin(-math.pi / 4)], color=RED, lw=1.2)
label(ax, M[0] + 4.6, M[1] - 1.2, "R 6", RED, 11)
ax.plot([30 - R - 2, 30], [10, 10], color=ORG, lw=1.3, ls="--")
ax.plot([30, 30], [10, 10 + R + 2], color=ORG, lw=1.3, ls="--")
ax.plot(30, 10, "o", color=ORG, ms=8)
label(ax, 31.5, 8, "scharfe Ecke\n(Radius „weggedacht“)", ORG, 10, ha="left", va="top")
dim(ax, (0, 0), (M[0], 0), "Mitte Z", off=(0, -4))
dim(ax, (M[0], 0), (M[0], M[1]), "X = 2 × Abstand", toff=(-5.2, 0), size=10)
ax.set_xlim(-5, 57); ax.set_ylim(-8, 26)
save(f, "d_radius.png")

# 4 Kegel
f, ax = fig(9.5, 4.8)
prof = [(0, 0), (0, 6), (30, 10), (42, 10), (42, 0)]
half_profile(ax, prof)
axis_line(ax, -6, 48)
dim(ax, (0, -6), (0, 6), "d", toff=(-2.2, 0))
dim(ax, (30, -10), (30, 10), "D", toff=(2.2, 0))
dim(ax, (0, 10), (30, 10), "L", off=(0, 4))
ax.plot([0, 34], [6, 6], color=AX, lw=0.9, ls="--")
ax.add_patch(Arc((0, 6), 22, 22, theta1=0, theta2=math.degrees(math.atan2(4, 30)), color=RED, lw=1.6))
label(ax, 13.5, 7.6, "α/2", RED, 11)
label(ax, 21, -17.5, "Verhältnis 1 : x  mit  x = L / (D − d)      Kegelwinkel = α      z. B. Morsekegel ≈ 1:20, Steilkegel 7:24",
      AX, 9.5)
ax.set_xlim(-8, 50); ax.set_ylim(-20, 17)
save(f, "d_kegel.png")

# 5 Einstich und Freistich
f, ax = fig(10, 4.8)
pts = [(0, 0), (0, 10), (12, 10), (12, 8), (16, 8), (16, 10), (34, 10)]
r = 1.2
t = np.linspace(-math.pi / 2, -math.pi, 12)
pts += [(34, 9.0)]
pts += [(36.2 + r * math.cos(a), 9.0 - 0 + r * math.sin(a) + 0) for a in np.linspace(math.pi, 1.5 * math.pi, 10)]
pts += [(37.8, 7.8 + 0.0), (39.4, 7.8 + 1.6 * math.tan(math.radians(15))), (40, 14), (55, 14), (55, 0)]
ax.add_patch(Polygon(pts, closed=True, fc=PART, ec=EDGE, lw=1.6))
axis_line(ax, -3, 60)
dim(ax, (12, 10), (16, 10), "Breite", off=(0, 3.2))
dim(ax, (17.5, 8), (17.5, 10), "Tiefe", toff=(2.8, 0), size=10)
label(ax, 14, 4.5, "Einstich", DIM, 11, bold=True)
label(ax, 38, 4.0, "Freistich\n(am Absatz)", RED, 11, bold=True)
label(ax, 46, 18.5, "Grundradius R + 15°-Flanke\n→ ähnlich DIN 509 Form E", RED, 9.5)
ax.annotate("", xy=(37, 7.9), xytext=(43, 16.6), arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlim(-3, 60); ax.set_ylim(-2, 22)
save(f, "d_einstich.png")

# 6 Werkzeug: Plattenwinkel
f, ax = fig(10, 5.4)
# Kontur: fällt in Vorschubrichtung (−Z = nach links) ab
ax.add_patch(Polygon([(0, 0), (0, 6), (16, 6), (30, 13), (60, 13), (60, 0)], closed=True, fc=PART, ec=EDGE, lw=1.6))
ax.add_patch(Arc((16, 6), 18, 18, theta1=0, theta2=math.degrees(math.atan2(7, 14)), color=RED, lw=1.6))
label(ax, 27.5, 8.2, "δ (fällt ab)", RED, 10)
ax.plot([16, 28], [6, 6], color=AX, lw=0.9, ls="--")
# Platte (55°) an der Kontur
tip = np.array([22, 9.2])
kr, eps = 93, 55
d1 = np.array([math.cos(math.radians(180 - kr + 90)), math.sin(math.radians(180 - kr + 90))])  # Hauptschneide nach oben
main = np.array([math.cos(math.radians(90 + (kr - 90))), math.sin(math.radians(90 + (kr - 90)))])
a_main = math.radians(180 - 93)          # Hauptschneide (κr 93° zur Vorschubrichtung −Z)
a_min = a_main - math.radians(eps)        # Nebenschneide
L = 11
p1 = tip + L * np.array([math.cos(a_main), math.sin(a_main)])
p2 = tip + L * np.array([math.cos(a_min), math.sin(a_min)])
ax.add_patch(Polygon([tip, p2, p2 + (p1 - tip), p1], closed=True, fc="#ffd9a8", ec=ORG, lw=1.8))
label(ax, tip[0] + 6.5, tip[1] + 9.5, "Platte ε 55°", ORG, 10)
ax.plot([tip[0], tip[0] + 14], [tip[1], tip[1]], color=AX, lw=0.9, ls="--")
ax.add_patch(Arc(tip, 12, 12, theta1=0, theta2=math.degrees(a_min), color=GRN, lw=1.6))
label(ax, tip[0] + 9.8, tip[1] + 2.5, "κ'", GRN, 12, bold=True)
arrow(ax, (12, 20), (2, 20), DIM, both=False, lw=2)
label(ax, 7, 21.8, "Vorschub −Z", DIM, 10)
label(ax, 30, -4.5, "Die Platte kommt hin, wenn κ' = 180° − κr − ε größer ist als δ (+2° Sicherheit).\n"
      "CNMG 80° → κ' 5°   ·   TNMG 60° → 27°   ·   DNMG 55° → 32°   ·   VBMT 35° → 52° bzw. 72,5°", AX, 9.5)
ax.set_xlim(-2, 62); ax.set_ylim(-8, 25)
save(f, "d_werkzeug.png")

# 7 Eintrittspunkt Schrägbohrung (Längsschnitt)
f, ax = fig(10, 5.4)
RS, TP = 5.0, (2.5, 4.330)
t30 = math.tan(math.radians(30)); t15 = math.tan(math.radians(15))
A1 = TP[0] + (7 - TP[1]) / t30; A2 = A1 + 3 / t15
pts = [(5 + RS * math.cos(a), RS * math.sin(a)) for a in np.linspace(math.pi, math.radians(120), 20)]
pts += [(A1, 7), (A2, 10), (34, 10), (34, -10), (A2, -10), (A1, -7)]
pts += [(5 + RS * math.cos(a), RS * math.sin(a)) for a in np.linspace(math.radians(240), math.pi, 20)]
ax.add_patch(Polygon(pts, closed=True, fc=PART, ec=EDGE, lw=1.6))
axis_line(ax, -4, 38)
ex = (TP[0] + (5.6 - TP[1]) / t30, 5.6)
d = np.array([1, -1]) / math.sqrt(2)
ax.plot([ex[0] - 7 * d[0], ex[0] + 7 * d[0]], [ex[1] - 7 * d[1], ex[1] + 7 * d[1]], color=DIM, lw=1.6, ls="--")
tt = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
ax.plot([ex[0] - 6 * tt[0], ex[0] + 7 * tt[0]], [ex[1] - 6 * tt[1], ex[1] + 7 * tt[1]], color=ORG, lw=1.6, ls="--")
ax.plot(*ex, "o", color=RED, ms=10, zorder=5)
label(ax, ex[0] - 3.5, ex[1] + 4.6, "Eintrittspunkt\n(roter Punkt)", RED, 10, bold=True)
label(ax, ex[0] - 6.6, ex[1] + 7.6 - 2, "", RED)
label(ax, ex[0] + 6.6, ex[1] - 6.8, "Bohrungsachse\nB 45°", DIM, 9.5)
label(ax, ex[0] + 7.7, ex[1] + 4.6, "30°-Flanke", ORG, 9.5)
dim(ax, (0, -10.5), (ex[0], -10.5), "Z 4,699", off=(0, -2))
ax.plot([ex[0], ex[0]], [ex[1], -12.5], color=DIM, lw=0.8)
dim(ax, (ex[0], 0), (ex[0], ex[1]), "", toff=(0, 0))
label(ax, ex[0] - 5.2, 2.4, "X Ø11,2\n(r 5,6)", DIM, 9.5)
ax.plot(0, 0, "o", color=RED, ms=6)
label(ax, -1, -1.6, "Z0", RED, 9, ha="right")
ax.set_xlim(-5, 38); ax.set_ylim(-16, 14)
save(f, "d_eintritt.png")

# 8 Außermittige Bohrung (Blick auf die Stirnseite)
f, ax = fig(8.5, 6)
R = 30
ax.add_patch(Circle((0, 0), R, fc=PART, ec=EDGE, lw=1.6))
yo = 8
ze = -math.sqrt(R * R - yo * yo)
ax.add_patch(Polygon([(yo - 3, ze - 2), (yo + 3, ze - 2), (yo + 3, ze + 10), (yo - 3, ze + 10)], closed=True,
                     fc="#f4c48e", ec=ORG, lw=1.4))
ax.plot([yo, yo], [ze - 12, ze + 12], color=DIM, lw=1.4, ls="--")
ax.plot(yo, ze, "o", color=RED, ms=10, zorder=5)
ax.plot([-R - 5, R + 5], [0, 0], color=AX, lw=0.8, ls=(0, (8, 3, 2, 3)))
ax.plot([0, 0], [-R - 14, R + 5], color=AX, lw=0.8, ls=(0, (8, 3, 2, 3)))
dim(ax, (0, ze - 9), (yo, ze - 9), "Y 8", toff=(0, -1.6))
dim(ax, (-10, 0), (-10, ze), "X/2 = 28,91", toff=(-6.5, 0), size=10)
arrow(ax, (yo, ze - 20), (yo, ze - 4), RED, both=False, lw=2)
label(ax, yo + 2, ze - 19, "Werkzeug von unten → C 180°", RED, 10, ha="left")
label(ax, 0, R + 8, "Bohrung geht nicht durch die Mitte: X (Ø), Y-Versatz, Z, C", AX, 10)
ax.set_xlim(-45, 50); ax.set_ylim(-55, 42)
save(f, "d_ausmittig.png")

# 9 Senkung, Bohrspitze, Tiefe bis voller Ø
f, ax = fig(9.5, 5)
W, D0, sk, d, depth, tip = 14, 0, 0.9, 5, 15, 5 / 2 / math.tan(math.radians(59))
hole = [(-d / 2 - sk, 0), (-d / 2, -sk), (-d / 2, -depth), (0, -depth - tip), (d / 2, -depth), (d / 2, -sk), (d / 2 + sk, 0)]
ax.add_patch(Polygon([(-W, 0)] + hole + [(W, 0), (W, -24), (-W, -24)], closed=True, fc=PART, ec=EDGE, lw=1.6))
ax.plot([0, 0], [3, -depth - tip - 3], color=AX, lw=0.8, ls=(0, (8, 3, 2, 3)))
dim(ax, (-d / 2 - sk, 0), (d / 2 + sk, 0), "Senkung Ø6,4  90°", off=(0, 3))
dim(ax, (-d / 2, -8), (d / 2, -8), "Ø5", toff=(0, 0.9), size=10)
dim(ax, (d / 2 + 1, 0), (d / 2 + 1, -depth), "Tiefe voller Ø 15", off=(4, 0), toff=(5.6, 0), size=10)
dim(ax, (-d / 2 - 1, 0), (-d / 2 - 1, -depth - tip), "Gesamttiefe", off=(-4, 0), toff=(-4.4, 0), size=10)
label(ax, 0, -depth - tip - 4, "Bohrspitze 118°, Länge 1,5", RED, 10)
ax.set_xlim(-W - 4, W + 8); ax.set_ylim(-25, 6)
save(f, "d_senkung.png")

# 10 Fräsen: Nut und Schlüsselfläche (Querschnitte)
f, ax = fig(10.5, 5)
R = 20
th = np.linspace(0, 2 * math.pi, 200)
b, t1 = 12, 5
pts = []
for a in np.linspace(math.radians(90) + math.asin(b / 2 / R), math.radians(450) - math.asin(b / 2 / R), 200):
    pts.append((R * math.cos(a), R * math.sin(a)))
pts += [(b / 2, math.sqrt(R * R - b * b / 4)), (b / 2, R - t1), (-b / 2, R - t1), (-b / 2, math.sqrt(R * R - b * b / 4))]
ax.add_patch(Polygon(pts, closed=True, fc=PART, ec=EDGE, lw=1.6))
dim(ax, (-b / 2, R), (b / 2, R), "Breite 12", off=(0, 4))
dim(ax, (b / 2 + 2, R), (b / 2 + 2, R - t1), "Tiefe 5\nab Ø", off=(3, 0), toff=(5, 0), size=10)
ax.plot([0, b / 2 + 6], [R, R], color=AX, lw=0.8, ls="--")
label(ax, 0, -R - 5, "Passfedernut am Ø40", AX, 10)
ox, Rs, sw = 58, 30, 25
a0 = math.asin(sw / Rs)
pts = [(ox + Rs * math.cos(a), Rs * math.sin(a)) for a in np.linspace(-a0, a0, 60)]
pts += [(ox + Rs * math.cos(a), Rs * math.sin(a)) for a in np.linspace(math.pi - a0, math.pi + a0, 60)]
ax.add_patch(Polygon(pts, closed=True, fc=PART, ec=EDGE, lw=1.6))
dim(ax, (ox + Rs + 3, -sw), (ox + Rs + 3, sw), "SW 50", toff=(4.5, 0))
label(ax, ox, -Rs - 5, "Schlüsselfläche am Ø60", AX, 10)
ax.set_xlim(-28, 100); ax.set_ylim(-38, 30)
save(f, "d_fraesen.png")

# 11 Gewinde
f, ax = fig(9.5, 5)
W, dk, dn, gt, kt = 14, 5.0, 6.0, 12, 15
tip = dk / 2 / math.tan(math.radians(59))
hole = [(-dk / 2, 0), (-dk / 2, -kt), (0, -kt - tip), (dk / 2, -kt), (dk / 2, 0)]
ax.add_patch(Polygon([(-W, 0)] + hole + [(W, 0), (W, -22), (-W, -22)], closed=True, fc=PART, ec=EDGE, lw=1.6))
for s in (-1, 1):
    ax.add_patch(Polygon([(s * dk / 2, 0), (s * dn / 2, 0), (s * dn / 2, -gt), (s * dk / 2, -gt)], closed=True,
                         fc="#7fcf8f", ec=GRN, lw=1.2))
dim(ax, (-dn / 2, 0), (dn / 2, 0), "M6 (Nenn-Ø)", off=(0, 3))
dim(ax, (dn / 2 + 1, 0), (dn / 2 + 1, -gt), "Gewindetiefe 12", off=(3, 0), toff=(5.8, 0), size=10)
dim(ax, (-dn / 2 - 1, 0), (-dn / 2 - 1, -kt), "Kernlochtiefe 15", off=(-3, 0), toff=(-6, 0), size=10)
label(ax, 0, -kt - tip - 3, "Kernloch Ø5 · grün = Bereich mit Gewinde", AX, 10)
ax.set_xlim(-W - 6, W + 9); ax.set_ylim(-23, 6)
save(f, "d_gewinde.png")

# 12 Bohrbild
f, ax = fig(7.5, 6.5)
ax.add_patch(Circle((0, 0), 30, fc=PART, ec=EDGE, lw=1.6))
ax.add_patch(Circle((0, 0), 10, fc="white", ec=EDGE, lw=1.2))
ax.add_patch(Circle((0, 0), 18, fc="none", ec=DIM, lw=1, ls="--"))
for k in range(6):
    a = math.radians(60 * k)
    ax.add_patch(Circle((18 * math.cos(a), 18 * math.sin(a)), 2.5, fc="#f4c48e", ec=ORG, lw=1.2))
    ax.text(36 * math.cos(a), 36 * math.sin(a), f"{60 * k}°", color=DIM, ha="center", va="center", fontsize=11)
ax.plot([0, 33], [0, 0], color=RED, lw=1.4)
label(ax, 0, -38, "LK Ø36 · 6× M6 · 60° gleich geteilt · 0° = C0°", AX, 10)
ax.set_xlim(-42, 42); ax.set_ylim(-42, 40)
save(f, "d_bohrbild.png")
print("Grafiken:", sorted(n for n in os.listdir(OUT) if n.startswith("d_")))
