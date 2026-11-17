# -*- coding: utf-8 -*-
"""
3D Viewer – Viewer und Messwerkzeug für .s3d-3D-Dateien (XML, vortesselliert)

Bedienung:
  Klick              Fläche oder Kante wählen (Kanten haben Vorrang, wenn nah)
  Strg + Klick       zur Auswahl hinzufügen / entfernen
  Shift + Klick      Punkt auf der Oberfläche hinzufügen
  Doppelklick        Drehpunkt auf angeklickte Stelle setzen
  2 Elemente gewählt -> Messung (Abstand, Achsabstand, Winkel, Ø)
  Nullpunkt-Tab      Werkstück-Nullpunkt aus Auswahl (z. B. Mitte Lochkreis)
  Bohrungen-Tab      Bohrungstabelle im Werkstück-Koordinatensystem, CSV-Export
  Strg + Mausrad     Schriftgröße (außerhalb der 3D-Ansicht)
"""

import math
import os
import sys
import traceback

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0] or __file__)), "S3D_Viewer_fehler.log")


def _msgbox(text, title="3D Viewer", flags=0x10):
    try:
        import ctypes
        return ctypes.windll.user32.MessageBoxW(0, text, title, flags)
    except Exception:
        print(text, file=sys.stderr)
        return 0


def _log(text):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception:
        pass


def _check_dependencies():
    needed = {"numpy": "numpy", "PyQt6": "PyQt6", "pyqtgraph": "pyqtgraph", "OpenGL": "PyOpenGL"}
    missing = []
    for mod, pkg in needed.items():
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if not missing:
        return
    _log("Fehlende Pakete: " + ", ".join(missing))
    ans = _msgbox("Folgende Python-Pakete fehlen:\n\n  " + "\n  ".join(missing)
                  + "\n\nJetzt automatisch installieren (pip)?", "3D Viewer – Pakete fehlen", 0x4 | 0x30)
    if ans != 6:
        sys.exit(1)
    import subprocess
    py = sys.executable
    if py.lower().endswith("pythonw.exe"):
        py = py[:-len("pythonw.exe")] + "python.exe"
    r = subprocess.run([py, "-m", "pip", "install", *missing])
    if r.returncode != 0:
        _msgbox("Installation fehlgeschlagen.\nBitte manuell ausführen:\n\n"
                f'"{py}" -m pip install ' + " ".join(missing))
        sys.exit(1)
    _msgbox("Pakete installiert. Der Viewer startet jetzt.", flags=0x40)


def _excepthook(etype, value, tb):
    text = "".join(traceback.format_exception(etype, value, tb))
    _log(text)
    _msgbox("Unerwarteter Fehler:\n\n" + text[-1500:] + f"\n\nDetails: {LOG_FILE}")


sys.excepthook = _excepthook
_check_dependencies()

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders as _sh
from OpenGL import GL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3d_core as core
from s3d_core import fmt

APP_NAME = "3D Viewer"
APP_URL = "https://github.com/Rabenstaub"
APP_VERSION = "3.3.0"

# Beidseitige Beleuchtung: Kopflicht + Führungslicht + leichter Glanz
_sh.Shaders.append(_sh.ShaderProgram("s3d_shaded", [
    _sh.VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    _sh.FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            vec3 n = normalize(v_normal);
            float head = abs(n.z);
            float key  = abs(dot(n, normalize(vec3(0.45, 0.55, 0.70))));
            float l = 0.25 + 0.50 * head + 0.30 * key;
            float spec = pow(head, 40.0) * 0.22;
            gl_FragColor = vec4(min(v_color.rgb * l + vec3(spec), 1.0), v_color.a);
        }
    """),
]))

_CUT_VS = """
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        varying vec3 v_pos;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            v_pos = a_position.xyz;
            gl_Position = u_mvp * a_position;
        }
"""
# u_clip = [nx, ny, nz, d, aktiv]  -> verworfen wird alles mit dot(p, n) > d
_sh.Shaders.append(_sh.ShaderProgram("s3d_cut", [
    _sh.VertexShader(_CUT_VS),
    _sh.FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform float u_clip[5];
        varying vec4 v_color;
        varying vec3 v_normal;
        varying vec3 v_pos;
        void main() {
            bool cut = u_clip[4] > 0.5;
            if (cut && dot(v_pos, vec3(u_clip[0], u_clip[1], u_clip[2])) > u_clip[3]) discard;
            vec3 n = normalize(v_normal);
            if (cut && n.z < 0.0) {
                // Innenseite sichtbar -> Schnittfläche (schraffiert)
                float h = mod(gl_FragCoord.x + gl_FragCoord.y, 9.0);
                vec3 c = h < 1.6 ? vec3(0.45, 0.10, 0.10) : vec3(0.86, 0.36, 0.33);
                gl_FragColor = vec4(c, 1.0);
                return;
            }
            float head = abs(n.z);
            float key  = abs(dot(n, normalize(vec3(0.45, 0.55, 0.70))));
            float l = 0.25 + 0.50 * head + 0.30 * key;
            float spec = pow(head, 40.0) * 0.22;
            gl_FragColor = vec4(min(v_color.rgb * l + vec3(spec), 1.0), v_color.a);
        }
    """),
], uniforms={"u_clip": [0.0, 0.0, 1.0, 0.0, 0.0]}))

_sh.Shaders.append(_sh.ShaderProgram("s3d_glasscut", [
    _sh.VertexShader(_CUT_VS),
    _sh.FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform float u_clip[5];
        varying vec4 v_color;
        varying vec3 v_normal;
        varying vec3 v_pos;
        void main() {
            if (u_clip[4] > 0.5 && dot(v_pos, vec3(u_clip[0], u_clip[1], u_clip[2])) > u_clip[3]) discard;
            vec3 n = normalize(v_normal);
            float head = abs(n.z);
            float key  = abs(dot(n, normalize(vec3(0.45, 0.55, 0.70))));
            float l = 0.25 + 0.50 * head + 0.30 * key;
            gl_FragColor = vec4(min(v_color.rgb * l, 1.0), v_color.a);
        }
    """),
], uniforms={"u_clip": [0.0, 0.0, 1.0, 0.0, 0.0]}))


def set_clip_uniform(vals):
    for nm in ("s3d_cut", "s3d_glasscut"):
        _sh.ShaderProgram.names[nm]["u_clip"] = list(vals)


GL_EDGES = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, "glColorMask": (GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE),
            "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)}
GL_ONTOP = {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, "glColorMask": (GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE),
            "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)}
# nur Tiefe schreiben (unsichtbare Flächen verdecken die Linien dahinter)
GL_DEPTHONLY = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: False, GL.GL_CULL_FACE: False, "glDepthMask": (GL.GL_TRUE,),
                "glColorMask": (GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE),
                GL.GL_POLYGON_OFFSET_FILL: True, "glPolygonOffset": (2.0, 6.0)}

GL_GLASS = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_CULL_FACE: False, "glColorMask": (GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE),
            "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA), "glDepthMask": (GL.GL_FALSE,)}
GL_SOLID = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: False, GL.GL_CULL_FACE: False, "glDepthMask": (GL.GL_TRUE,), "glColorMask": (GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE),
            GL.GL_POLYGON_OFFSET_FILL: True, "glPolygonOffset": (1.0, 1.0)}
GL_HOVER = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_CULL_FACE: False,
            "glColorMask": (GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE),
            "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA), "glDepthMask": (GL.GL_FALSE,),
            GL.GL_POLYGON_OFFSET_FILL: True, "glPolygonOffset": (-1.0, -2.0)}
GLASS_PALETTE = [(0.90, 0.08, 0.08), (0.10, 0.35, 0.95), (0.05, 0.65, 0.15), (0.80, 0.10, 0.80),
                 (0.95, 0.55, 0.00), (0.00, 0.65, 0.70), (0.55, 0.30, 0.10), (0.35, 0.35, 0.35)]
SEL_COLOR = np.array([0.90, 0.08, 0.08, 1.0], np.float32)   # rot
MEAS_COLOR = (0.05, 0.25, 0.85, 1.0)
QUALITY = {"Grob (Original)": None, "Mittel": 2.5e-4, "Fein": 4e-5, "Sehr fein": 1.2e-5}

THEME_CLASSIC = """
QWidget { background:#d9d4cc; color:#2b2724; font-size:%(fs).1fpt; }
QToolBar { background:#cfc9bf; border:none; spacing:4px; padding:3px; }
QToolButton, QPushButton { background:#e6e1d9; border:1px solid #b5ada1; border-radius:4px; padding:3px 6px; }
QToolButton:hover, QPushButton:hover { background:#f0ece6; }
QToolButton:checked { background:#bfb6a8; }
QComboBox, QDoubleSpinBox { background:#f3f0eb; border:1px solid #b5ada1; padding:2px 4px; }
QTabWidget::pane { border:1px solid #b5ada1; background:#e6e1d9; }
QTabBar::tab { background:#cfc9bf; padding:5px 9px; border:1px solid #b5ada1; border-bottom:none; }
QTabBar::tab:selected { background:#e6e1d9; }
QTableWidget, QTreeWidget, QLineEdit, QTextBrowser { background:#f3f0eb; border:1px solid #b5ada1;
    alternate-background-color:#ebe7e0; selection-background-color:#b8a98f; selection-color:#1d1a17; }
QHeaderView::section { background:#cfc9bf; border:1px solid #b5ada1; padding:3px; }
QStatusBar { background:#cfc9bf; }
QGroupBox { border:1px solid #b5ada1; margin-top:10px; padding-top:6px; }
QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 3px; }
QToolButton#aboutButton { border-radius:12px; padding:0; color:#6e665d; font-style:italic; font-weight:bold; }
"""


# Modernes Glas-Design (hell, Milchglas-Flächen, blaue Akzente)
THEME_GLASS = """
QMainWindow, QDialog { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #f3f7fd, stop:0.55 #e7eef8, stop:1 #dbe5f3); }
QWidget { color:#24304a; font-family:"Segoe UI Variable Text","Segoe UI",sans-serif; font-size:%(fs).1fpt; }
QToolBar { background: rgba(255,255,255,0.62); border:1px solid #d9e3f1; border-radius:14px;
    margin:4px 8px 2px 8px; padding:5px 6px; spacing:6px; }
QToolBar::separator { width:1px; background:#d6e0ee; margin:6px 4px; }
QToolBar QLabel { background:transparent; color:#6b7890; padding:0 2px; }
QToolButton, QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f5fb);
    border:1px solid #d3deee; border-radius:10px; padding:5px 12px; color:#24304a; }
QToolButton:hover, QPushButton:hover { background:#ffffff; border:1px solid #8fb6ff; color:#1d4fb8; }
QToolButton:pressed, QPushButton:pressed { background:#e3edff; border:1px solid #6aa0ff; }
QToolButton:checked { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7db3ff, stop:1 #3c7cf0);
    color:#ffffff; border:1px solid #3c7cf0; font-weight:600; }
QToolButton:checked:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8fbeff, stop:1 #4a88f5);
    color:#ffffff; }
QPushButton:default { border:1px solid #6aa0ff; }
QPushButton:disabled, QToolButton:disabled { color:#a6b1c2; background:#f3f6fa; border:1px solid #e1e8f2; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QKeySequenceEdit {
    background: rgba(255,255,255,0.92); border:1px solid #d3deee; border-radius:9px; padding:4px 9px;
    selection-background-color:#3c7cf0; selection-color:#ffffff; }
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover { border:1px solid #9cbff7; }
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border:1px solid #3c7cf0; }
QComboBox::drop-down { border:none; width:22px; subcontrol-origin:padding; subcontrol-position:center right; }
QComboBox::down-arrow { image:url(%(arrow_down)s); width:10px; height:10px; margin-right:8px; }
QAbstractSpinBox { padding-right:22px; }
QAbstractSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:20px;
    border:none; background:transparent; margin:3px 4px 0 0; }
QAbstractSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:20px;
    border:none; background:transparent; margin:0 4px 3px 0; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background:#e3edff; border-radius:5px; }
QAbstractSpinBox::up-arrow { image:url(%(arrow_up)s); width:9px; height:9px; }
QAbstractSpinBox::down-arrow { image:url(%(arrow_down)s); width:9px; height:9px; }
QComboBox QAbstractItemView { background:#ffffff; border:1px solid #d3deee; border-radius:8px; padding:4px;
    selection-background-color:#e3edff; selection-color:#1d3f8f; outline:0; }
QDockWidget { color:#6b7890; font-weight:600; }
QDockWidget::title { background:transparent; padding:8px 12px 4px 14px; text-align:left; }
QDockWidget > QWidget { background:transparent; }
QTabWidget::pane { background: rgba(255,255,255,0.58); border:1px solid #d9e3f1; border-radius:14px;
    top:-1px; padding:6px; }
QTabBar { background:transparent; }
QTabBar::tab { background:transparent; color:#6b7890; padding:6px 9px; margin:2px 1px 6px 1px;
    border:1px solid transparent; border-radius:10px; }
QTabBar::tab:hover { color:#1d4fb8; background: rgba(255,255,255,0.6); }
QTabBar::tab:selected { background:#ffffff; color:#1d4fb8; border:1px solid #d3deee; font-weight:600; }
QTableWidget, QTreeWidget, QListWidget, QTextBrowser {
    background: rgba(255,255,255,0.80); border:1px solid #dde6f2; border-radius:11px;
    gridline-color:#edf2f8; alternate-background-color:#f5f8fd;
    selection-background-color:#dce9ff; selection-color:#16306b; }
QTextBrowser { padding:6px; }
QTableWidget::item, QTreeWidget::item, QListWidget::item { padding:3px 4px; }
QHeaderView { background:transparent; border:none; }
QHeaderView::section { background:transparent; color:#6b7890; border:none; border-bottom:1px solid #dde6f2;
    padding:6px 8px; font-weight:600; }
QTableCornerButton::section { background:transparent; border:none; }
QScrollBar:vertical { background:transparent; width:10px; margin:3px 2px; }
QScrollBar:horizontal { background:transparent; height:10px; margin:2px 3px; }
QScrollBar::handle:vertical { background:#c9d6ea; border-radius:4px; min-height:28px; }
QScrollBar::handle:horizontal { background:#c9d6ea; border-radius:4px; min-width:28px; }
QScrollBar::handle:hover { background:#98b3dc; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:transparent; }
QStatusBar { background: rgba(255,255,255,0.66); border-top:1px solid #dde6f2; color:#4c5a73; padding:2px 8px; }
QStatusBar QLabel { background:transparent; }
QMenu { background:#ffffff; border:1px solid #d3deee; border-radius:12px; padding:6px; }
QMenu::item { padding:6px 26px 6px 14px; border-radius:7px; color:#24304a; }
QMenu::item:selected { background:#e3edff; color:#16306b; }
QMenu::item:disabled { color:#a6b1c2; }
QMenu::separator { height:1px; background:#e6edf6; margin:5px 10px; }
QMenu::indicator { width:14px; height:14px; margin-left:4px; }
QToolTip { background:#ffffff; color:#24304a; border:1px solid #d3deee; border-radius:8px; padding:5px 8px; }
QGroupBox { background: rgba(255,255,255,0.55); border:1px solid #dde6f2; border-radius:12px;
    margin-top:16px; padding:10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left:12px; padding:0 4px; color:#6b7890; font-weight:600; }
QLabel { background:transparent; }
QCheckBox { background:transparent; spacing:8px; }
QCheckBox::indicator { width:16px; height:16px; border-radius:5px; border:1px solid #b9c9e2; background:#ffffff; }
QCheckBox::indicator:checked { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7db3ff, stop:1 #3c7cf0);
    border:1px solid #3c7cf0; }
QMainWindow::separator { background:transparent; width:6px; height:6px; }
QMessageBox { background:#f3f7fd; }
QToolButton#aboutButton { background: rgba(255,255,255,0.5); color:#8a97ad; border:1px solid #d9e3f1;
    border-radius:12px; padding:0; font-family:Georgia,"Times New Roman",serif; font-style:italic;
    font-weight:600; font-size:11pt; }
QToolButton#aboutButton:hover { color:#1d4fb8; background:#ffffff; border:1px solid #8fb6ff; }
"""

THEME = THEME_GLASS

# Farbwerte je Design (Ansichtsfenster, Textfelder, Hervorhebung)
DESIGN_COLORS = {
    "Glas": {"bg_top": "#c3d3ec", "bg_bottom": "#f4f7fc", "text_bg": "#ffffff", "text_bg_alpha": 0.82,
             "text_color": "#1c2740", "hover_color": "#805b9bff", "contour_color": "#3a4458", "tip_radius": 12},
    "Klassisch": {"bg_top": "#8796aa", "bg_bottom": "#e6e6db", "text_bg": "#ffffff", "text_bg_alpha": 0.6,
                  "text_color": "#000000", "hover_color": "#80b76767", "contour_color": "#4c4c4c", "tip_radius": 10},
}


def html_table(rows, title=None):
    if THEME is THEME_GLASS:
        h = (f"<div style='color:#1d4fb8; font-size:large; font-weight:600; margin:2px 0 8px 0'>{title}</div>"
             if title else "")
        h += "<table cellspacing=0 cellpadding=4 width='100%'>"
        for i, (k, v) in enumerate(rows):
            bg = " style='background:#f3f7fd'" if i % 2 else ""
            h += (f"<tr{bg}><td style='color:#6b7890; padding-right:14px'>{k}</td>"
                  f"<td style='color:#1c2740'>{v}</td></tr>")
        return h + "</table>"
    h = f"<h3 style='margin:2px 0 6px 0'>{title}</h3>" if title else ""
    h += "<table cellspacing=3>"
    for k, v in rows:
        h += f"<tr><td style='padding-right:10px'><b>{k}</b></td><td>{v}</td></tr>"
    return h + "</table>"


def vtxt(p):
    return " / ".join(fmt(x) for x in p)


class InfoTable(QtWidgets.QTableWidget):
    def __init__(self, headers, parent=None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)

    def set_rows(self, rows):
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                it = QtWidgets.QTableWidgetItem(str(val))
                if len(str(val)) > 18:
                    it.setToolTip(str(val))
                self.setItem(r, c, it)
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)


# ---------------------------------------------------------------------------
# 3D-Ansicht
# ---------------------------------------------------------------------------

class Viewport(gl.GLViewWidget):
    clicked = QtCore.pyqtSignal(object, object)     # pos, modifiers
    dclicked = QtCore.pyqtSignal(object)
    viewChanged = QtCore.pyqtSignal()
    hovered = QtCore.pyqtSignal(object)
    viewMoving = QtCore.pyqtSignal()
    contextRequested = QtCore.pyqtSignal(object)
    sectionDrag = QtCore.pyqtSignal(str, float, float)   # "rot"/"move", dx, dy
    rectSelected = QtCore.pyqtSignal(object, object)      # QRect, modifiers

    FOV_ORTHO = 8.0
    FOV_PERSP = 40.0

    def __init__(self, parent=None):
        super().__init__(parent, rotationMethod="quaternion")
        self.opts["fov"] = self.FOV_ORTHO       # fast parallele Projektion (wie CAD)
        self.setMouseTracking(True)
        self.setBackgroundColor(QtGui.QColor("#e6e6db"))
        self._press = None
        self._press_btn = None
        self._last = None
        self.lock = None          # (Punkt, Richtung) der gesperrten Drehachse
        self.pan_ndc = [0.0, 0.0]  # Bildverschiebung (rechte Maustaste), Drehpunkt bleibt erhalten
        self.bg_top = QtGui.QColor("#8796aa")
        self.bg_bottom = QtGui.QColor("#e6e6db")
        self.callouts = []         # [(pos3d, text, bg QColor, fg QColor)]
        self.glass = True
        self.zoom_step = 0.10
        self.empty_logo = None
        self.show_empty = True
        self.text_scale = 1.0
        self.wheel_invert = False
        self.tip_radius = 10
        self._anim = QtCore.QTimer(self)
        self._anim.timeout.connect(self._anim_step)
        self._band = None
        self.rect_mode = False      # Rechteckselektion ohne Alt (aus dem Menü gestartet)

    # ---------- Projektion / Zeichnen
    def projectionMatrix(self, region, viewport):
        m = super().projectionMatrix(region, viewport)
        if self.pan_ndc[0] or self.pan_ndc[1]:
            t = QtGui.QMatrix4x4()
            t.translate(self.pan_ndc[0], self.pan_ndc[1], 0.0)
            m = t * m
        return m

    def paint(self, *, region, viewport, useItemNames=False):
        self.setProjection(region, viewport)
        self.setModelview()
        GL.glDepthMask(GL.GL_TRUE)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glClearColor(*self.bg_bottom.getRgbF())
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        # Hintergrund-Farbverlauf per QPainter
        try:
            p = QtGui.QPainter(self)
            g = QtGui.QLinearGradient(0, 0, 0, self.height())
            g.setColorAt(0.0, self.bg_top)
            g.setColorAt(1.0, self.bg_bottom)
            p.fillRect(self.rect(), g)
            p.end()
        except Exception:
            pass
        dpr = self.devicePixelRatioF()
        GL.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        self.drawItemTree(useItemNames=useItemNames)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        if self.callouts:
            self._draw_callouts()
        if getattr(self, "empty_logo", None) is not None and self.show_empty:
            self._draw_empty()

    def _draw_empty(self):
        """Startbild ohne geladenes Modell: Logo + Hinweis."""
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        s = int(min(self.width(), self.height()) * 0.28)
        s = max(96, min(s, 220))
        x = (self.width() - s) // 2
        y = int(self.height() * 0.5 - s * 0.8)
        p.setOpacity(0.9)
        p.drawPixmap(QtCore.QRect(x, y, s, s), self.empty_logo)
        p.setOpacity(1.0)
        f = QtGui.QFont("Segoe UI")
        f.setPixelSize(max(22, s // 5))
        f.setWeight(QtGui.QFont.Weight.ExtraBold)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        w1, w2 = fm.horizontalAdvance("3D "), fm.horizontalAdvance("VIEWER")
        tx = (self.width() - w1 - w2) / 2
        ty = y + s + fm.ascent() + 12
        p.setPen(QtGui.QColor("#1f6fe5")); p.drawText(QtCore.QPointF(tx, ty), "3D")
        p.setPen(QtGui.QColor("#2b3446")); p.drawText(QtCore.QPointF(tx + w1, ty), "VIEWER")
        f2 = QtGui.QFont("Segoe UI"); f2.setPixelSize(15); p.setFont(f2)
        p.setPen(QtGui.QColor("#5b6b86"))
        hint = "Datei öffnen (Strg+O) oder .s3d-Datei hierher ziehen  ·  Hilfe: F1"
        fm2 = QtGui.QFontMetrics(f2)
        p.drawText(QtCore.QPointF((self.width() - fm2.horizontalAdvance(hint)) / 2, ty + 34), hint)
        p.end()

    def _draw_callouts(self):
        pts = np.array([c[0] for c in self.callouts], float)
        S, ok = self.to_screen(pts)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        f = QtGui.QFont("Segoe UI")
        f.setPointSizeF(10.5 * self.text_scale)
        f.setBold(True)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        placed = []
        for (pos, text, bg, fg), s, k in zip(self.callouts, S, ok):
            if not k:
                continue
            lines = text.split("\n")
            padx, pady = (11, 7) if self.glass else (7, 4)
            w = max(fm.horizontalAdvance(x) for x in lines) + 2 * padx
            h = fm.height() * len(lines) + 2 * pady
            r = QtCore.QRectF(s[0] + 8, s[1] - h - 8, w, h)
            if r.right() > self.width() - 4:
                r.moveRight(s[0] - 8)
            if r.top() < 4:
                r.moveTop(s[1] + 8)
            # überlappende Felder nach oben versetzen, damit alles lesbar bleibt
            for _ in range(8):
                hit = next((q for q in placed if q.intersects(r.adjusted(-2, -2, 2, 2))), None)
                if hit is None:
                    break
                r.moveBottom(hit.top() - 3)
            placed.append(QtCore.QRectF(r))
            rad = self.tip_radius * 0.75
            if self.glass:
                # weicher Schatten + Milchglas-Rand
                for k_, a_ in ((3, 18), (2, 22), (1, 26)):
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QtGui.QColor(40, 70, 130, a_))
                    p.drawRoundedRect(r.adjusted(-k_, -k_ + 2, k_, k_ + 2), rad + k_, rad + k_)
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 235), 1.4))
                p.setBrush(bg)
                p.drawRoundedRect(r, rad, rad)
                p.setPen(QtGui.QPen(QtGui.QColor(120, 150, 200, 90), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(r.adjusted(-0.5, -0.5, 0.5, 0.5), rad, rad)
            else:
                p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60, 160), 1))
                p.setBrush(bg)
                p.drawRoundedRect(r, self.tip_radius * 0.6, self.tip_radius * 0.6)
            p.setPen(fg)
            for i, ln in enumerate(lines):
                p.drawText(QtCore.QPointF(r.left() + padx, r.top() + pady + fm.ascent() + i * fm.height()), ln)
        p.end()

    # ---------- Kamera-Helfer
    def reset_pan(self):
        self.pan_ndc = [0.0, 0.0]
        self.update()

    def set_view_angles(self, elev, azim):
        """Standardansicht (Elevation/Azimut wie Euler-Modus) als Quaternion setzen."""
        q = (QtGui.QQuaternion.fromAxisAndAngle(1, 0, 0, elev - 90)
             * QtGui.QQuaternion.fromAxisAndAngle(0, 0, -1, azim + 90))
        self.opts["rotation"] = q
        self.update()

    def set_view_dir(self, toward, up_hint):
        """Kamera so drehen, dass 'toward' (Welt) zum Betrachter zeigt."""
        z = toward / np.linalg.norm(toward)
        up = np.asarray(up_hint, float) - (np.asarray(up_hint, float) @ z) * z
        if np.linalg.norm(up) < 1e-6:
            up = np.array([0, 0, 1.0]) - z[2] * z
            if np.linalg.norm(up) < 1e-6:
                up = np.array([0, 1.0, 0]) - z[1] * z
        y = up / np.linalg.norm(up)
        x = np.cross(y, z)
        m = QtGui.QMatrix3x3(np.array([x, y, z], float).flatten().tolist())
        self.opts["rotation"] = QtGui.QQuaternion.fromRotationMatrix(m)
        self.update()

    def toward_camera(self):
        tc = self.opts["rotation"].conjugated().rotatedVector(QtGui.QVector3D(0, 0, 1))
        return np.array([tc.x(), tc.y(), tc.z()])

    def screen_axes(self):
        q = self.opts["rotation"].conjugated()
        r = q.rotatedVector(QtGui.QVector3D(1, 0, 0))
        u = q.rotatedVector(QtGui.QVector3D(0, 1, 0))
        return np.array([r.x(), r.y(), r.z()]), np.array([u.x(), u.y(), u.z()])

    def set_perspective(self, on):
        old = self.opts["fov"]
        new = self.FOV_PERSP if on else self.FOV_ORTHO
        k = math.tan(math.radians(old) / 2) / math.tan(math.radians(new) / 2)
        self.opts["distance"] *= k
        self.opts["fov"] = new
        self.update()

    def world_per_px(self):
        return 2 * self.opts["distance"] * math.tan(math.radians(self.opts["fov"]) / 2) / max(self.height(), 1)

    def center_on_lock(self):
        if self.lock is None:
            return
        p, a = self.lock
        c = self.opts["center"]
        c = np.array([c.x(), c.y(), c.z()])
        foot = p + ((c - p) @ a) * a
        self.opts["center"] = QtGui.QVector3D(*foot)

    def rotate_locked(self, ang):
        if self.lock is None:
            return
        self.opts["rotation"] = self.opts["rotation"] * QtGui.QQuaternion.fromAxisAndAngle(
            QtGui.QVector3D(*self.lock[1]), ang)
        self.update()

    def zoom(self, factor, at=None):
        """factor > 1 = verkleinern. at = Mausposition (Zoom zum Mauszeiger)."""
        self.opts["distance"] = max(self.opts["distance"] * factor, 1e-3)
        if at is not None:
            mx = 2.0 * at.x() / max(self.width(), 1) - 1.0
            my = 1.0 - 2.0 * at.y() / max(self.height(), 1)
            self.pan_ndc[0] = mx - (mx - self.pan_ndc[0]) / factor
            self.pan_ndc[1] = my - (my - self.pan_ndc[1]) / factor
        self.update()

    # ---------- Animation
    def start_animation(self):
        self._anim.start(16)

    def stop_animation(self):
        was = self._anim.isActive()
        self._anim.stop()
        if was:
            self.viewChanged.emit()
        return was

    def _anim_step(self):
        if self.lock is not None:
            self.rotate_locked(0.8)
        else:
            _, up = self.screen_axes()
            self.opts["rotation"] = self.opts["rotation"] * QtGui.QQuaternion.fromAxisAndAngle(
                QtGui.QVector3D(*up), 0.8)
            self.update()
        self.viewMoving.emit()

    # ---------- Ereignisse
    def wheelEvent(self, ev):
        hook = getattr(self, "wheel_hook", None)
        if hook is not None and hook(ev):
            ev.accept()
            return
        d = ev.angleDelta().y() or ev.angleDelta().x()
        if self.wheel_invert:
            d = -d
        step = self.zoom_step                         # Anteil je Raste (z. B. 0,10 = 10 %)
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = self.zoom_step / 4.0               # Umschalt: feiner zoomen
        factor = (1.0 - step) ** (d / 120.0)
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(factor, at=ev.position())       # Zoom zum Mauszeiger
        else:
            self.zoom(factor)                         # Zoom zur Mitte
        self.viewMoving.emit()
        QtCore.QTimer.singleShot(150, self.viewChanged.emit)
        ev.accept()

    def keyPressEvent(self, ev):
        hook = getattr(self, "key_hook", None)
        if hook is not None and hook(ev):
            ev.accept()
            return
        if ev.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            QtCore.QTimer.singleShot(50, self.viewChanged.emit)
        if self.lock is not None and ev.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right,
                                                  Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = 15 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 5
            self.rotate_locked(step if ev.key() in (Qt.Key.Key_Right, Qt.Key.Key_Up) else -step)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _locked_angle(self, diff):
        q = self.opts["rotation"]
        a = self.lock[1]
        toward = self.toward_camera()
        v = np.cross(a, toward)
        vv = q.rotatedVector(QtGui.QVector3D(*v))
        sx, sy = vv.x(), -vv.y()
        n = math.hypot(sx, sy)
        av = q.rotatedVector(QtGui.QVector3D(*a))
        ax_, ay_ = av.x(), -av.y()
        na = math.hypot(ax_, ay_)
        along_roll = (diff.x() * sx + diff.y() * sy) / n if n > 1e-6 else 0.0
        along_axis = (diff.x() * ax_ + diff.y() * ay_) / na if na > 1e-6 else 0.0
        if n < 1e-6:
            return 0.5 * diff.x() * (1 if a @ toward > 0 else -1)
        if abs(along_roll) >= abs(along_axis):
            return 0.5 * along_roll
        return 0.5 * along_axis

    def mousePressEvent(self, ev):
        self._press = ev.position()
        self._press_btn = ev.button()
        self._last = ev.position()
        self.stop_animation()
        if ev.button() == Qt.MouseButton.LeftButton and (
                ev.modifiers() & Qt.KeyboardModifier.AltModifier or self.rect_mode):
            self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self)
            self._band.setGeometry(QtCore.QRect(ev.position().toPoint(), QtCore.QSize()))
            self._band.show()
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.RightButton and not ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        ev.accept()

    def mouseMoveEvent(self, ev):
        b = ev.buttons()
        if b == Qt.MouseButton.NoButton:
            self.hovered.emit(ev.position())
            return
        p = ev.position()
        if self._last is None:
            self._last = p
        diff = p - self._last
        self._last = p
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if self._band is not None:
            self._band.setGeometry(QtCore.QRect(self._press.toPoint(), p.toPoint()).normalized())
            return
        if b & Qt.MouseButton.LeftButton:
            if shift:
                self.sectionDrag.emit("rot", diff.x(), diff.y())
                return
            if self.lock is not None:
                ang = self._locked_angle(diff)
                self.opts["rotation"] = self.opts["rotation"] * QtGui.QQuaternion.fromAxisAndAngle(
                    QtGui.QVector3D(*self.lock[1]), ang)
                self.update()
            else:
                k = 0.5 if ctrl else 0.4
                self.orbit(-diff.x() * k, diff.y() * k)
            self.viewMoving.emit()
            return
        if b & Qt.MouseButton.RightButton:
            if shift:
                self.sectionDrag.emit("move", diff.x(), diff.y())
                return
            # Bauteil klebt am Mauszeiger: reine Bildverschiebung in Bildschirmkoordinaten
            self.pan_ndc[0] += 2.0 * diff.x() / max(self.width(), 1)
            self.pan_ndc[1] -= 2.0 * diff.y() / max(self.height(), 1)
            self.update()
            self.viewMoving.emit()
            return
        if b & Qt.MouseButton.MiddleButton:
            # Zoom entlang der Vertikalen (Strg: aus der Mitte)
            f = 1.01 ** diff.y()
            self.zoom(f, at=None if ctrl else self._press)
            self.viewMoving.emit()
            return

    def mouseReleaseEvent(self, ev):
        moved = self._press is not None and (ev.position() - self._press).manhattanLength() >= 4
        btn = ev.button()
        self.unsetCursor()
        if self._band is not None:
            r = self._band.geometry()
            self._band.hide()
            self._band.deleteLater()
            self._band = None
            self.rect_mode = False
            self._press = None
            self._last = None
            if moved:
                self.rectSelected.emit(r, ev.modifiers())
            ev.accept()
            return
        if btn == Qt.MouseButton.LeftButton and not moved:
            self.clicked.emit(ev.position(), ev.modifiers())
        elif btn == Qt.MouseButton.RightButton and not moved:
            self.contextRequested.emit(ev.position())
        elif moved:
            self.viewChanged.emit()
        self._press = None
        self._last = None
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.dclicked.emit(ev.position())

    def mvp(self):
        w, h = self.width(), self.height()
        vp = (0, 0, w, h)
        m = self.projectionMatrix(vp, vp) * self.viewMatrix()
        return np.array(m.data(), dtype=float).reshape(4, 4).T

    def ray(self, pos):
        M = self.mvp()
        try:
            inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return None, None
        x = 2.0 * pos.x() / self.width() - 1.0
        y = 1.0 - 2.0 * pos.y() / self.height()
        a = inv @ np.array([x, y, -1.0, 1.0])
        b = inv @ np.array([x, y, 1.0, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        return a, d / np.linalg.norm(d)

    def to_screen(self, P):
        M = self.mvp()
        H = np.c_[P, np.ones(len(P))] @ M.T
        w = H[:, 3:4]
        ndc = H[:, :3] / np.where(np.abs(w) < 1e-12, 1e-12, w)
        sx = (ndc[:, 0] + 1) * 0.5 * self.width()
        sy = (1 - ndc[:, 1]) * 0.5 * self.height()
        return np.c_[sx, sy], w[:, 0] > 0


# ---------------------------------------------------------------------------
# Dialog: Nullpunkt verschieben
# ---------------------------------------------------------------------------

class ShiftDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nullpunkt verschieben (im Werkstück-KS)")
        lay = QtWidgets.QFormLayout(self)
        self.sp = []
        for ax in "XYZ":
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-100000, 100000)
            s.setDecimals(4)
            s.setSuffix(" mm")
            lay.addRow(f"Δ{ax}", s)
            self.sp.append(s)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                        | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def values(self):
        return [s.value() for s in self.sp]


# ---------------------------------------------------------------------------
# Dialog: Eigenschaften (Farben, Linien, Steuerung, Tastenbelegung)
# ---------------------------------------------------------------------------

class ColorButton(QtWidgets.QPushButton):
    def __init__(self, color, alpha=False, parent=None):
        super().__init__(parent)
        self.alpha = alpha
        self.set_color(color)
        self.clicked.connect(self.pick)

    def set_color(self, c):
        self.color = QtGui.QColor(c)
        name = self.color.name(QtGui.QColor.NameFormat.HexArgb if self.alpha else QtGui.QColor.NameFormat.HexRgb)
        light = self.color.lightness() > 140 or self.color.alpha() < 120
        self.setText(name)
        self.setStyleSheet(f"background:{self.color.name()}; color:{'#000' if light else '#fff'};"
                           "border:1px solid #b9c9e2; border-radius:8px; padding:4px 12px;")

    def pick(self):
        opts = QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel if self.alpha else \
            QtWidgets.QColorDialog.ColorDialogOption(0)
        c = QtWidgets.QColorDialog.getColor(self.color, self, "Farbe wählen", opts)
        if c.isValid():
            self.set_color(c)

    def value(self):
        return self.color.name(QtGui.QColor.NameFormat.HexArgb if self.alpha else QtGui.QColor.NameFormat.HexRgb)


def help_dir():
    """Ordner mit Handbuch und Bildern (neben dem Programm bzw. in der EXE)."""
    base = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, "frozen", False) else __file__))
    for d in (os.path.join(getattr(sys, "_MEIPASS", base), "hilfe"), os.path.join(base, "hilfe")):
        if os.path.isfile(os.path.join(d, "handbuch.html")):
            return d
    return None


class ManualDialog(QtWidgets.QWidget):
    """Ausführliches Handbuch mit Bildern: Inhaltsverzeichnis links, Text rechts, Suche.
    Eigenes, freies Fenster (nicht modal): kann neben dem Viewer offen bleiben, auf einen zweiten
    Bildschirm gezogen und mit der Pinnadel immer im Vordergrund gehalten werden."""

    def __init__(self, win):
        super().__init__(None, Qt.WindowType.Window)
        self.win = win
        self.setWindowTitle(f"{APP_NAME} – Handbuch")
        self.setWindowIcon(win.windowIcon())
        self.resize(1180, 820)
        g = win.settings.value("manual_geometry")
        if g is not None:
            try:
                self.restoreGeometry(g)
            except Exception:
                pass
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.pin = QtWidgets.QPushButton("📌 Immer im Vordergrund")
        self.pin.setCheckable(True)
        self.pin.setToolTip("Handbuch bleibt über dem Viewer, während du dort ausprobierst")
        self.pin.toggled.connect(self._set_pin)
        top.addWidget(self.pin)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Im Handbuch suchen … (Enter = weiter)")
        self.search.returnPressed.connect(self._find)
        top.addWidget(self.search, 1)
        b = QtWidgets.QPushButton("Im Browser öffnen / drucken")
        b.clicked.connect(self._open_browser)
        top.addWidget(b)
        lay.addLayout(top)
        split = QtWidgets.QSplitter()
        self.toc = QtWidgets.QListWidget()
        self.toc.setMaximumWidth(320)
        self.toc.setWordWrap(True)
        self.toc.itemClicked.connect(lambda it: self.tb.scrollToAnchor(it.data(Qt.ItemDataRole.UserRole)))
        self.tb = QtWidgets.QTextBrowser()
        self.tb.setOpenExternalLinks(True)
        split.addWidget(self.toc)
        split.addWidget(self.tb)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        self.dir = help_dir()
        self.html = ""
        if self.dir is None:
            self.tb.setHtml("<p>Handbuch nicht gefunden – der Ordner <b>hilfe</b> fehlt neben dem Programm.</p>"
                            + win.help_html())
            return
        html = open(os.path.join(self.dir, "handbuch.html"), encoding="utf-8").read()
        html = html.replace("<!--TASTEN-->", win.keys_table_html())
        html = html.replace("3D Viewer – Handbuch</h1>", f"3D Viewer – Handbuch <span style='color:#6b7890; "
                            f"font-size:11pt; font-weight:400'>Version {APP_VERSION}</span></h1>", 1)
        self.html = html
        self.tb.setSearchPaths([self.dir])
        self.tb.document().setMetaInformation(QtGui.QTextDocument.MetaInformation.DocumentUrl,
                                              QtCore.QUrl.fromLocalFile(self.dir + os.sep).toString())
        self.tb.setHtml(html)
        import re
        for anchor, title in re.findall(r'<a name="([^"]+)"></a>\s*<h2>([^<]+)</h2>', html):
            it = QtWidgets.QListWidgetItem(title)
            it.setData(Qt.ItemDataRole.UserRole, anchor)
            self.toc.addItem(it)

    def _set_pin(self, on):
        geo = self.geometry()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(on))
        self.setGeometry(geo)
        self.show()
        self.win.settings.setValue("manual_pin", bool(on))

    def closeEvent(self, ev):
        self.win.settings.setValue("manual_geometry", self.saveGeometry())
        super().closeEvent(ev)

    def _find(self):
        t = self.search.text().strip()
        if t and not self.tb.find(t):
            self.tb.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
            self.tb.find(t)

    def _open_browser(self):
        if not self.html:
            return
        import tempfile
        fn = os.path.join(tempfile.gettempdir(), "3D_Viewer_Handbuch.html")
        base = QtCore.QUrl.fromLocalFile(self.dir + os.sep).toString()
        with open(fn, "w", encoding="utf-8") as fh:
            fh.write(self.html.replace("<head>", f'<head><base href="{base}">', 1))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fn))


class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, win, start_tab=None):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Eigenschaften / Hilfe")
        self.resize(780, 640)
        o = dict(win.opt)
        lay = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        lay.addWidget(tabs)
        self.w = {}
        hw = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(hw)
        hl.setContentsMargins(0, 0, 0, 0)
        mb = QtWidgets.QPushButton("📖  Ausführliches Handbuch mit Bildern öffnen (F1)")
        mb.clicked.connect(win.open_manual)
        # F1 (bzw. die eingestellte Taste) wirkt auch, solange dieses Fenster offen ist
        ka = win.actions_reg.get("help", (None,))[0]
        seq = ka.shortcut() if ka is not None and not ka.shortcut().isEmpty() else QtGui.QKeySequence("F1")
        self._help_sc = QtGui.QShortcut(seq, self)
        self._help_sc.activated.connect(win.open_manual)
        hl.addWidget(mb)
        hb = QtWidgets.QTextBrowser()
        hb.setOpenExternalLinks(True)
        hb.setHtml(win.help_html())
        hl.addWidget(hb, 1)
        tabs.addTab(hw, "Hilfe")

        def form_tab(title):
            w = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(w)
            tabs.addTab(w, title)
            return f

        def color(f, key, label, alpha=False):
            b = ColorButton(o[key], alpha)
            f.addRow(label, b)
            self.w[key] = b

        def dspin(f, key, label, lo, hi, step, dec=1):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setValue(float(o[key]))
            f.addRow(label, s)
            self.w[key] = s

        def check(f, key, label):
            c = QtWidgets.QCheckBox()
            c.setChecked(bool(o[key]))
            f.addRow(label, c)
            self.w[key] = c

        f = form_tab("Allgemein")
        cb = QtWidgets.QComboBox()
        cb.addItems(["Glas", "Klassisch"])
        cb.setCurrentText(o.get("design", "Glas"))
        f.addRow("Design:", cb)
        self.w["design"] = cb

        def _design_changed(name):
            for k, val in DESIGN_COLORS.get(name, {}).items():
                wdg = self.w.get(k)
                if isinstance(wdg, ColorButton):
                    wdg.set_color(val)
                elif isinstance(wdg, QtWidgets.QSpinBox):
                    wdg.setValue(int(val))
                elif wdg is not None:
                    wdg.setValue(val)
        cb.currentTextChanged.connect(_design_changed)
        color(f, "bg_top", "Hintergrundfarbe (oben):")
        color(f, "bg_bottom", "Farbverlauf (unten):")
        ui = QtWidgets.QComboBox()
        for pct in MainWindow.UI_STEPS:
            ui.addItem(f"{pct} %", pct)
        ui.setCurrentIndex(max(0, ui.findData(int(win.ui_scale))))
        f.addRow("Größe der Bedienoberfläche:", ui)
        self.ui_combo = ui
        check(f, "keys_on_buttons", "Tastenkürzel auf den Knöpfen anzeigen:")
        dspin(f, "text_scale", "Skalierung der Schrift im 3D-Fenster:", 0.5, 3.0, 0.1)
        color(f, "text_color", "Textfarbe:")
        color(f, "text_bg", "Hintergrundfarbe Text:")
        dspin(f, "text_bg_alpha", "Deckkraft Text-Hintergrund (0–1):", 0.0, 1.0, 0.1)
        s = QtWidgets.QSpinBox()
        s.setRange(0, 30)
        s.setValue(int(o["tip_radius"]))
        f.addRow("Eckenradius:", s)
        self.w["tip_radius"] = s

        f = form_tab("Modell")
        dspin(f, "line_width", "Standardlinienbreite:", 0.2, 5.0, 0.1)
        dspin(f, "hl_width", "Linienbreite hervorgehobener Objekte:", 0.5, 8.0, 0.1)
        color(f, "contour_color", "Farbe der Konturen:")
        check(f, "hover_on", "Dynamisches Hervorheben der Flächen:")
        color(f, "hover_color", "Dynamisches Hervorheben:", alpha=True)
        color(f, "hover_sel_color", "Dynamisches Hervorheben der Auswahl:", alpha=True)
        color(f, "sel_color", "Auswahl:")
        color(f, "dim_color", "Bemaßungen:")
        color(f, "thread_color", "Gewindebereich:")

        f = form_tab("Steuerung")
        info = QtWidgets.QLabel("Alle Maus- und Tastenbefehle stehen im Reiter <b>Hilfe</b>, das ausführliche Handbuch mit Bildern öffnet die Taste F1.")
        info.setWordWrap(True)
        f.addRow(info)
        check(f, "wheel_invert", "Mausrad-Richtung umkehren:")
        dspin(f, "zoom_step", "Zoom je Mausrad-Raste (%):", 2.0, 30.0, 1.0, 0)

        # Tastenbelegung
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("Taste anklicken und neue Kombination drücken. "
                                     "Entf/Rücktaste im Feld = keine Taste."))
        self.key_table = QtWidgets.QTableWidget(0, 2)
        self.key_table.setHorizontalHeaderLabels(["Funktion", "Taste"])
        self.key_table.verticalHeader().setVisible(False)
        self.key_table.horizontalHeader().setStretchLastSection(True)
        self.key_edits = {}
        for aid, (a, text, default, _) in win.actions_reg.items():
            r = self.key_table.rowCount()
            self.key_table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(text)
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.key_table.setItem(r, 0, it)
            ed = QtWidgets.QKeySequenceEdit(a.shortcut())
            try:
                ed.setMaximumSequenceLength(1)
            except Exception:
                pass
            self.key_table.setCellWidget(r, 1, ed)
            self.key_edits[aid] = ed
        self.key_table.resizeColumnToContents(0)
        v.addWidget(self.key_table)
        bt = QtWidgets.QPushButton("Tasten auf Standard zurücksetzen")
        bt.clicked.connect(self._reset_keys)
        v.addWidget(bt)
        tabs.addTab(w, "Tastenbelegung")

        bb = QtWidgets.QDialogButtonBox()
        bb.addButton("OK", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton("Abbrechen", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        rst = bb.addButton("Auf Standardwerte zurücksetzen", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        rst.clicked.connect(self._reset_all)
        bb.accepted.connect(self._check_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _reset_keys(self):
        for aid, ed in self.key_edits.items():
            d = self.win.actions_reg[aid][2]
            ed.setKeySequence(QtGui.QKeySequence(d) if d else QtGui.QKeySequence())

    def _reset_all(self):
        d = MainWindow.OPT_DEFAULTS
        for k, wdg in self.w.items():
            if isinstance(wdg, ColorButton):
                wdg.set_color(d[k])
            elif isinstance(wdg, QtWidgets.QCheckBox):
                wdg.setChecked(bool(d[k]))
            elif isinstance(wdg, QtWidgets.QComboBox):
                wdg.setCurrentText(str(d[k]))
            else:
                wdg.setValue(d[k])
        self._reset_keys()

    def _check_accept(self):
        seen = {}
        for aid, ed in self.key_edits.items():
            k = ed.keySequence().toString()
            if not k:
                continue
            if k in seen:
                QtWidgets.QMessageBox.warning(
                    self, "Tastenbelegung", f"Die Taste „{k}“ ist doppelt belegt:\n"
                    f"{self.win.actions_reg[seen[k]][1]}  und  {self.win.actions_reg[aid][1]}")
                return
            seen[k] = aid
        self.accept()

    def values(self):
        o = dict(self.win.opt)
        for k, wdg in self.w.items():
            if isinstance(wdg, ColorButton):
                o[k] = wdg.value()
            elif isinstance(wdg, QtWidgets.QCheckBox):
                o[k] = wdg.isChecked()
            elif isinstance(wdg, QtWidgets.QComboBox):
                o[k] = wdg.currentText()
            else:
                o[k] = wdg.value()
        return o

    def keymap(self):
        return {aid: ed.keySequence().toString() for aid, ed in self.key_edits.items()}


# ---------------------------------------------------------------------------
# Hauptfenster
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1450, 880)
        self.setAcceptDrops(True)
        self.settings = QtCore.QSettings("privat", "S3DViewer")
        # Größe der Bedienoberfläche in Prozent (alte Schriftgröße wird einmalig übernommen)
        if self.settings.value("ui_scale") is None and self.settings.value("font_pt") is not None:
            try:
                self.settings.setValue("ui_scale", int(round(int(self.settings.value("font_pt")) * 10 / 5)) * 5)
            except (TypeError, ValueError):
                pass
        try:
            self.ui_scale = int(self.settings.value("ui_scale", 100))
        except (TypeError, ValueError):
            self.ui_scale = 100
        self.ui_scale = max(70, min(200, self.ui_scale))
        self.font_pt = 10.0 * self.ui_scale / 100.0
        self.model = None
        self.mesh = None
        self.sel = []
        self.wcs = core.WCS()
        self.holes = []
        self.items_overlay = []

        self.opt = self._load_options()
        self.nav_mode = False          # Viewermodus "Navigieren": Klicks wählen nichts aus
        self.view_frame_name = self.settings.value("view_frame", "CLASSMATE")
        self.user_views = []
        self.rect_filter = dict(self.RECT_FILTER_DEFAULTS)
        self.annot_types = {"Gewinde": True, "Eintritt": True, "only_selected": False}
        self.view = Viewport()
        try:
            _ic = QtWidgets.QApplication.instance().windowIcon()
            if not _ic.isNull():
                self.view.empty_logo = _ic.pixmap(256, 256)
        except Exception:
            pass
        self.view.clicked.connect(self.on_click)
        self.view.dclicked.connect(self.on_dclick)
        self.view.contextRequested.connect(self.on_context_menu)
        self.view.sectionDrag.connect(self._on_section_drag)
        self.view.rectSelected.connect(self.on_rect_selected)
        self.view.viewChanged.connect(self._update_angle_overlay)
        self.view.hovered.connect(self._on_hover)
        self._sil_timer = QtCore.QTimer(self)
        self._sil_timer.setSingleShot(True)
        self._sil_timer.timeout.connect(self._update_silhouette)
        self.view.viewMoving.connect(self._sil_tick)
        self.view.viewChanged.connect(self._update_silhouette)
        self._hover_pos = None
        self._hover_key = None
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._do_hover)
        self.view.viewChanged.connect(lambda: self._build_pattern_items() if self.a_pattern.isChecked() else None)
        self.view.viewChanged.connect(lambda: self._build_box_items() if self.a_box.isChecked() else None)
        self.angle_items = []
        self.sel_callouts = []
        self.annot_callouts = []
        self.annot_items = []
        self.show_hidden = bool(int(self.settings.value("show_hidden", 0)))
        self.section = None        # None | {"kind", "n", "d"}
        self.section_flip = False
        self.section_items = []
        self.view.wheel_hook = self._wheel_hook
        self.view.key_hook = self._section_key
        self.setCentralWidget(self.view)
        self.lock_badge = QtWidgets.QLabel(self.view)
        self.lock_badge.move(10, 10)
        self.lock_badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._build_toolbar()
        self._build_docks()
        self._build_detail_dock()
        self._apply_options()
        self.status = self.statusBar()
        self.status.showMessage("Datei öffnen (Strg+O) oder .s3d hierher ziehen")
        self.apply_theme()
        self._update_lock_badge()
        QtWidgets.QApplication.instance().installEventFilter(self)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    # ---------------- UI ----------------
    def _build_toolbar(self):
        """Alle Funktionen zentral registriert – Tasten frei änderbar (Eigenschaften → Tasten)."""
        self.actions_reg = {}      # id -> (QAction, Grundtext, Standardtaste)
        try:
            import json
            keymap = json.loads(self.settings.value("keymap_json", "{}") or "{}")
        except Exception:
            keymap = {}

        self._act_default = {}

        def reg(aid, text, slot, key=None, checkable=False, checked=False, tip=None, tb=None, show_key=True):
            a = QtGui.QAction(text, self)
            k = keymap.get(aid, key) if key is not None or aid in keymap else None
            if k:
                a.setShortcut(QtGui.QKeySequence(k))
                a.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            a.setCheckable(checkable)
            a.setChecked(checked)
            a.triggered.connect(slot)
            if tip:
                a.setToolTip(tip)
            self.actions_reg[aid] = (a, text, key, show_key)
            if checkable:
                self._act_default[aid] = bool(checked)
            self._label_action(aid)
            if tb is not None:
                tb.addAction(a)
            else:
                self.addAction(a)
            return a

        self._reg = reg
        tb = self.addToolBar("Ansicht")
        tb.setMovable(False)
        reg("open", "Öffnen", self.open_dialog, "Ctrl+O", tb=tb)
        reg("refresh", "Aktualisieren", self.reload, "F5", tb=tb)
        reg("reset_all", "Zurücksetzen", self.reset_all, "Ctrl+R", tb=tb,
            tip="Alles auf Anfang: Ansicht, Schnitt, Darstellung, Nullpunkt, Auswahl, Fenster – "
                "wie frisch geladen (Farben und Tastenbelegung bleiben)")
        tb.addSeparator()
        reg("view_iso", "Iso", lambda: self.std_view(30, 45), "I", tb=tb)
        reg("view_front", "Vorne", lambda: self.std_view(0, -90), "F", tb=tb)
        reg("view_back", "Hinten", lambda: self.std_view(0, 90), "B", tb=tb)
        reg("view_top", "Oben", lambda: self.std_view(90, -90), "T", tb=tb)
        reg("view_bottom", "Unten", lambda: self.std_view(-90, -90), "D", tb=tb)
        reg("view_left", "Links", lambda: self.std_view(0, 180), "L", tb=tb)
        reg("view_right", "Rechts", lambda: self.std_view(0, 0), "R", tb=tb)
        reg("view_normal", "Normal auf", self.view_normal, "N", tb=tb,
            tip="Blick senkrecht auf die gewählte Fläche / entlang der gewählten Achse")
        reg("fit", "Fenstergröße", self.fit_view, "W", tb=tb)
        reg("home", "Anfang", self.home_view, "Space", tb=tb)
        # kleiner Info-Knopf ganz rechts (Urheber)
        sp = QtWidgets.QWidget()
        sp.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sp.setStyleSheet("background:transparent;")
        tb.addWidget(sp)
        self.btn_about = QtWidgets.QToolButton()
        self.btn_about.setObjectName("aboutButton")
        self.btn_about.setText("i")
        self.btn_about.setToolTip("Über 3D Viewer")
        self.btn_about.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_about.setFixedSize(24, 24)
        self.btn_about.clicked.connect(self.show_about)
        tb.addWidget(self.btn_about)

        # ---- zweite Zeile: Darstellung / Funktionen
        self.addToolBarBreak()
        tb = self.addToolBar("Funktionen")
        tb.setMovable(False)
        self.a_cut = reg("section", "Schnitt", self.cycle_section, "S", tb=tb,
                         tip="Schnittebene: S = Längsschnitt → Querschnitt → aus.\n"
                             "Umschalt+Mausrad oder Pfeiltasten: verschieben\nStrg+Umschalt+Mausrad oder Strg+Pfeiltasten: kippen\n"
                             "Umschalt+linke Maus: Ebene drehen, Umschalt+rechte Maus: verschieben\n"
                             "Umschalt+S: Seite wechseln")
        reg("section_flip", "Schnitt: Seite wechseln", self.flip_section, "Shift+S")
        self.a_lines = reg("lines", "Draht", self.on_lines, "Y", checkable=True, tb=tb,
                           tip="Drahtdarstellung: nur Kanten-/Messlinien (verdeckte ausgeblendet)")
        reg("hidden_lines", "Verdeckte Linien", self.toggle_hidden_lines, "Shift+Y")
        self.a_glass = reg("glass", "Transparent", self.on_glass, "G", checkable=True, tb=tb,
                           tip="Transparente Darstellung – Kanäle/Bohrungen sichtbar, Bohrung anklicken = farbig")
        self.a_wire = reg("tess", "Tesselierung", self.update_display, "Q", checkable=True, tb=tb,
                          tip="Tesselierung des Modells anzeigen")
        self.a_edges = reg("contours", "Konturen", self.update_display, "C", checkable=True, checked=True, tb=tb)
        self.a_thread_col = reg("thread_col", "Gewinde farbig", self.toggle_thread_color, "Ctrl+G", checkable=True,
                                checked=str(self.settings.value("thread_color_on", "true")).lower() in ("1", "true"),
                                tb=tb, tip="Gewindebereich in Gewindebohrungen farbig zeigen (Lage und Tiefe)")
        self.a_nocolor = reg("nocolor", "Farben ignorieren", self.on_nocolor, "H", checkable=True, tb=tb,
                             tip="Modellfarben ignorieren (einheitlich grau)")
        self.a_box = reg("box", "Box", self._build_box_items, "X", checkable=True, tb=tb,
                         tip="Zeichen-Box: Kasten um die Außenmaße mit Länge/Breite/Höhe")
        self.a_wcs = reg("wcs", "Koordinatensystem", self.update_display, "K", checkable=True, checked=True, tb=tb)
        self.a_annot = reg("annot", "Annotationen", self._build_annotations, "O", checkable=True,
                           checked=True, tb=tb, tip="Annotationen (Gewinde) anzeigen")
        self.a_pattern = reg("pattern", "Bohrbild", self._build_pattern_items, "Ctrl+B", checkable=True, tb=tb,
                             tip="Lochkreise und Winkel aller Bohrungen außerhalb des Bauteils")
        self.a_persp = reg("persp", "Perspektive", self.on_perspective, "P", checkable=True, tb=tb,
                           tip="Perspektivische Projektion")

        # ---- dritte Zeile: Einstellungen
        self.addToolBarBreak()
        tb = self.addToolBar("Einstellungen")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel(" Qualität: "))
        self.cb_quality = QtWidgets.QComboBox()
        self.cb_quality.addItems(list(QUALITY))
        self.cb_quality.setCurrentText(self.settings.value("quality", "Fein"))
        self.cb_quality.currentTextChanged.connect(self.on_quality)
        self.cb_quality.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tb.addWidget(self.cb_quality)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Sperrachse: "))
        self.cb_lock = QtWidgets.QComboBox()
        self.cb_lock.addItems(["Frei", "Längsachse (Teil)", "X (Nullpunkt)", "Y (Nullpunkt)",
                               "Z (Nullpunkt)", "Gewählte Achse"])
        self.cb_lock.setToolTip("Drehung auf eine Achse sperren – Teil lässt sich endlos um diese Achse drehen.\n"
                                "Gewählte Achse: vorher Bohrung/Zylinder, Kreiskante oder gerade Kante anklicken.")
        self.cb_lock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lock_choice = int(self.settings.value("lock_choice2", 1))
        if self.lock_choice not in (1, 2, 3, 4):
            self.lock_choice = 1
        self.cb_lock.setCurrentIndex(0)
        self.cb_lock.currentIndexChanged.connect(self.on_lock)
        tb.addWidget(self.cb_lock)
        self.a_lock = reg("zrot", "Z-Rotation", self.toggle_lock, "Z", checkable=True, tb=tb,
                          tip="Drehung auf die Längsachse sperren / wieder freigeben (endlos drehbar)")
        tb.addSeparator()
        self.a_dock_l = reg("dock_left", "Fenster links", lambda: self._toggle_dock(self.dock_left, self.a_dock_l),
                            "Ctrl+1", checkable=True, checked=True, tb=tb,
                            tip="Datenfenster links (Modell/Parameter/Features/Flächen) ein-/ausblenden")
        self.a_dock_r = reg("dock_right", "Fenster rechts", lambda: self._toggle_dock(self.dock_right, self.a_dock_r),
                            "Ctrl+2", checkable=True, checked=True, tb=tb,
                            tip="Analysefenster rechts (Auswahl/Messen, Nullpunkt, Bohrungen, Gewinde) ein-/ausblenden")
        tb.addSeparator()
        reg("props", "Eigenschaften / Hilfe", self.open_options, None, tb=tb,
            tip="Einstellungen, Farben, Tastenbelegung und Hilfe zu allen Maus- und Tastenbefehlen")
        reg("help", "Handbuch", self.open_manual, "F1")
        reg("ui_bigger", "Bedienoberfläche größer", lambda: self.step_ui_scale(+1), "Ctrl++")
        reg("ui_smaller", "Bedienoberfläche kleiner", lambda: self.step_ui_scale(-1), "Ctrl+-")
        reg("ui_reset", "Bedienoberfläche 100 %", lambda: self.set_ui_scale(100), "Ctrl+0")
        # ohne Knopf: Modellbaum, Modellinfo, Animation, Viewermodus, Esc
        self.a_tree = reg("tree", "Modellbaum anzeigen", self.toggle_tree, "M", checkable=True, checked=True)
        self.a_info = reg("info", "Modellinformationen anzeigen", self.show_info, "E")
        reg("anim_start", "Animation starten", self.view_anim_start, "A")
        reg("detail_views", "Detailansichten-Baum anzeigen", self.toggle_detail_views, "Ctrl+E")
        reg("rect_sel", "Rechteckselektion starten", self.start_rect_selection, None)
        reg("mosaic", "Mosaikansicht", self.mosaic_view, None)
        reg("escape", "Animation anhalten / Auswahl löschen", self.on_escape, "Esc")
        reg("del_fillet", "Radius/Fase löschen (scharfe Ecke)", self.delete_fillet, "Del")
        reg("restore_fillet", "Gelöschte Radien/Fasen wiederherstellen", self.restore_fillets, "Shift+Del")
        self.a_shaded = QtGui.QAction("Schattiert", self)
        self.a_shaded.setCheckable(True)
        self.a_shaded.setChecked(True)

    def _label_action(self, aid):
        a, text, default, show_key = self.actions_reg[aid]
        k = a.shortcut().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
        k = k.replace("Space", "Leertaste").replace("Ctrl", "Strg").replace("Shift", "Umschalt")
        on_btn = bool(getattr(self, "opt", {}).get("keys_on_buttons", False))
        a.setText(f"{text} ({k})" if (k and show_key and on_btn) else text)
        if not hasattr(a, "_base_tip"):
            t = a.toolTip()
            a._base_tip = t if (t and t != a.text() and t != text) else ""
        tip = a._base_tip or text
        a.setToolTip(f"{tip}   [{k}]" if k else tip)

    def apply_keymap(self, keymap):
        for aid, (a, text, default, show_key) in self.actions_reg.items():
            k = keymap.get(aid, default)
            a.setShortcut(QtGui.QKeySequence(k) if k else QtGui.QKeySequence())
            a.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self._label_action(aid)

    def _build_docks(self):
        self.tabs_left = QtWidgets.QTabWidget()
        self.model_table = InfoTable(["Eigenschaft", "Wert"])
        self.tabs_left.addTab(self.model_table, "Modell")
        pw = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pw)
        pl.setContentsMargins(2, 2, 2, 2)
        self.param_filter = QtWidgets.QLineEdit()
        self.param_filter.setPlaceholderText("Filter …")
        self.param_filter.textChanged.connect(self.filter_params)
        self.param_table = InfoTable(["Name", "Wert", "Typ"])
        pl.addWidget(self.param_filter)
        pl.addWidget(self.param_table)
        self.tabs_left.addTab(pw, "Parameter")
        self.feat_tree = QtWidgets.QTreeWidget()
        self.feat_tree.setHeaderLabels(["Feature", "Typ / Wert"])
        self.feat_tree.setAlternatingRowColors(True)
        self.feat_tree.itemSelectionChanged.connect(self.on_feature_row)
        self.tabs_left.addTab(self.feat_tree, "Features")
        self.face_table = InfoTable(["Fläche", "Typ", "Ø / R", "Länge", "Fläche mm²", "Feature"])
        self.face_table.itemSelectionChanged.connect(self.on_face_row)
        self.tabs_left.addTab(self.face_table, "Flächen")
        dl = QtWidgets.QDockWidget("Daten", self)
        self.dock_left = dl
        dl.setMinimumWidth(340)
        dl.setObjectName("dock_left")
        dl.setWidget(self.tabs_left)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dl)

        self.tabs_right = QtWidgets.QTabWidget()
        self.sel_info = QtWidgets.QTextBrowser()
        self.tabs_right.addTab(self.sel_info, "Auswahl / Messen")
        self.tabs_right.addTab(self._build_wcs_tab(), "Nullpunkt")
        hw = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(hw)
        hl.setContentsMargins(2, 2, 2, 2)
        self.hole_table = InfoTable(["Nr", "Ø", "Gewinde", "Gew.-Tiefe", "Kernloch-T.", "X", "Y", "Z", "I", "J", "K",
                                     "Tiefe", "Art", "Winkel°", "Abst. Mitte",
                                     "Masch. X Ø", "Masch. Y", "Masch. Z", "C°", "B°", "Tiefe bis X0", "Senkung/Stufe", "Spitze",
                                     "Tiefe voller Ø", "Feature"])
        self.hole_table.itemSelectionChanged.connect(self.on_hole_row)
        self.hole_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        bt = QtWidgets.QPushButton("Als CSV speichern …")
        bt.clicked.connect(self.export_holes)
        lab = QtWidgets.QLabel("X/Y/Z = Eintrittspunkt, I/J/K = Werkzeugachse (aus dem Material), "
                               "bezogen auf den aktuellen Nullpunkt. Mehrfachauswahl mit Strg.")
        lab.setWordWrap(True)
        hl.addWidget(lab)
        hl.addWidget(self.hole_table)
        hl.addWidget(bt)
        self.tabs_right.addTab(hw, "Bohrungen")
        self.thread_table = InfoTable(["Feature", "Gewinde", "Kern-Ø", "Tiefe", "Steigung"])
        self.tabs_right.addTab(self.thread_table, "Gewinde")
        self.tabs_right.addTab(self._build_lathe_tab(), "Drehteil")
        dr = QtWidgets.QDockWidget("Analyse", self)
        self.dock_right = dr
        dr.setObjectName("dock_right")
        dr.setWidget(self.tabs_right)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dr)
        for _dk in (self.dock_left, dr):
            _dk.visibilityChanged.connect(lambda _v: QtCore.QTimer.singleShot(0, self._sync_dock_actions))
        self.resizeDocks([dl, dr], [320, 470], Qt.Orientation.Horizontal)

    def _build_wcs_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        help_ = QtWidgets.QLabel(
            "<b>Nullpunkt aus Auswahl</b> – Beispiele:<br>"
            "• Teilmitte zu Außenbohrungen: alle Bohrungen mit <b>Strg+Klick</b> wählen "
            "(optional + Planfläche für Z) → Lochkreismitte<br>"
            "• Bohrung/Zylinder + Planfläche → Achse ∩ Fläche<br>"
            "• Kreiskante → Kreismittelpunkt<br>"
            "• Radiale Bohrungen → Schnittpunkt der Achsen<br>"
            "• Planfläche allein → nur Z auf diese Fläche<br>"
            "• Shift+Klick-Punkt → Nullpunkt auf Punkt<br><br>"
            "<b>C0° festlegen</b> (Z bleibt Längsachse): Bohrung anklicken → "
            "<b>C0° auf Auswahl legen</b>")
        help_.setWordWrap(True)
        help_.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(help_)
        grid = QtWidgets.QGridLayout()
        buttons = [
            ("Nullpunkt aus Auswahl", self.wcs_from_selection),
            ("Drehmaschine (Z = Länge)", self.wcs_lathe),
            ("Modell-Nullpunkt (CAD)", self.wcs_reset),
            ("Z umkehren", lambda: self._wcs_op(self.wcs.flip_z)),
            ("Um Z drehen +90°", lambda: self._wcs_op(lambda: self.wcs.rotate_z(90))),
            ("Um Z drehen −90°", lambda: self._wcs_op(lambda: self.wcs.rotate_z(-90))),
            ("C0° um 180° drehen", lambda: self._wcs_op(lambda: self.wcs.rotate_z(180))),
            ("C0° um Winkel drehen …", self.wcs_rotate_free),
            ("C0° auf Auswahl legen", self.wcs_align_x),
            ("Verschieben …", self.wcs_shift),
        ]
        for i, (t, f) in enumerate(buttons):
            b = QtWidgets.QPushButton(t)
            b.clicked.connect(f)
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        self.wcs_info = QtWidgets.QTextBrowser()
        lay.addWidget(self.wcs_info, 1)
        return w

    @staticmethod
    def _theme_icons():
        """Kleine Pfeil-Grafiken für Auswahlfelder (Stylesheets brauchen dafür Dateien).
        Als PNG gezeichnet – funktioniert auch in der EXE ohne SVG-Modul."""
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "viewer_3d_theme")
        out = {}
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return {"arrow_down": "", "arrow_up": ""}
        for name, pts in (("arrow_down", [(2.5, 4.5), (6, 8), (9.5, 4.5)]),
                          ("arrow_up", [(2.5, 7.5), (6, 4), (9.5, 7.5)])):
            fn = os.path.join(d, name + ".png")
            img = QtGui.QImage(48, 48, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.scale(4, 4)
            pen = QtGui.QPen(QtGui.QColor("#5b6b86"), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
            p.end()
            img.save(fn)
            out[name] = fn.replace("\\", "/")
        return out

    UI_STEPS = (70, 80, 90, 100, 110, 125, 150, 175, 200)

    def _resize_tables(self):
        for t in self.findChildren(QtWidgets.QTableWidget):
            t.resizeColumnsToContents()
            t.horizontalHeader().setStretchLastSection(True)

    def set_ui_scale(self, pct):
        self.ui_scale = int(max(70, min(200, pct)))
        self.font_pt = 10.0 * self.ui_scale / 100.0
        self.settings.setValue("ui_scale", self.ui_scale)
        self.apply_theme()
        self.status.showMessage(f"Bedienoberfläche: {self.ui_scale} %   (AltGr + Mausrad, Strg+Plus/Minus, Strg+0 = 100 %)")

    def step_ui_scale(self, direction):
        st = self.UI_STEPS
        cur = self.ui_scale
        if direction > 0:
            nxt = next((x for x in st if x > cur), st[-1])
        else:
            nxt = next((x for x in reversed(st) if x < cur), st[0])
        self.set_ui_scale(nxt)

    def apply_theme(self):
        global THEME
        glass = self.opt.get("design", "Glas") != "Klassisch"
        THEME = THEME_GLASS if glass else THEME_CLASSIC
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(THEME % {"fs": self.font_pt, **self._theme_icons()})
        self.setStyleSheet("")
        self.view.glass = glass
        k = self.ui_scale / 100.0
        if getattr(self, "dock_left", None) is not None:
            self.dock_left.setMinimumWidth(int(340 * k))
        if getattr(self, "btn_about", None) is not None:
            self.btn_about.setFixedSize(int(24 * k), int(24 * k))
        QtCore.QTimer.singleShot(0, self._resize_tables)
        self._update_lock_badge()

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.ShortcutOverride and isinstance(
                obj, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QKeySequenceEdit,
                      QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            if not ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                ev.accept()          # Tippen in Eingabefeldern löst keine Funktionstasten aus
                return False
        if ev.type() == QtCore.QEvent.Type.Wheel:
            mods = ev.modifiers()
            ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            alt = bool(mods & Qt.KeyboardModifier.AltModifier)
            if ctrl and alt:
                # AltGr (= Strg+Alt) + Mausrad: Bedienoberfläche größer/kleiner – überall im Fenster
                d = ev.angleDelta().y() or ev.angleDelta().x()
                if d:
                    now = QtCore.QTime.currentTime().msecsSinceStartOfDay()
                    if now - getattr(self, "_ui_wheel_t", -1000) > 120:     # ein Schritt je Raste
                        self._ui_wheel_t = now
                        self.step_ui_scale(1 if d > 0 else -1)
                return True
        return super().eventFilter(obj, ev)

    def closeEvent(self, ev):
        self.settings.setValue("geometry", self.saveGeometry())
        m = getattr(self, "_manual", None)
        if m is not None:
            m.close()
        super().closeEvent(ev)

    # ---------------- Datei ----------------
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for u in ev.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(".s3d"):
                self.load(p)
                break

    def open_dialog(self):
        start = self.settings.value("last_dir", "")
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "3D-Datei öffnen", start, "S3D-Dateien (*.s3d);;Alle Dateien (*)")
        if p:
            self.settings.setValue("last_dir", os.path.dirname(p))
            self.load(p)

    def load(self, path):
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            m = core.parse_s3d(path)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Datei konnte nicht gelesen werden:\n{e}")
            return
        self.model = m
        self.view.show_empty = False
        self.sel = []
        self.removed = {}
        self.wcs = core.WCS()
        self.wcs.set_lathe(m)
        self.section = None
        self.section_flip = False
        if self.cb_lock.currentIndex() == 5:
            self.cb_lock.blockSignals(True)
            self.cb_lock.setCurrentIndex(0)
            self.cb_lock.blockSignals(False)
            self.view.lock = None
        self.holes = core.find_holes(m)
        self._prepare_edges()
        self._build_scene()
        self._fill_tables()
        self._fill_holes()
        self._fill_lathe()
        self._build_annotations()
        self.user_views = []
        self._fill_view_list()
        self._update_wcs_view()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} – {os.path.basename(path)}")
        self._update_lock_badge()
        self.sel_info.setHtml(
            "<i>Klick: Fläche/Kante · Strg+Klick: mehrere · Shift+Klick: Punkt · "
            "2 Elemente = Messung · Doppelklick: Drehpunkt</i>")
        self.fit_view()

    def on_quality(self, txt):
        self.settings.setValue("quality", txt)
        if self.model:
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._build_scene()
            self._build_annotations()
            self._refresh_selection()
            self._show_contour_overlay()
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---------------- Szene ----------------
    def _prepare_edges(self):
        a, b, ids = [], [], []
        skip = self._removed_edge_ids()
        for e in self.model.edges.values():
            if e.id in skip or len(e.points) < 2:
                continue
            if e.etype not in ("CIRCLE", "LINE") and not getattr(e, "_from_mesh", False):
                continue
            p = e.points
            a.append(p[:-1])
            b.append(p[1:])
            ids += [e.id] * (len(p) - 1)
        self.edge_a = np.concatenate(a) if a else np.zeros((0, 3))
        self.edge_b = np.concatenate(b) if b else np.zeros((0, 3))
        self.edge_ids = np.array(ids)
        # Eckpunkte: Stellen, an denen mindestens zwei verschiedene Kanten zusammentreffen
        vmap = {}
        for e in self.model.edges.values():
            if e.id in skip:
                continue
            for p in (e.start, e.end):
                if p is None:
                    continue
                k = tuple(np.round(p, 4))
                vmap.setdefault(k, [p, set()])[1].add(e.id)
        self.vertices = np.array([v[0] for v in vmap.values() if len(v[1]) >= 2]) if vmap else np.zeros((0, 3))
        # Bohrungs-Eintrittspunkte (Achse ∩ Fläche) schräger und radialer Bohrungen – anklickbar
        self.entry_pts = []
        if self.model is not None and self.holes:
            mo, md = core.main_axis(self.model)
            for i, h in enumerate(self.holes):
                if abs(float(core.unit(h.axis_d) @ md)) < 0.9999:
                    self.entry_pts.append((np.asarray(h.entry, float), i))
        if self.entry_pts:
            self.vertices = np.vstack([self.vertices.reshape(-1, 3)] + [p[None, :] for p, _ in self.entry_pts])

    def _freeform_edges_from_mesh(self, V, T, F):
        """Freiform-Kanten (z. B. Verschneidung Querbohrung/Mantel) haben in der Datei keine Punkte.
        Ihr Verlauf wird aus dem Netz gewonnen: Dreieckskanten zwischen den beiden angrenzenden Flächen."""
        m = self.model
        todo = [e for e in m.edges.values() if e.etype not in ("CIRCLE", "LINE") and
                (getattr(e, "_from_mesh", False) or len(e.points) < 3)]
        if not todo:
            return False
        E = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
        ef = np.repeat(F, 3)
        nv = np.int64(len(V))
        key = E[:, 0] * nv + E[:, 1]
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        fmin = np.full(len(u), np.iinfo(np.int64).max)
        fmax = np.full(len(u), -1)
        np.minimum.at(fmin, inv, ef)
        np.maximum.at(fmax, inv, ef)
        UE = np.stack([u // nv, u % nv], axis=1)
        e2f = core._edge_faces(m)
        changed = False
        for e in todo:
            fs = sorted(e2f.get(e.id, ()))
            if len(fs) != 2:
                continue
            k = np.where((fmin == fs[0]) & (fmax == fs[1]))[0]
            if len(k) == 0:
                continue
            pts = self._chain_segments(UE[k], V, getattr(e, "_seed", e.points[0]))
            if pts is None or len(pts) < 2:
                continue
            if not getattr(e, "_from_mesh", False):
                e._seed = np.asarray(e.points[0], float).copy()
            e._from_mesh = True
            e.points = pts
            e.length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            changed = True
        return changed

    @staticmethod
    def _chain_segments(S, V, seed):
        """Segmente (Vertex-Paare) zu einem Linienzug verketten; Zusammenhangsteil nahe seed wählen."""
        adj = {}
        for a, b in S:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        seen, comps = set(), []
        for v0 in adj:
            if v0 in seen:
                continue
            comp, stack = [], [v0]
            seen.add(v0)
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            comps.append(comp)
        seed = np.asarray(seed, float)
        comp = min(comps, key=lambda c: float(np.linalg.norm(V[c] - seed, axis=1).min()))
        ends = [v for v in comp if len(adj[v]) == 1]
        start = ends[0] if ends else comp[0]
        path, prev, cur = [start], None, start
        used = set()
        while True:
            nxt = [w for w in adj[cur] if w != prev and (min(cur, w), max(cur, w)) not in used]
            if not nxt:
                break
            w = nxt[0]
            used.add((min(cur, w), max(cur, w)))
            prev, cur = cur, w
            path.append(cur)
            if cur == start:
                break
        return V[path]

    def _build_scene(self):
        for it in list(self.view.items):
            self.view.removeItem(it)
        self.hover_items = []
        self._hover_key = None
        self.items_overlay = []
        self.annot_items = []
        self.sel_callouts = []
        m = self.model
        rel = QUALITY.get(self.cb_quality.currentText())
        t0 = QtCore.QElapsedTimer()
        t0.start()
        if rel is None:
            V, T, F = core.refine_mesh(m, max_levels=0)
        else:
            V, T, F = core.refine_mesh(m, tol=max(m.size * rel, 2e-4), max_levels=7)
        V, T, F = self._apply_removed(V, T, F)
        # Gewindebereich scharf abgrenzen: Kernloch-Flächen an Gewindeanfang/-ende auftrennen
        for h in (getattr(self, "holes", None) or []):
            if not h.thread or h.thread_face < 0:
                continue
            ti = core.thread_info(m, h)
            if ti is None:
                continue
            ax = core.unit(h.axis_d)
            for t in (ti["t_from"], ti["t_to"]):
                V, T, F = core.split_face_by_plane(V, T, F, h.thread_face, h.entry - t * ax, ax)
        N = core.vertex_normals(m, V, T, F)
        flip = core.inward_faces(m)
        if flip:
            fm = np.isin(F, list(flip))
            N[fm] *= -1
        self.rtris = V[T]                      # (n,3,3) für Picking
        self.rface = F
        self.vert_face = np.repeat(F, 3)
        self.base_colors = self._base_vertex_colors()
        verts = self.rtris.reshape(-1, 3).astype(np.float32)
        self.normals = N.reshape(-1, 3).astype(np.float32)
        self.meshdata = gl.MeshData(vertexes=verts, faces=np.arange(len(verts)).reshape(-1, 3),
                                    vertexColors=self.base_colors.copy())
        self.meshdata._vertexNormals = self.normals
        self.mesh = gl.GLMeshItem(
            meshdata=self.meshdata, smooth=True, shader="s3d_cut", drawEdges=False,
            edgeColor=(0.15, 0.15, 0.15, 0.5),
            glOptions={GL.GL_DEPTH_TEST: True, GL.GL_BLEND: False, GL.GL_CULL_FACE: False,
                       GL.GL_POLYGON_OFFSET_FILL: True, "glPolygonOffset": (1.0, 1.0), "glDepthMask": (GL.GL_TRUE,),
                       "glColorMask": (GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)})
        self.view.addItem(self.mesh)
        # Glas-Darstellung (eigene Farben mit wenig Deckkraft, schreibt keine Tiefe)
        gcol = self.base_colors.copy()
        gcol[:, :3] = gcol[:, :3] * 0.45 + 0.55
        gcol[:, 3] = 0.10
        self.glass_md = gl.MeshData(vertexes=verts, faces=np.arange(len(verts)).reshape(-1, 3), vertexColors=gcol)
        self.glass_md._vertexNormals = self.normals
        self.glass_mesh = gl.GLMeshItem(meshdata=self.glass_md, smooth=True, shader="s3d_glasscut",
                                        glOptions=GL_GLASS)
        self.glass_mesh.setDepthValue(5)
        self.glass_mesh.setVisible(False)
        self.view.addItem(self.glass_mesh)
        self.hi_mesh = None
        # Tiefen-Maske für den Linien-Modus (verdeckte Linien ausblenden)
        self.depth_mesh = gl.GLMeshItem(meshdata=self.meshdata, smooth=True, shader="s3d_cut", glOptions=GL_DEPTHONLY)
        self.depth_mesh.setDepthValue(-20)
        self.depth_mesh.setVisible(False)
        self.view.addItem(self.depth_mesh)

        seg = self._filter_removed_seg(core.feature_edges(V, T, F))
        self.edge_seg = seg
        if self._freeform_edges_from_mesh(V, T, F):
            self._prepare_edges()
        # Daten für Umrisslinien (Silhouette) im Linien-Modus
        Ek = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
        nv = np.int64(len(V))
        key = Ek[:, 0] * nv + Ek[:, 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = np.where(ks[1:] == ks[:-1])[0]
        tri_of = order // 3
        self.sil_a = tri_of[same]
        self.sil_b = tri_of[same + 1]
        self.sil_seg = V[Ek[order[same]]]                 # (m,2,3)
        P = self.rtris
        tn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.tri_n = tn / np.linalg.norm(tn, axis=1, keepdims=True).clip(1e-15)
        self.sil_item = gl.GLLinePlotItem(pos=np.zeros((0, 3)), mode="lines", color=(0.05, 0.05, 0.08, 1.0),
                                          width=1.4, antialias=True, glOptions=GL_EDGES)
        self.sil_item.setVisible(False)
        self.edge_item = gl.GLLinePlotItem(pos=seg, mode="lines", color=(0.12, 0.11, 0.10, 1.0),
                                           width=1.2, antialias=True, glOptions=GL_EDGES)
        self.view.addItem(self.edge_item)
        self.view.addItem(self.sil_item)

        self.sel_edges = gl.GLLinePlotItem(pos=np.zeros((0, 3)), mode="lines",
                                           color=tuple(SEL_COLOR), width=4, antialias=True,
                                           glOptions=GL_ONTOP)
        self.view.addItem(self.sel_edges)
        self.sel_pts = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=11, color=(0.9, 0.1, 0.1, 1),
                                            pxMode=True)
        self.sel_pts.setGLOptions(GL_ONTOP)
        self.view.addItem(self.sel_pts)
        self.wcs_items = []
        self._build_wcs_items()
        self.lock_item = None
        if self.cb_lock.currentIndex() != 5:
            self.view.lock = self._lock_axis_for(self.cb_lock.currentIndex())
        self._show_lock_line()
        self._update_lock_badge()
        self.pattern_items = []
        self._build_pattern_items()
        self.section_items = []
        self._apply_section()
        self.box_items = []
        self._build_box_items()
        self._build_annotations()
        self.edge_item.setData(width=max(1.0, 1.5 * float(self.opt["line_width"])))
        self.update_display()
        self.status.showMessage(
            f"{len(m.faces)} Flächen · {len(m.edges)} Kanten · {len(m.features)} Features · "
            f"{len(m.tris)} → {len(T)} Dreiecke ({t0.elapsed()} ms)")

    def _build_wcs_items(self):
        for it in self.wcs_items:
            self.view.removeItem(it)
        self.wcs_items = []
        if self.model is None:
            return
        L = self.model.size * 0.12
        o = self.wcs.origin
        cols = [(0.85, 0.1, 0.1, 1), (0.1, 0.6, 0.1, 1), (0.1, 0.25, 0.9, 1)]
        font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold)
        for i, name in enumerate("XYZ"):
            d = self.wcs.R[i]
            ln = gl.GLLinePlotItem(pos=np.array([o, o + L * d]), color=cols[i], width=3, antialias=True)
            ln.setGLOptions(GL_ONTOP)
            ln.setDepthValue(10)
            self.view.addItem(ln)
            tx = gl.GLTextItem(pos=o + L * 1.1 * d, text=name, color=QtGui.QColor.fromRgbF(*cols[i]), font=font)
            self.view.addItem(tx)
            self.wcs_items += [ln, tx]
        dot = gl.GLScatterPlotItem(pos=np.array([o]), size=9, color=(0.1, 0.1, 0.1, 1), pxMode=True,
                                   glOptions=GL_ONTOP)
        self.view.addItem(dot)
        self.wcs_items.append(dot)
        for it in self.wcs_items:
            it.setVisible(self.a_wcs.isChecked())

    def update_display(self):
        if self.mesh is None:
            return
        shaded, wire = self.a_shaded.isChecked(), self.a_wire.isChecked()
        lines = self.a_lines.isChecked()
        glass = self.a_glass.isChecked() and not lines
        self.glass_mesh.setVisible(glass)
        if self.hi_mesh is not None:
            self.hi_mesh.setVisible(glass or lines)
        self.depth_mesh.setVisible(lines and not self.show_hidden)
        self.sil_item.setVisible(lines)
        if lines:
            self._update_silhouette()
        if lines:
            self.mesh.setVisible(False)
            c = self._contour_rgba()
            self.edge_item.setData(color=(c[0] * 0.5, c[1] * 0.5, c[2] * 0.5, 1.0))
            self.edge_item.setVisible(True)
            for it in self.wcs_items:
                it.setVisible(self.a_wcs.isChecked())
            return
        if glass:
            self.mesh.setVisible(False)
            self.edge_item.setData(color=(0.15, 0.15, 0.20, 0.35))
            self.edge_item.setVisible(self.a_edges.isChecked())
            for it in self.wcs_items:
                it.setVisible(self.a_wcs.isChecked())
            return
        self.edge_item.setData(color=self._contour_rgba())
        self.mesh.setVisible(shaded or wire)
        self.mesh.opts["drawFaces"] = shaded
        self.mesh.opts["drawEdges"] = wire
        self.mesh.update()
        self.edge_item.setVisible(self.a_edges.isChecked())
        for it in self.wcs_items:
            it.setVisible(self.a_wcs.isChecked())

    def fit_view(self):
        if self.model is None:
            return
        V = self.model.tris.reshape(-1, 3)
        lo, hi = V.min(axis=0), V.max(axis=0)
        c = (lo + hi) / 2
        r = float(np.linalg.norm(hi - lo)) / 2 or 1.0
        dist = r / math.tan(math.radians(self.view.opts["fov"]) / 2) * 1.1
        self.view.pan_ndc = [0.0, 0.0]
        self.view.setCameraPosition(pos=QtGui.QVector3D(*c), distance=dist)
        self.view.center_on_lock()

    # ---------------- Drehachse sperren ----------------
    def _lock_axis_for(self, idx):
        m = self.model
        if m is None or idx == 0:
            return None
        if idx == 1:
            ax = m.axes.get(m.main_axis)
            if ax is None:
                # Ersatz: längste Ausdehnung durch Modellmitte
                V = m.tris.reshape(-1, 3)
                c = (V.min(axis=0) + V.max(axis=0)) / 2
                d = np.zeros(3)
                d[int(np.argmax(V.max(axis=0) - V.min(axis=0)))] = 1
                return c, d
            return np.array(ax[0], float), core.unit(ax[1])
        if idx in (2, 3, 4):
            return self.wcs.origin.copy(), self.wcs.R[idx - 2].copy()
        if idx == 5:
            for e in self.sel:
                ax = e.axis()
                if ax is not None:
                    return np.array(ax[0], float), core.unit(ax[1])
                pr = e.primary()
                if pr.kind == "line":
                    return pr.p, core.unit(pr.d)
                if pr.kind == "plane":
                    return pr.p, core.unit(pr.d)
            return None
        return None

    def keys_table_html(self):
        """Tabelle aller Tasten aus der aktuellen Tastenbelegung."""
        h = "<table cellspacing=0 cellpadding=4 width='100%'>"
        i = 0
        for aid, (a, text, default, _) in self.actions_reg.items():
            k = a.shortcut().toString(QtGui.QKeySequence.SequenceFormat.NativeText) if a is not None else ""
            if not k:
                continue
            k = k.replace("Space", "Leertaste").replace("Ctrl", "Strg").replace("Shift", "Umschalt")
            bg = " style='background:#f3f7fd'" if i % 2 else ""
            h += f"<tr{bg}><td width='42%' style='color:#6b7890'>{text}</td><td><b>{k}</b></td></tr>"
            i += 1
        return h + "</table>"

    def open_manual(self):
        mw = QtWidgets.QApplication.activeModalWidget()
        if isinstance(mw, OptionsDialog):
            mw.reject()          # Eigenschaften-Fenster sperrt sonst das Handbuch
        m = getattr(self, "_manual", None)
        if m is None:
            m = self._manual = ManualDialog(self)
            if str(self.settings.value("manual_pin", "false")).lower() in ("1", "true"):
                m.pin.setChecked(True)
        m.show()
        if m.isMinimized():
            m.showNormal()
        m.raise_()
        m.activateWindow()

    def help_html(self):
        """Hilfe mit allen Bedienbefehlen; Tasten aus der aktuellen Tastenbelegung."""
        def key(aid):
            a = self.actions_reg.get(aid, (None,))[0]
            if a is None:
                return ""
            k = a.shortcut().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
            return k.replace("Space", "Leertaste").replace("Ctrl", "Strg").replace("Shift", "Umschalt")

        def table(rows):
            h = "<table cellspacing=0 cellpadding=4 width='100%'>"
            for i, (a, b) in enumerate(rows):
                bg = " style='background:#f3f7fd'" if i % 2 else ""
                h += f"<tr{bg}><td width='42%' style='color:#6b7890'>{a}</td><td><b>{b}</b></td></tr>"
            return h + "</table>"

        H = lambda t: f"<h3 style='color:#1d4fb8; margin:14px 0 4px 0'>{t}</h3>"
        h = (f"<div style='font-size:large; font-weight:700'><span style='color:#1f6fe5'>3D</span> VIEWER "
             f"<span style='color:#6b7890; font-weight:400'>– Hilfe (Version {APP_VERSION})</span></div>"
             "<p style='color:#4c5a73'>Tipp: Rechtsklick (ohne Ziehen) ins 3D-Bild öffnet ein Menü mit allen "
             "Funktionen. Hält man die Maus kurz auf einen Knopf, zeigt er seine Taste. "
             f"Alles verstellt? <b>Zurücksetzen</b> ({key('reset_all')}).</p>")
        h += H("Maus")
        h += table([("Drehen", "linke Maustaste ziehen"), ("Drehen (feiner)", "linke Maustaste + Strg"),
                    ("Verschieben", "rechte Maustaste ziehen"),
                    ("Zoomen", "mittlere Maustaste ziehen"),
                    ("Fläche / Kante wählen", "Klick (wird rot), nochmal = abwählen"),
                    ("Maß zwischen zwei Elementen", "zwei Elemente nacheinander anklicken"),
                    ("Mehrere wählen", "Strg + Klick"), ("Punkt setzen (Koordinaten)", "Umschalt + Klick"),
                    ("Drehpunkt setzen", "Doppelklick"), ("Rechteckauswahl", "Alt + linke Maustaste ziehen"),
                    ("Kontextmenü", "Rechtsklick ohne Bewegen")])
        h += H("Mausrad")
        h += table([("Zoomen", "Mausrad"), ("Zoomen zum Mauszeiger", "Strg + Mausrad"),
                    ("Feiner zoomen", "Umschalt + Mausrad (ohne Schnitt)"),
                    ("Um die Mittelachse drehen", "Alt + Mausrad (5° je Raste)"),
                    ("Bedienoberfläche größer/kleiner", "AltGr + Mausrad"),
                    ("Schnitt verschieben", "Umschalt + Mausrad (fein: + Alt)"),
                    ("Schnitt kippen", "Strg + Umschalt + Mausrad (1° je Raste)")])
        h += H("Schnitt")
        h += table([("Schnitt ein / Längs → Quer → aus", key("section")),
                    ("Seite wechseln", key("section_flip")),
                    ("Verschieben", "Pfeiltasten (1 mm, mit Umschalt 10 mm)"),
                    ("Kippen", "Strg + Pfeiltasten (1°, mit Umschalt 5°)"),
                    ("Ebene mit der Maus drehen / verschieben", "Umschalt + linke / rechte Maustaste")])
        h += H("Messen und Prüfen")
        h += table([("Durchmesser, Abstand, Wandstärke, Winkel", "zwei Flächen/Kanten anklicken"),
                    ("Radius-Mittelpunkt (Z und X als Ø)", "Radius anklicken"),
                    ("Gewindetiefe und Kernloch", "Gewindebohrung anklicken"),
                    ("Scharfe Ecke ohne Radius/Fase", f"Radius anklicken → {key('del_fillet')} "
                                                     f"(zurück: {key('restore_fillet')})"),
                    ("C0° auf eine Bohrung legen", "Bohrung anklicken → Rechtsklick → C0° auf Auswahl legen"),
                    ("Nullpunkt setzen", "rechts im Reiter Nullpunkt"),
                    ("Alle Maße des Drehteils (Ø, Planflächen, Fasen, Radien)", "rechts: Drehteil → Maßliste"),
                    ("Drehkontur als Punktliste / DXF / G-Code", "rechts: Drehteil → Kontur "
                                                                 "(Steuerung oben einstellen)"),
                    ("Eintritt schräger/radialer Bohrung (X Ø, Y, Z, C, B)",
                     "roten Punkt in der Bohrung anklicken (Annotationen an, Transparent aus)"),
                    ("Senkung, Bohrspitze, Tiefe bis voller Ø, Maschinenwerte", "rechts: Bohrungen"),
                    ("Einstiche / Freistiche", "rechts: Drehteil → Einstiche"),
                    ("Welche Platte kommt hin? (35°/55°/60°/80°)", "rechts: Drehteil → Werkzeug"),
                    ("Nuten, Schlüsselflächen (Breite/SW, Tiefe, Z, C)", "rechts: Drehteil → Fräsen"),
                    ("Kegel 1:x, Normkegel (Morse, 7:24 …)", "Drehteil → Maßliste / Kontur, Spalte Info")])
        h += H("Alle Tasten")
        h += self.keys_table_html()
        h += (f"<p style='color:#6b7890; margin-top:14px'>© 2026 Christian Diezinger · "
              f"<a href='{APP_URL}'>{APP_URL.replace('https://', '')}</a></p>")
        return h

    def reset_all(self):
        """Alles wie frisch geladen: Darstellungsschalter, Schnitt, Nullpunkt, Auswahl, Fenster, Größe."""
        self.view.stop_animation()
        if self.cb_lock.currentIndex() != 0:
            self.cb_lock.setCurrentIndex(0)
        self.view.lock = None
        defaults = dict(self._act_default)
        defaults["thread_col"] = True
        for aid, on in defaults.items():
            if aid in ("dock_left", "dock_right", "tree", "zrot"):
                continue
            a = self.actions_reg[aid][0]
            if a.isChecked() != on:
                a.trigger() if self.model is not None else a.setChecked(on)
        self.settings.setValue("thread_color_on", True)
        self.annot_types = {"Gewinde": True, "Eintritt": True, "only_selected": False}
        self.nav_mode = False
        self.show_hidden = False
        self.settings.setValue("show_hidden", 0)
        # Fenster wieder an ihren Platz
        for dock in (self.dock_left, self.dock_right):
            dock.setFloating(False)
            dock.setVisible(True)
        self.removeDockWidget(self.dock_left)
        self.removeDockWidget(self.dock_right)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_left)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_right)
        self.dock_left.show(); self.dock_right.show()
        if getattr(self, "dock_views", None) is not None:
            self.dock_views.hide()
        self._sync_dock_actions()
        self.tabs_left.setCurrentIndex(0)
        self.tabs_right.setCurrentIndex(0)
        if self.ui_scale != 100:
            self.set_ui_scale(100)
        # Bauteil neu laden: Auswahl, Schnitt, Nullpunkt, gelöschte Radien, Drehpunkt – alles frisch
        if self.model is not None and os.path.isfile(self.model.path):
            self.load(self.model.path)
            self.view.pan_ndc = [0.0, 0.0]
            self.home_view()
        self.view.update()
        self.status.showMessage("Alles zurückgesetzt – wie frisch geladen.")

    def toggle_lock(self):
        if self.cb_lock.currentIndex() == 0:
            self.cb_lock.setCurrentIndex(self.lock_choice)
        else:
            self.cb_lock.setCurrentIndex(0)
        self._sync_lock_action()

    def _sync_lock_action(self):
        self.a_lock.blockSignals(True)
        self.a_lock.setChecked(self.view.lock is not None)
        self.a_lock.blockSignals(False)

    def on_lock(self, idx):
        if idx in (1, 2, 3, 4):
            self.lock_choice = idx
            self.settings.setValue("lock_choice2", idx)
        if self.model is None:
            self.view.lock = None
            self._sync_lock_action()
            self._update_lock_badge()
            return
        lk = self._lock_axis_for(idx)
        if idx == 5 and lk is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Bitte zuerst eine Bohrung/Zylinderfläche, Kreiskante, gerade Kante "
                "oder Planfläche (Normale) anklicken.")
            self.cb_lock.blockSignals(True)
            self.cb_lock.setCurrentIndex(0)
            self.cb_lock.blockSignals(False)
            lk = None
        self.view.lock = lk
        self._sync_lock_action()
        self._show_lock_line()
        self._update_lock_badge()
        if lk is not None:
            self.view.center_on_lock()
            self.view.update()
            self.status.showMessage("Drehung gesperrt auf: " + self.cb_lock.currentText()
                                    + " – mit linker Maustaste ziehen (endlos drehbar)")
        else:
            self.status.showMessage("Freies Drehen")

    def _choice_name(self):
        return {1: "Längsachse", 2: "X-Achse", 3: "Y-Achse", 4: "Z-Achse"}.get(self.lock_choice, "Längsachse")

    def _update_lock_badge(self):
        b = getattr(self, "lock_badge", None)
        if b is None:
            return
        if self.view.lock is not None:
            b.setText(f"🔒 Drehung gesperrt: {self.cb_lock.currentText()}   (Z = frei)")
            self.setWindowTitle(self.windowTitle().split("  [")[0] + f"  [🔒 {self.cb_lock.currentText()}]")
            if getattr(self.view, "glass", False):
                b.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #7db3ff, stop:1 #3c7cf0);"
                                "color:white; padding:6px 14px; border-radius:13px; font-weight:600;"
                                "border:1px solid rgba(255,255,255,0.7);")
            else:
                b.setStyleSheet("background:rgba(110,30,140,215); color:white; padding:4px 8px;"
                                "border-radius:4px; font-weight:bold;")
        else:
            b.setText(f"Drehung frei   (Z = auf {self._choice_name()} sperren)")
            self.setWindowTitle(self.windowTitle().split("  [")[0])
            if getattr(self.view, "glass", False):
                b.setStyleSheet("background:rgba(255,255,255,0.78); color:#24304a; padding:6px 14px;"
                                "border-radius:13px; border:1px solid rgba(160,185,225,0.75);")
            else:
                b.setStyleSheet("background:rgba(60,55,50,170); color:white; padding:4px 8px; border-radius:4px;")
        if getattr(self, "section", None) is not None:
            b.setText(b.text() + f"     ✂ {self._section_label()}")
        b.adjustSize()

    def _show_lock_line(self):
        if getattr(self, "lock_item", None) is not None:
            try:
                self.view.removeItem(self.lock_item)
            except Exception:
                pass
            self.lock_item = None
        if self.view.lock is None or self.model is None:
            return
        p, a = self.view.lock
        V = self.model.tris.reshape(-1, 3)
        t = (V - p) @ a
        ext = (t.max() - t.min()) * 0.15 + 1
        pts = np.array([p + (t.min() - ext) * a, p + (t.max() + ext) * a])
        self.lock_item = gl.GLLinePlotItem(pos=pts, color=(0.55, 0.1, 0.7, 0.9), width=2,
                                           antialias=True, glOptions=GL_ONTOP)
        self.view.addItem(self.lock_item)

    # ---------------- Optionen (Eigenschaften) ----------------
    OPT_DEFAULTS = {
        "design": "Glas",
        "bg_top": "#c3d3ec", "bg_bottom": "#f4f7fc",
        "text_scale": 1.0, "text_color": "#1c2740", "text_bg": "#ffffff", "text_bg_alpha": 0.82,
        "tip_radius": 12,
        "line_width": 0.8, "hl_width": 2.0, "contour_color": "#3a4458",
        "hover_on": True, "hover_color": "#805b9bff", "hover_sel_color": "#80ff4747",
        "sel_color": "#ff0000", "dim_color": "#888888", "thread_color": "#20b46c",
        "wheel_invert": False, "keys_on_buttons": False, "zoom_step": 10.0,
    }

    def _load_options(self):
        if str(self.settings.value("design_version", "")) != "2":
            for k in DESIGN_COLORS["Glas"]:
                self.settings.remove("opt/" + k)
            self.settings.remove("opt/design")
            self.settings.setValue("design_version", "2")
        o = dict(self.OPT_DEFAULTS)
        for k, v in self.OPT_DEFAULTS.items():
            val = self.settings.value("opt/" + k)
            if val is None:
                continue
            try:
                if isinstance(v, bool):
                    o[k] = str(val).lower() in ("1", "true", "yes")
                elif isinstance(v, float):
                    o[k] = float(val)
                elif isinstance(v, int):
                    o[k] = int(val)
                else:
                    o[k] = str(val)
            except (TypeError, ValueError):
                pass
        return o

    def _save_options(self):
        for k, v in self.opt.items():
            self.settings.setValue("opt/" + k, v)

    def _annot_bg(self):
        """Hinweisfelder (Gewinde, Bohrbild): Gelb (klassisch) bzw. helles Bernstein im Glas-Design."""
        return QtGui.QColor("#f2e3a6") if getattr(self.view, "glass", False) else QtGui.QColor("#c3bc62")

    @staticmethod
    def _qc(s):
        return QtGui.QColor(s)

    def _rgba(self, s):
        c = QtGui.QColor(s)
        return (c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def _apply_options(self):
        o = self.opt
        v = self.view
        v.bg_top, v.bg_bottom = self._qc(o["bg_top"]), self._qc(o["bg_bottom"])
        v.text_scale = float(o["text_scale"])
        v.wheel_invert = bool(o["wheel_invert"])
        v.zoom_step = max(0.01, min(0.4, float(o.get("zoom_step", 10.0)) / 100.0))
        v.tip_radius = int(o["tip_radius"])
        global SEL_COLOR, MEAS_COLOR
        SEL_COLOR = np.array(self._rgba(o["sel_color"]), np.float32)
        MEAS_COLOR = self._rgba(o["dim_color"])
        if self.model is not None and self.mesh is not None:
            self.edge_item.setData(width=max(1.0, 1.5 * float(o["line_width"])))
            self.sel_edges.setData(color=tuple(SEL_COLOR), width=max(2.0, 2.0 * float(o["hl_width"])))
            self.update_display()
            self.on_nocolor()
            self._build_annotations()
        v.update()

    def _contour_rgba(self):
        return self._rgba(self.opt["contour_color"])

    def open_options(self, help_tab=False):
        dlg = OptionsDialog(self)
        dlg.tabs.setCurrentIndex(0)           # immer mit dem Reiter „Hilfe“ beginnen
        if dlg.exec():
            pct = dlg.ui_combo.currentData()
            if pct and int(pct) != self.ui_scale:
                self.set_ui_scale(int(pct))
            self.opt = dlg.values()
            self._save_options()
            km = dlg.keymap()
            import json
            self.settings.setValue("keymap_json", json.dumps(km))
            self.apply_keymap(km)
            self._apply_options()
            self.apply_theme()
            if self.sel:
                self._refresh_selection()

    # ---------------- Ansicht-Funktionen ----------------
    def reload(self):
        if self.model is not None and os.path.isfile(self.model.path):
            self.load(self.model.path)

    def home_view(self):
        if self.model is None:
            return
        self.view.stop_animation()
        self.std_view(30, 45)

    def view_normal(self):
        if self.model is None or not self.sel:
            self.status.showMessage("Normal auf: bitte zuerst eine Fläche, Bohrung oder Kreiskante wählen")
            return
        e = self.sel[0]
        n = None
        if e.kind == "face" and isinstance(e.face.surf, core.PlaneSurf):
            n = core.outward_normal(self.model, e.ref, e.face.surf.n)
        else:
            ax = e.axis()
            if ax is not None:
                n = core.unit(ax[1])
                if n @ self.view.toward_camera() < 0:
                    n = -n
            elif e.primary().kind == "line":
                self.status.showMessage("Normal auf: Gerade hat keine eindeutige Blickrichtung")
                return
        if n is None:
            return
        self.view.set_view_dir(n, self.wcs.R[0] if abs(self.wcs.R[0] @ n) < 0.9 else self.wcs.R[1])
        c = self._entity_center(e)
        self.view.opts["center"] = QtGui.QVector3D(*c)
        self.view.pan_ndc = [0.0, 0.0]
        self.view.center_on_lock()
        self.view.update()
        self._update_silhouette()
        self.view.viewChanged.emit()

    def view_anim_start(self):
        if self.model is not None:
            self.view.start_animation()
            self.status.showMessage("Animation läuft – Esc oder Mausklick hält an")

    def on_escape(self):
        if not self.view.stop_animation():
            self.clear_selection()

    def on_perspective(self):
        self.view.set_perspective(self.a_persp.isChecked())
        self.view.viewChanged.emit()

    def _toggle_dock(self, dock, act):
        dock.setVisible(act.isChecked())
        if act.isChecked() and dock.isFloating():
            dock.setFloating(False)

    def _sync_dock_actions(self):
        for dock, act in ((self.dock_left, getattr(self, "a_dock_l", None)),
                          (self.dock_right, getattr(self, "a_dock_r", None)),
                          (self.dock_left, getattr(self, "a_tree", None))):
            if act is not None and act.isChecked() != dock.isVisible():
                act.blockSignals(True)
                act.setChecked(dock.isVisible())
                act.blockSignals(False)

    def show_about(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Über")
        dlg.setFixedWidth(360)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(24, 22, 24, 18)
        lay.setSpacing(6)
        ico = QtWidgets.QLabel()
        pm = self.windowIcon().pixmap(88, 88) if not self.windowIcon().isNull() else None
        if pm is not None and not pm.isNull():
            ico.setPixmap(pm)
            ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(ico)
        t = QtWidgets.QLabel("<div style='font-size:19pt; font-weight:800; letter-spacing:1px'>"
                             "<span style='color:#1f6fe5'>3D</span> <span style='color:#2b3446'>VIEWER</span></div>"
                             f"<div style='color:#6b7890'>Version {APP_VERSION}</div>")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        lay.addSpacing(8)
        a = QtWidgets.QLabel("<span style='color:#6b7890'>Besitzer und Entwickler</span><br>"
                             "<b style='font-size:12pt'>Christian Diezinger</b>")
        a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(a)
        lay.addSpacing(4)
        gh = QtWidgets.QLabel(f"<a href='{APP_URL}' style='color:#1d4fb8; text-decoration:none'>"
                              f"{APP_URL.replace('https://', '')}</a>")
        gh.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gh.setOpenExternalLinks(True)
        gh.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        gh.setToolTip("GitHub-Seite im Browser öffnen")
        lay.addWidget(gh)
        c = QtWidgets.QLabel("<span style='color:#6b7890'>© 2026 Christian Diezinger<br>Freie Software (MIT-Lizenz)</span>")
        c.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addSpacing(6)
        lay.addWidget(c)
        bb = QtWidgets.QDialogButtonBox()
        ok = bb.addButton("OK", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setMinimumWidth(90)
        bb.accepted.connect(dlg.accept)
        bb.setCenterButtons(True)
        lay.addSpacing(8)
        lay.addWidget(bb)
        dlg.exec()

    def toggle_tree(self):
        vis = self.a_tree.isChecked()
        self.dock_left.setVisible(vis)
        if vis:
            self.tabs_left.setCurrentIndex(2)       # Features

    def show_info(self):
        self.dock_left.setVisible(True)
        self.a_tree.setChecked(True)
        self._sync_dock_actions()
        self.tabs_left.setCurrentIndex(0)           # Modell

    def _face_color_table(self):
        if self.a_nocolor.isChecked():
            return np.array([[0.72, 0.72, 0.72, 1.0]] * len(self.model.faces), np.float32)
        return np.array([[*f.color, 1.0] for f in self.model.faces], np.float32)

    def _base_vertex_colors(self):
        """Farben je Netz-Ecke: Flächenfarbe, Gewindebereiche (Gewindetiefe) eingefärbt."""
        cols = np.repeat(self._face_color_table()[self.rface], 3, axis=0)
        if not getattr(self, "a_thread_col", None) or not self.a_thread_col.isChecked():
            return cols
        tc = np.array(self._rgba(self.opt.get("thread_color", "#20b46c")), np.float32)
        C = self.rtris.mean(axis=1)                      # Dreiecks-Mittelpunkte (Ecken je Dreieck eigen)
        for h in (getattr(self, "holes", None) or []):
            if not h.thread or h.thread_face < 0:
                continue
            ti = core.thread_info(self.model, h)
            if ti is None:
                continue
            m = self.rface == h.thread_face
            if not m.any():
                continue
            ax = core.unit(h.axis_d)
            dep = (h.entry - C[m]) @ ax
            sel = (dep >= ti["t_from"]) & (dep <= ti["t_to"])
            tri = np.where(m)[0][sel]
            idx = (tri[:, None] * 3 + np.arange(3)[None, :]).ravel()
            cols[idx] = tc
        return cols

    def toggle_thread_color(self):
        self.settings.setValue("thread_color_on", self.a_thread_col.isChecked())
        self.on_nocolor()
        if self.a_thread_col.isChecked():
            self.status.showMessage("Gewinde farbig: grüner Bereich = Gewinde (genaue Tiefe: Bohrung anklicken)")

    def on_nocolor(self):
        if self.model is None or self.mesh is None:
            return
        self.base_colors = self._base_vertex_colors()
        gcol = self.base_colors.copy()
        gcol[:, :3] = gcol[:, :3] * 0.45 + 0.55
        gcol[:, 3] = 0.10
        self.glass_md.setVertexColors(gcol)
        self.glass_md._vertexNormals = self.normals
        self.glass_mesh.meshDataChanged()
        self._refresh_selection()

    def set_rotation_center(self, pos=None):
        if self.model is None:
            return
        if pos is None:
            c = self._model_center()
        else:
            fh = self._pick_face(pos)
            if fh is None:
                return
            c = fh[1]
        self.view.opts["center"] = QtGui.QVector3D(*c)
        self.view.center_on_lock()
        self.view.update()
        self.status.showMessage("Drehmittelpunkt gesetzt" if pos is not None else "Drehmittelpunkt zurückgesetzt")

    def set_nav_mode(self, nav):
        self.nav_mode = nav
        if nav:
            self._clear_hover()
        self.status.showMessage("Viewermodus: Navigieren (Klicks wählen nichts aus)" if nav
                                else "Viewermodus: Messen")

    # ---------------- Radius/Fase löschen (scharfe Ecke) ----------------
    def _fillet_targets(self, pos=None):
        if self.model is None:
            return []
        t = [e.ref for e in self.sel if e.kind == "face" and self.model.faces[e.ref].ftype in ("TORUS", "SPHERE", "CONE")]
        if not t and pos is not None:
            h = self._pick_face(pos)
            if h is not None and self.model.faces[h[0]].ftype in ("TORUS", "SPHERE", "CONE"):
                t = [h[0]]
        return t

    def _removed_edge_ids(self):
        out = set()
        for fi in getattr(self, "removed", {}):
            out.update(self.model.faces[fi].edges)
        return out

    @staticmethod
    def _axis_frame(d):
        e1 = np.cross(d, [1.0, 0, 0])
        if np.linalg.norm(e1) < 0.1:
            e1 = np.cross(d, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(d, e1)

    def _fillet_span(self, fi, o, d, e1, e2):
        f = self.model.faces[fi]
        if f.face_angle is not None and f.face_angle >= 359:
            return 0.0, 2 * math.pi
        P = f.triangles.reshape(-1, 3) - o
        th = np.sort(np.mod(np.arctan2(P @ e2, P @ e1), 2 * math.pi))
        if len(th) < 2:
            return 0.0, 2 * math.pi
        gaps = np.diff(np.append(th, th[0] + 2 * math.pi))
        k = int(np.argmax(gaps))
        if gaps[k] < 0.15:
            return 0.0, 2 * math.pi
        start = th[(k + 1) % len(th)]
        return float(start), float(2 * math.pi - gaps[k])

    def _apply_removed(self, V, T, F):
        rem = getattr(self, "removed", {})
        if not rem:
            return V, T, F
        T0, F0 = T, F
        keep = ~np.isin(F, list(rem))
        T, F = T[keep], F[keep]
        Vn, Tn, Fn = [V], [T], [F]
        base = len(V)
        for fi, c in rem.items():
            o, d = c["o"], c["d"]
            e1, e2 = self._axis_frame(d)
            th0, span = c["span"]
            nseg = max(16, int(round(512 * span / (2 * math.pi))))
            th = th0 + np.linspace(0, span, nseg + 1)
            ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
            for j, ep in zip(c["faces"], c["edge_pts"]):
                if ep is None:
                    continue
                r0 = o + ep[0] * d + ep[1] * ring
                r1 = o + c["a"] * d + c["rho"] * ring
                pts = np.concatenate([r0, r1])
                n = nseg + 1
                i = np.arange(nseg)
                tri = np.concatenate([np.stack([i, i + 1, n + i], 1), np.stack([i + 1, n + i + 1, n + i], 1)])
                # Umlaufsinn wie bei der Nachbarfläche (Außenseite für Schnitt-Schraffur)
                sj = self._winding_sign(V, T0, F0, j)
                surf = self.model.faces[j].surf
                if surf is not None:
                    Q = pts[tri]
                    tn = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
                    sn = surf.normals(Q.mean(axis=1))
                    bad = np.sign(np.einsum("ij,ij->i", tn, sn)) != sj
                    tri[bad] = tri[bad][:, [0, 2, 1]]
                Vn.append(pts)
                Tn.append(tri + base)
                Fn.append(np.full(len(tri), j, dtype=F.dtype))
                base += len(pts)
        return np.concatenate(Vn), np.concatenate(Tn), np.concatenate(Fn)

    def _winding_sign(self, V, T, F, j):
        surf = self.model.faces[j].surf
        k = np.where(F == j)[0]
        if surf is None or len(k) == 0:
            return 1.0
        Q = V[T[k]]
        tn = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
        sn = surf.normals(Q.mean(axis=1))
        return 1.0 if np.einsum("ij,ij->i", tn, sn).sum() >= 0 else -1.0

    def _filter_removed_seg(self, seg):
        rem = getattr(self, "removed", {})
        if not rem or len(seg) == 0:
            return seg
        S = seg.reshape(-1, 2, 3)
        drop = np.zeros(len(S), bool)
        tol = max(1e-3, 2e-4 * self.model.size)
        for c in rem.values():
            o, d = c["o"], c["d"]
            for ep in c["edge_pts"]:
                if ep is None:
                    continue
                ok = np.ones(len(S), bool)
                for k in (0, 1):
                    q = S[:, k] - o
                    a = q @ d
                    r = np.linalg.norm(q - np.outer(a, d), axis=1)
                    ok &= (np.abs(a - ep[0]) < tol) & (np.abs(r - ep[1]) < tol)
                drop |= ok
        return S[~drop].reshape(-1, 3)

    def delete_fillet(self, targets=None):
        if self.model is None:
            return
        if not targets:
            targets = self._fillet_targets()
        if not targets:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Bitte zuerst einen Radius (Torusfläche) oder eine Fase (Kegelfläche) anklicken.")
            return
        msgs, new = [], []
        for fi in targets:
            if fi in self.removed:
                continue
            f = self.model.faces[fi]
            try:
                c = core.sharp_corner(self.model, fi)
            except ValueError as ex:
                msgs.append(f"{f.id}: {ex}")
                continue
            e1, e2 = self._axis_frame(c["d"])
            c["span"] = self._fillet_span(fi, c["o"], c["d"], e1, e2)
            eid = f"Ecke-{f.id}"
            th0, span = c["span"]
            th = th0 + np.linspace(0, span, max(9, int(round(180 * span / (2 * math.pi)))) + 1)
            ctr = c["o"] + c["a"] * c["d"]
            rho = max(c["rho"], 1e-6)
            pts = ctr + rho * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
            self.model.edges[eid] = core.Edge(id=eid, etype="CIRCLE", points=pts, length=rho * span,
                                              radius=rho, center=ctr, normal=c["d"].copy(),
                                              start=pts[0], end=pts[-1])
            c["edge"] = eid
            self.removed[fi] = c
            new.append(eid)
        if new:
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._prepare_edges()
                self._build_scene()
                self._build_annotations()
                self.sel = [core.Entity(self.model, "edge", new[-1])]
                self._refresh_selection()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        if msgs:
            QtWidgets.QMessageBox.information(self, APP_NAME, "\n".join(msgs))

    def restore_fillets(self):
        if self.model is None or not getattr(self, "removed", None):
            return
        for c in self.removed.values():
            self.model.edges.pop(c.get("edge"), None)
        self.removed = {}
        self.sel = [e for e in self.sel if not (e.kind == "edge" and str(e.ref).startswith("Ecke-"))]
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._prepare_edges()
            self._build_scene()
            self._build_annotations()
            self._refresh_selection()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.status.showMessage("Radien/Fasen wiederhergestellt.")

    def on_context_menu(self, pos):
        m = QtWidgets.QMenu(self)

        def add(aid, menu=m):
            a = self.actions_reg[aid][0]
            ma = menu.addAction(self.actions_reg[aid][1])
            ma.setShortcut(a.shortcut())
            ma.setCheckable(a.isCheckable())
            ma.setChecked(a.isChecked())
            ma.triggered.connect(lambda _=False, a=a: a.trigger())
            return ma

        tgt = self._fillet_targets(pos)
        if tgt:
            kinds = {self.model.faces[i].ftype for i in tgt}
            txt = ("Radius" if kinds == {"TORUS"} else "Fase" if kinds == {"CONE"} else "Radius/Fase")
            da = m.addAction(f"{txt} löschen (scharfe Ecke)", lambda t=tgt: self.delete_fillet(t))
            da.setShortcut(self.actions_reg["del_fillet"][0].shortcut())
        if getattr(self, "removed", None):
            ra = m.addAction(f"Radien/Fasen wiederherstellen ({len(self.removed)})", self.restore_fillets)
            ra.setShortcut(self.actions_reg["restore_fillet"][0].shortcut())
        if self.sel:
            m.addAction("C0° auf Auswahl legen", self.wcs_align_x)
        m.addAction("C0° um 180° drehen", lambda: self._wcs_op(lambda: self.wcs.rotate_z(180)))
        m.addAction("C0° um Winkel drehen …", self.wcs_rotate_free)
        m.addSeparator()
        if tgt or getattr(self, "removed", None):
            m.addSeparator()
        m.addAction("Drehmittelpunkt setzen", lambda: self.set_rotation_center(pos))
        m.addAction("Drehmittelpunkt zurücksetzen", lambda: self.set_rotation_center(None))
        m.addSeparator()
        pm = m.addMenu("Vorschaubilder anzeigen")
        for i, v in enumerate(self._all_views()):
            pm.addAction(f"Vorschaubild {i + 1} ({v['name']})", lambda v=v: self.apply_saved_view(v))
        vm = m.addMenu("Viewermodus")
        a1 = vm.addAction("Messen", lambda: self.set_nav_mode(False))
        a2 = vm.addAction("Navigieren", lambda: self.set_nav_mode(True))
        for a, on in ((a1, not self.nav_mode), (a2, self.nav_mode)):
            a.setCheckable(True)
            a.setChecked(on)
        am = m.addMenu("Ansicht")
        for aid in ("view_left", "view_right", "view_top", "view_bottom", "view_front", "view_back", "view_iso"):
            add(aid, am)
        am.addSeparator()
        names = ["CAD"] + ([n for n, _ in self.model.views] if self.model is not None else [])
        for nm in names:
            fa = am.addAction(nm, lambda nm=nm: self.set_view_frame(nm))
            fa.setCheckable(True)
            fa.setChecked(nm == self.view_frame_name)
        am.addSeparator()
        for aid in ("view_normal", "fit", "home", "mosaic"):
            add(aid, am)
        am.addSeparator()
        add("anim_start", am)
        ma = am.addAction("Animation anhalten", self.view.stop_animation)
        ma.setShortcut(self.actions_reg["escape"][0].shortcut())
        rm = m.addMenu("Rechteckselektion")
        rm.addAction("Rahmen aufziehen (Alt + linke Maustaste)", self.start_rect_selection)
        rm.addSeparator()

        def rchk(menu, key, text):
            ra = menu.addAction(text)
            ra.setCheckable(True)
            ra.setChecked(self._rect_filter(key))
            ra.toggled.connect(lambda on, key=key: self._set_rect_filter(key, on))

        rchk(rm, "faces", "Flächen auswählen")
        tm = rm.addMenu("Flächentyp auswählen")
        for key, text in (("PLANE", "Ebenen auswählen"), ("CONE", "Konusflächen auswählen"),
                          ("CYL_CONVEX", "Konvexe Zylinderflächen auswählen"),
                          ("CYL_CONCAVE", "Konkave Zylinderflächen auswählen"),
                          ("TORUS", "Torusflächen auswählen"), ("FREEFORM", "Freiformflächen auswählen")):
            rchk(tm, key, text)
        rchk(rm, "edges", "Kanten auswählen")
        rchk(rm, "annot", "Annotationen auswählen")
        rchk(rm, "hidden", "Verdeckte Elemente auswählen")
        anm = m.addMenu("Annotationen nach Typ anzeigen")
        for key, text in (("only_selected", "Nur Annotationen der gewählten Flächen anzeigen"),
                          ("Gewinde", "Gewinde anzeigen"),
                          ("Gewindetiefe", "Gewinde- und Kernlochtiefe anzeigen"),
                          ("Eintritt", "Eintrittspunkte schräger/radialer Bohrungen (rot)")):
            aa = anm.addAction(text)
            aa.setCheckable(True)
            aa.setChecked(self.annot_types.get(key, False))
            aa.toggled.connect(lambda on, key=key: self._set_annot_type(key, on))
        m.addSeparator()
        for aid in ("tess", "glass", "lines", "nocolor", "box", "wcs", "contours", "zrot", "annot", "section"):
            add(aid)
        m.addSeparator()
        for aid in ("persp", "dock_left", "dock_right", "tree", "detail_views", "info", "refresh"):
            add(aid)
        m.addSeparator()
        add("reset_all")
        add("help")
        m.addAction("Eigenschaften / Hilfe …", self.open_options)
        m.exec(self.view.mapToGlobal(pos.toPoint()))

    # ---------------- Annotationen (Gewinde) ----------------
    def _sync_callouts(self):
        self.view.callouts = (list(self.sel_callouts) + list(self.annot_callouts)
                              + list(getattr(self, "pattern_callouts", [])))
        self.view.update()

    def _callout(self, pos, text, bg=None, fg=None):
        o = self.opt
        if bg is None:
            bg = self._qc(o["text_bg"])
            bg.setAlphaF(float(o["text_bg_alpha"]))
        if fg is None:
            fg = self._qc(o["text_color"])
        return (np.asarray(pos, float), text, bg, fg)

    def _build_annotations(self):
        for it in self.annot_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.annot_items = []
        self.annot_callouts = []
        if self.model is not None and self.a_annot.isChecked():
            L = self.model.size * 0.06
            segs = []
            for h in (self.holes or []):
                if not h.thread or not self._annot_visible_for(h):
                    continue
                p0 = h.entry
                p1 = p0 + h.axis_d * L
                segs += [p0, p1]
                bg = self._annot_bg()
                txt = h.thread.split("x")[0] if "x" in h.thread else h.thread
                if self.annot_types.get("Gewindetiefe"):
                    ti = core.thread_info(self.model, h)
                    if ti:
                        txt = self._thread_short(ti)
                self.annot_callouts.append(self._callout(p1, txt, bg=bg, fg=QtGui.QColor("#000000")))
            if segs:
                it = gl.GLLinePlotItem(pos=np.array(segs), mode="lines", color=(0.25, 0.35, 0.55, 1.0),
                                       width=1.2, antialias=True, glOptions=GL_EDGES)
                self.view.addItem(it)
                self.annot_items.append(it)
            pts = [p for p, i in getattr(self, "entry_pts", [])
                   if self.annot_types.get("Eintritt", True) and (
                       not self.annot_types.get("only_selected", False)
                       or bool({e.ref for e in self.sel if e.kind == "face"} & set(self.holes[i].faces)))]
            if pts:
                it = gl.GLScatterPlotItem(pos=np.array(pts), size=10, color=(0.86, 0.05, 0.05, 1.0), pxMode=True)
                it.setGLOptions(GL_ONTOP if self.a_glass.isChecked() else GL_EDGES)
                self.view.addItem(it)
                self.annot_items.append(it)
        self._sync_callouts()

    def _entry_at(self, p, tol=1e-4):
        """Bohrungsindex, falls p ein Eintrittspunkt ist."""
        for q, i in getattr(self, "entry_pts", []):
            if np.linalg.norm(q - np.asarray(p, float)) < tol:
                return i
        return None

    def _entry_text(self, i):
        h = self.holes[i]
        ei = core.entry_info(self.model, h, self.wcs)
        dtxt = h.thread if h.thread else "Ø" + "/".join(fmt(x) for x in h.diameters)
        t = ei["tool"]
        ytxt = f"Y {fmt(ei['y'])}\n" if abs(ei["y"]) > 1e-4 else ""
        x0txt = f"\nTiefe bis X0 {fmt(ei['depth_x0'])}" if ei["depth_x0"] is not None else ""
        short = (f"Eintritt {dtxt}\nX {fmt(ei['dia'])}  (Ø)\n{ytxt}Z {fmt(ei['z'])}\n"
                 f"C {fmt(ei['c'], 3)}°   B {fmt(ei['b'], 3)}°{x0txt}")
        long = (f"Bohrungseintritt {dtxt} ({', '.join(h.features)}):  X {fmt(ei['dia'])} (Ø, r {fmt(ei['dia'] / 2)})"
                + (f"  Y {fmt(ei['y'])} (außermittig)" if abs(ei["y"]) > 1e-4 else "") +
                f"  Z {fmt(ei['z'])}  C {fmt(ei['c'], 3)}°  B {fmt(ei['b'], 3)}° zur Längsachse   ·   "
                f"Punkt X {fmt(ei['wcs'][0])} Y {fmt(ei['wcs'][1])} Z {fmt(ei['wcs'][2])}   ·   "
                f"Werkzeugrichtung ({fmt(t[0], 4)}, {fmt(t[1], 4)}, {fmt(t[2], 4)})   ·   Tiefe {fmt(h.depth)}"
                + (f"   ·   Spitze bis Mitte X0: {fmt(ei['depth_x0'])} (volle Schneide +{fmt(ei['tip118'])} "
                   f"bei 118°-Bohrer)" if ei["depth_x0"] is not None else ""))
        return short, long

    # ---------------- Ansichtssystem (CAD / CLASSMATE) und gespeicherte Ansichten ----------------
    def _view_frame(self):
        """Zeilen = Achsen des Ansichtssystems in Modellkoordinaten."""
        if self.model is not None and self.view_frame_name != "CAD":
            for name, T in self.model.views:
                if name == self.view_frame_name:
                    return T
        return np.eye(3)

    def set_view_frame(self, name):
        self.view_frame_name = name
        self.settings.setValue("view_frame", name)
        self.status.showMessage(f"Ansichtssystem: {name}")

    def std_view(self, elev, azim):
        e, a = math.radians(elev), math.radians(azim)
        toward = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        if abs(elev) < 89:
            up = np.array([0, 0, 1.0])
        elif elev > 0:
            up = np.array([-math.cos(a), -math.sin(a), 0.0])
        else:
            up = np.array([math.cos(a), math.sin(a), 0.0])
        T = self._view_frame()
        self.view.set_view_dir(T.T @ toward, T.T @ up)
        self.fit_view()
        self._update_silhouette()
        self.view.viewChanged.emit()

    def apply_saved_view(self, v):
        kind = v["kind"]
        if kind == "file":
            T = v["T"]
            self.view.set_view_dir(T[2], T[1])
            self.fit_view()
        elif kind == "cad":
            self.view.set_view_dir(np.array([0, 0, 1.0]), np.array([0, 1.0, 0]))
            self.fit_view()
        else:
            self.view.opts["rotation"] = QtGui.QQuaternion(*v["q"])
            self.view.opts["center"] = QtGui.QVector3D(*v["c"])
            self.view.opts["distance"] = v["dist"]
            self.view.pan_ndc = list(v["pan"])
            self.view.update()
        self._update_silhouette()
        self.view.viewChanged.emit()

    def _all_views(self):
        out = [{"name": "CAD", "kind": "cad"}]
        if self.model is not None:
            out += [{"name": n, "kind": "file", "T": T} for n, T in self.model.views]
        out += self.user_views
        return out

    def save_current_view(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Ansicht speichern", "Name der Ansicht:",
                                                  text=f"Ansicht {len(self.user_views) + 1}")
        if not ok or not name:
            return
        q = self.view.opts["rotation"]
        c = self.view.opts["center"]
        self.user_views.append({"name": name, "kind": "user",
                                "q": (q.scalar(), q.x(), q.y(), q.z()), "c": (c.x(), c.y(), c.z()),
                                "dist": self.view.opts["distance"], "pan": tuple(self.view.pan_ndc)})
        self._fill_view_list()

    def _fill_view_list(self):
        self.view_list.clear()
        for v in self._all_views():
            it = QtWidgets.QListWidgetItem(v["name"])
            it.setData(Qt.ItemDataRole.UserRole, v)
            self.view_list.addItem(it)

    def _build_detail_dock(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        self.view_list = QtWidgets.QListWidget()
        self.view_list.itemActivated.connect(lambda it: self.apply_saved_view(it.data(Qt.ItemDataRole.UserRole)))
        self.view_list.itemClicked.connect(lambda it: self.apply_saved_view(it.data(Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.view_list)
        b = QtWidgets.QPushButton("Aktuelle Ansicht speichern …")
        b.clicked.connect(self.save_current_view)
        lay.addWidget(b)
        b2 = QtWidgets.QPushButton("Mosaikansicht")
        b2.clicked.connect(self.mosaic_view)
        lay.addWidget(b2)
        self.dock_views = QtWidgets.QDockWidget("Detailansichten", self)
        self.dock_views.setObjectName("dock_views")
        self.dock_views.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_views)
        self.dock_views.setVisible(False)

    def toggle_detail_views(self):
        vis = not self.dock_views.isVisible()
        self.dock_views.setVisible(vis)
        if vis:
            self._fill_view_list()

    def mosaic_view(self):
        if self.model is None:
            return
        v = self.view
        keep = (v.opts["rotation"], v.opts["center"], v.opts["distance"], list(v.pan_ndc))
        shots = []
        for title, (e, a) in (("Vorne", (0, -90)), ("Oben", (90, -90)), ("Links", (0, 180)), ("Isometrisch", (30, 45))):
            self.std_view(e, a)
            v.repaint()
            QtWidgets.QApplication.processEvents()
            shots.append((title, v.grabFramebuffer()))
        v.opts["rotation"], v.opts["center"], v.opts["distance"], v.pan_ndc = keep
        v.update()
        self._update_silhouette()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Mosaikansicht")
        g = QtWidgets.QGridLayout(dlg)
        for i, (title, img) in enumerate(shots):
            box = QtWidgets.QGroupBox(title)
            bl = QtWidgets.QVBoxLayout(box)
            lab = QtWidgets.QLabel()
            lab.setPixmap(QtGui.QPixmap.fromImage(img).scaled(560, 380, Qt.AspectRatioMode.KeepAspectRatio,
                                                              Qt.TransformationMode.SmoothTransformation))
            bl.addWidget(lab)
            g.addWidget(box, i // 2, i % 2)
        dlg.exec()

    # ---------------- Rechteckselektion ----------------
    RECT_FILTER_DEFAULTS = {"faces": True, "PLANE": True, "CONE": True, "CYL_CONVEX": True, "CYL_CONCAVE": True,
                            "TORUS": True, "FREEFORM": True, "edges": True, "annot": True, "hidden": True}

    def _rect_filter(self, key):
        return bool(self.rect_filter.get(key, True))

    def _set_rect_filter(self, key, on):
        self.rect_filter[key] = on

    def start_rect_selection(self):
        self.view.rect_mode = True
        self.status.showMessage("Rechteckselektion: Rahmen mit linker Maustaste aufziehen "
                                "(oder jederzeit Alt + linke Maustaste ziehen)")

    def _face_type_key(self, f):
        if f.ftype == "CYLINDER":
            return "CYL_CONCAVE" if f.concavity == "1" else "CYL_CONVEX"
        if f.ftype in ("PLANE", "CONE", "TORUS"):
            return f.ftype
        return "FREEFORM"

    def on_rect_selected(self, rect, mods):
        if self.model is None or self.nav_mode:
            return
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        ents = []
        hidden_ok = self._rect_filter("hidden")
        o, d = self.view.ray(QtCore.QPointF(rect.center()))
        cam = self.view.toward_camera()
        if self._rect_filter("faces"):
            for fi, f in enumerate(self.model.faces):
                if not self._rect_filter(self._face_type_key(f)):
                    continue
                T = f.triangles
                c = T.mean(axis=1)
                S, ok = self.view.to_screen(c)
                inside = ok & (S[:, 0] >= x0) & (S[:, 0] <= x1) & (S[:, 1] >= y0) & (S[:, 1] <= y1)
                if not inside.any():
                    continue
                if not hidden_ok:
                    vis = False
                    for k in np.where(inside)[0][:6]:
                        fh = self._pick_face(QtCore.QPointF(*S[k]))
                        if fh is not None and fh[0] == fi:
                            vis = True
                            break
                    if not vis:
                        continue
                ents.append(core.Entity(self.model, "face", fi))
        if self._rect_filter("edges"):
            for e in self.model.edges.values():
                if e.etype not in ("CIRCLE", "LINE"):
                    continue
                S, ok = self.view.to_screen(e.points)
                if (ok & (S[:, 0] >= x0) & (S[:, 0] <= x1) & (S[:, 1] >= y0) & (S[:, 1] <= y1)).all():
                    if not hidden_ok:
                        mid = e.points[len(e.points) // 2]
                        sm, _ = self.view.to_screen(mid[None])
                        fh = self._pick_face(QtCore.QPointF(*sm[0]))
                        if fh is not None:
                            oo, dd = self.view.ray(QtCore.QPointF(*sm[0]))
                            if (mid - oo) @ dd > fh[2] + self.model.size * 0.004:
                                continue
                    ents.append(core.Entity(self.model, "edge", e.id))
        if self._rect_filter("annot") and self.a_annot.isChecked():
            for h in (self.holes or []):
                if not h.thread:
                    continue
                S, ok = self.view.to_screen((h.entry + h.axis_d * self.model.size * 0.06)[None])
                if ok[0] and x0 <= S[0, 0] <= x1 and y0 <= S[0, 1] <= y1:
                    cyl = [i for i in h.faces if self.model.faces[i].ftype == "CYLINDER"]
                    if cyl:
                        ents.append(core.Entity(self.model, "face", min(cyl, key=lambda i: self.model.faces[i].surf.radius)))
        keys = set()
        uniq = []
        base = self.sel if mods & Qt.KeyboardModifier.ControlModifier else []
        for e in base + ents:
            if e.key() not in keys:
                keys.add(e.key())
                uniq.append(e)
        self.sel = uniq
        self._refresh_selection()
        self.status.showMessage(f"Rechteckselektion: {len(ents)} Element(e) gewählt")

    # ---------------- Gewindetiefen ----------------
    @staticmethod
    def _thread_short(ti):
        t = ti["thread"].replace(".", ",")
        lines = [t, f"Gewinde {fmt(ti['length'])} tief"]
        if ti["t_from"] > 1e-3:
            lines[-1] += f"  (ab {fmt(ti['t_from'])})"
        lines.append(f"Kernloch Ø{fmt(ti['core'])} × {fmt(ti['core_end'])}" if ti["core"] else
                     f"Kernloch {fmt(ti['core_end'])}")
        if ti["blind"]:
            lines.append(f"Bohrspitze {fmt(ti['tip'])}")
        return "\n".join(lines)

    def _thread_rows(self, ti):
        rows = [("Gewinde", f"<b>{ti['thread']}</b>")]
        if ti["pitch"]:
            rows.append(("Steigung P", fmt(ti["pitch"])))
        if ti["core"]:
            rows.append(("Kernloch Ø", f"<b>{fmt(ti['core'])}</b>"))
        rows += [("Gewindetiefe (nutzbar)", f"<b>{fmt(ti['length'])} mm</b>"),
                 ("Gewinde von … bis", f"{fmt(ti['t_from'])} … {fmt(ti['t_to'])}"),
                 ("Kernlochtiefe (zyl.)", f"<b>{fmt(ti['core_end'])} mm</b>")]
        if ti["blind"]:
            rows.append(("Tiefe bis Bohrspitze", f"{fmt(ti['tip'])} mm"))
            ov = ti["over"]
            txt = f"{fmt(ov)} mm" + (f"  (= {fmt(ti['over_p'], 2)} × P)" if ti["over_p"] is not None else "")
            if ov < 1e-3:
                txt += "  – Gewinde bis Kernlochgrund modelliert, Auslauf nach DIN 76 prüfen"
            rows.append(("Kernloch-Überhang", txt))
        rows.append(("Bezug", "Tiefen ab Bohrungseintritt (Oberfläche)"))
        return rows

    def _show_thread_overlay(self, ti):
        pt = ti["pt"]
        d0 = pt(0.0)
        axv = pt(1.0) - d0
        perp = core.any_perp(axv)
        k = (ti["major"] or ti["core"] or 1.0) * 0.75
        items = []

        def seg(a, b, col, wdt):
            it = gl.GLLinePlotItem(pos=np.array([pt(a), pt(b)]), mode="lines", color=col, width=wdt,
                                   antialias=True, glOptions=GL_ONTOP)
            it.setDepthValue(20)
            items.append(it)

        if ti["t_from"] > 1e-3:
            seg(0.0, ti["t_from"], (0.45, 0.45, 0.45, 1.0), 1.5)
        seg(ti["t_from"], ti["t_to"], (0.95, 0.55, 0.05, 1.0), 5.0)          # Gewinde
        if ti["core_end"] - ti["t_to"] > 1e-3:
            seg(ti["t_to"], ti["core_end"], (0.15, 0.35, 0.85, 1.0), 4.0)     # Kernloch-Überhang
        if ti["tip"] - ti["core_end"] > 1e-3:
            seg(ti["core_end"], ti["tip"], (0.45, 0.45, 0.45, 1.0), 1.5)      # Bohrspitze
        ticks = []
        for t in (0.0, ti["t_from"], ti["t_to"], ti["core_end"], ti["tip"]):
            c = pt(t)
            ticks += [c - k * perp, c + k * perp]
        it = gl.GLLinePlotItem(pos=np.array(ticks), mode="lines", color=(0.1, 0.1, 0.1, 1.0), width=1.5,
                               antialias=True, glOptions=GL_ONTOP)
        it.setDepthValue(20)
        items.append(it)
        for it in items:
            self.view.addItem(it)
        self.items_overlay = list(self.items_overlay) + items
        self.sel_callouts = [self._callout(pt(0.0), self._thread_short(ti),
                                           bg=self._annot_bg(), fg=QtGui.QColor("#000000"))]
        self._sync_callouts()
        self.status.showMessage(f"{ti['thread']}: Gewinde {fmt(ti['length'])} tief (ab {fmt(ti['t_from'])}), "
                                f"Kernloch Ø{fmt(ti['core'])} {fmt(ti['core_end'])} tief, Spitze {fmt(ti['tip'])}  "
                                "– orange = Gewinde, blau = Kernloch-Überhang, grau = Senkung/Bohrspitze")

    def _thread_for_entity(self, e):
        if self.model is None or e.kind != "face":
            return None
        h = core.hole_for_entity(self.model, e)
        return core.thread_info(self.model, h) if h is not None else None

    # ---------------- Annotationen nach Typ ----------------
    def _annot_visible_for(self, h):
        if not self.annot_types.get("Gewinde", True):
            return False
        if self.annot_types.get("only_selected", False):
            fs = {e.ref for e in self.sel if e.kind == "face"}
            return bool(fs & set(h.faces))
        return True

    def _set_annot_type(self, key, on):
        self.annot_types[key] = on
        self._build_annotations()

    # ---------------- Kasten (Außenmaße) ----------------
    def _build_box_items(self):
        for it in getattr(self, "box_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.box_items = []
        if self.model is None or not self.a_box.isChecked():
            return
        m, w = self.model, self.wcs
        V = m.tris.reshape(-1, 3)
        P = (V - w.origin) @ w.R.T
        lo, hi = P.min(axis=0), P.max(axis=0)
        size = hi - lo

        def to_model(q):
            return w.origin + np.asarray(q, float) @ w.R

        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        segs = []
        for a, b in edges:
            segs += [corners[a], corners[b]]
        segs = to_model(np.array(segs))
        ln = gl.GLLinePlotItem(pos=segs, mode="lines", color=(0.15, 0.15, 0.15, 0.9), width=1.4,
                               antialias=True, glOptions=GL_ONTOP)
        items = [ln]
        font = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Weight.Bold)
        col = QtGui.QColor("#202020")
        center = to_model((lo + hi) / 2)
        cands = {"X": [(0, 4), (1, 5), (2, 6), (3, 7)], "Y": [(0, 2), (1, 3), (4, 6), (5, 7)],
                 "Z": [(0, 1), (2, 3), (4, 5), (6, 7)]}
        csc, _ = self.view.to_screen(center[None])
        for ax in "XYZ":
            # außen liegende Kante (am weitesten von der Bildmitte des Kastens) beschriften
            mids = [to_model((corners[a] + corners[b]) / 2) for a, b in cands[ax]]
            sc, _ = self.view.to_screen(np.array(mids))
            mid = mids[int(np.argmax(np.linalg.norm(sc - csc[0], axis=1)))]
            lab = {"X": "X (Höhe)", "Y": "Y (Breite)", "Z": "Z (Länge)"}[ax]
            items.append(gl.GLTextItem(pos=mid, text=f"{lab} {fmt(size['XYZ'.index(ax)])}", color=col, font=font,
                                       alignment=self._align_for(mid, center)))
        for it in items:
            self.view.addItem(it)
        self.box_items = items
        self.status.showMessage(f"Außenmaße (Nullpunkt-KS):  X {fmt(size[0])}  ×  Y {fmt(size[1])}  ×  Z {fmt(size[2])} mm"
                                f"   |   Ø max {fmt(2 * max(np.hypot(P[:, 0], P[:, 1])))} mm")

    # ---------------- Schnitt ----------------
    SECTION_NAMES = {"long": "Längsschnitt", "cross": "Querschnitt", "free": "Schnitt (gedreht)"}

    def _model_center(self):
        V = self.model.tris.reshape(-1, 3)
        return (V.min(axis=0) + V.max(axis=0)) / 2

    def _section_plane(self):
        s = self.section
        n, dd = s["n"], s["d"]
        if self.section_flip:
            return -n, -dd
        return n, dd

    def _section_label(self):
        s = self.section
        n = s["n"]
        for k in range(3):
            c = float(n @ self.wcs.R[k])
            if abs(abs(c) - 1) < 1e-6:
                off = (s["d"] - n @ self.wcs.origin) * (1 if c > 0 else -1)
                return f"{self.SECTION_NAMES.get(s['kind'], 'Schnitt')}  {'XYZ'[k]} = {fmt(off)}"
        cs = [abs(float(n @ self.wcs.R[k])) for k in range(3)]
        k = int(np.argmax(cs))
        ang = math.degrees(math.acos(min(1.0, cs[k])))
        return f"Schnitt gekippt {fmt(ang, 2)}° (zur {'XYZ'[k]}-Ebene)"

    def cycle_section(self):
        if self.model is None:
            return
        cur = None if self.section is None else self.section["kind"]
        if cur is None:
            n = self.wcs.R[1].copy()
            self.section = {"kind": "long", "n": n, "d": float(n @ self.wcs.origin)}
        elif cur == "long":
            n = self.wcs.R[2].copy()
            c = self._model_center()
            self.section = {"kind": "cross", "n": n, "d": round(float(n @ c), 1)}
        else:
            self.section = None
        self._apply_section()

    def flip_section(self):
        if self.section is None:
            return
        self.section_flip = not self.section_flip
        self._apply_section()

    def move_section(self, delta):
        if self.section is None:
            return
        n = self.section["n"]
        V = self.model.tris.reshape(-1, 3)
        c = V @ n
        self.section["d"] = float(np.clip(self.section["d"] + delta, c.min() - 1, c.max() + 1))
        self._apply_section()

    def rotate_section(self, dx, dy):
        if self.section is None:
            return
        s = self.section
        n = s["n"]
        c = self._model_center()
        p0 = c - (n @ c - s["d"]) * n
        r, u = self.view.screen_axes()

        def rot(v, axis, deg):
            a = math.radians(deg)
            axis = axis / np.linalg.norm(axis)
            return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * (axis @ v) * (1 - math.cos(a))

        n = rot(n, u, dx * 0.4)
        n = rot(n, r, dy * 0.4)
        n /= np.linalg.norm(n)
        s.update(n=n, d=float(n @ p0), kind="free")
        self._apply_section()

    def tilt_section(self, about, deg):
        """Schnittebene um eine Bildschirmachse kippen ('h' = waagrechte, 'v' = senkrechte Achse)."""
        if self.section is None:
            return
        s = self.section
        n = s["n"]
        c = self._model_center()
        p0 = c - (n @ c - s["d"]) * n                 # Drehpunkt: Ebenenpunkt nahe Bauteilmitte
        r, u = self.view.screen_axes()
        # Kippachse in der Schnittebene (nächstliegend zur Bildschirmrichtung) -> genau deg Grad je Schritt
        axis = r if about == "h" else u
        axis = axis - (axis @ n) * n
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(n, u if about == "h" else r)
        axis = axis / np.linalg.norm(axis)
        a = math.radians(deg)
        n = n * math.cos(a) + np.cross(axis, n) * math.sin(a) + axis * (axis @ n) * (1 - math.cos(a))
        n /= np.linalg.norm(n)
        s.update(n=n, d=float(n @ p0), kind="free")
        self._apply_section()
        self.status.showMessage(self._section_label())

    def _on_section_drag(self, mode, dx, dy):
        if self.section is None:
            return
        if mode == "rot":
            self.rotate_section(dx, dy)
        else:
            n = self.section["n"] * (-1 if self.section_flip else 1)
            r, u = self.view.screen_axes()
            nr, nu = n @ r, n @ u
            L = math.hypot(nr, nu)
            wpp = self.view.world_per_px()
            if L < 0.2:
                delta = -dy * wpp
            else:
                delta = (dx * nr - dy * nu) / L * wpp
            self.move_section(delta * (-1 if self.section_flip else 1))

    def _wheel_hook(self, ev):
        mods = ev.modifiers()
        if (mods & Qt.KeyboardModifier.AltModifier and not mods & Qt.KeyboardModifier.ShiftModifier
                and not mods & Qt.KeyboardModifier.ControlModifier and self.model is not None):
            # Alt + Mausrad: Bauteil um seine Mittelachse drehen (5° je Raste)
            d = ev.angleDelta().y() or ev.angleDelta().x()
            step = 5.0
            self.spin_main_axis(d / 120.0 * step)
            return True
        return self._section_wheel(ev)

    def spin_main_axis(self, deg):
        mo, md = core.main_axis(self.model)
        v = self.view
        c = v.opts["center"]
        c = np.array([c.x(), c.y(), c.z()])
        q = c - mo
        c_ax = mo + (q @ md) * md                        # Drehpunkt auf die Mittelachse legen
        v.opts["center"] = QtGui.QVector3D(*map(float, c_ax))
        v.opts["rotation"] = v.opts["rotation"] * QtGui.QQuaternion.fromAxisAndAngle(
            QtGui.QVector3D(*map(float, md)), float(deg))
        v.update()
        self.status.showMessage(f"Drehung um die Mittelachse: {fmt(deg, 1)}° (Alt + Mausrad)")
        v.viewMoving.emit()
        QtCore.QTimer.singleShot(150, v.viewChanged.emit)

    def _section_wheel(self, ev):
        if self.section is None or not ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        d = ev.angleDelta().y() or ev.angleDelta().x()
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.tilt_section("h", d / 120.0 * 1.0)       # Strg+Umschalt+Mausrad: kippen (1° je Raste)
            return True
        step = 0.1 if ev.modifiers() & Qt.KeyboardModifier.AltModifier else 1.0   # + Alt = fein
        self.move_section(d / 120.0 * step)
        return True

    def _section_key(self, ev):
        if self.section is None:
            return False
        k = ev.key()
        if k in (Qt.Key.Key_Up, Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_PageUp,
                 Qt.Key.Key_PageDown):
            mods = ev.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and k not in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                # Strg + Pfeiltasten: Schnittebene kippen (1°, mit Umschalt 5°)
                deg = 5.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
                if k in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                    self.tilt_section("h", deg if k == Qt.Key.Key_Up else -deg)
                else:
                    self.tilt_section("v", deg if k == Qt.Key.Key_Right else -deg)
                return True
            step = 10.0 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            if k in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                step = 10.0
            sgn = 1 if k in (Qt.Key.Key_Up, Qt.Key.Key_Right, Qt.Key.Key_PageUp) else -1
            self.move_section(sgn * step)
            return True
        return False

    def _apply_section(self):
        for it in self.section_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.section_items = []
        if self.model is None or self.mesh is None:
            return
        if self.section is None:
            set_clip_uniform([0, 0, 1, 0, 0])
            self.edge_item.setData(pos=self.edge_seg)
            self._update_lock_badge()
            self.view.update()
            return
        n, dd = self._section_plane()
        set_clip_uniform([n[0], n[1], n[2], dd, 1.0])
        # Kanten auf der weggeschnittenen Seite ausblenden (auf die Ebene kürzen)
        seg = self.edge_seg.reshape(-1, 2, 3)
        sd = seg @ n - dd
        keep = (sd[:, 0] <= 0) & (sd[:, 1] <= 0)
        cross = (sd[:, 0] <= 0) != (sd[:, 1] <= 0)
        cs = seg[cross].copy()
        csd = sd[cross]
        t = csd[:, 0] / (csd[:, 0] - csd[:, 1])
        ip = cs[:, 0] + (cs[:, 1] - cs[:, 0]) * t[:, None]
        out_first = csd[:, 0] > 0
        cs[out_first, 0] = ip[out_first]
        cs[~out_first, 1] = ip[~out_first]
        newseg = np.concatenate([seg[keep], cs]).reshape(-1, 3)
        self.edge_item.setData(pos=newseg if len(newseg) else np.zeros((0, 3)))
        # Schnittkontur
        P = self.rtris
        sv = P @ n - dd
        pos = (sv > 0).sum(axis=1)
        ct = (pos == 1) | (pos == 2)
        pts = []
        if ct.any():
            Pc, sc = P[ct], sv[ct]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                m_ = (sc[:, a] > 0) != (sc[:, b] > 0)
                tt = np.zeros(len(sc))
                den = sc[m_, a] - sc[m_, b]
                tt[m_] = sc[m_, a] / den
                q = Pc[:, a] + (Pc[:, b] - Pc[:, a]) * tt[:, None]
                q[~m_] = np.nan
                pts.append(q)
            Q = np.stack(pts, axis=1)          # (k,3,3) – je Dreieck genau 2 gültige Punkte
            ok = ~np.isnan(Q[:, :, 0])
            two = ok.sum(axis=1) == 2
            Q, ok = Q[two], ok[two]
            seg2 = Q[ok].reshape(-1, 2, 3).reshape(-1, 3)
            it = gl.GLLinePlotItem(pos=seg2, mode="lines", color=(0.35, 0.02, 0.02, 1.0), width=2.2,
                                   antialias=True, glOptions=GL_EDGES)
            self.view.addItem(it)
            self.section_items.append(it)
        txt = self._section_label() + ("  (Seite gewechselt)" if self.section_flip else "")
        self.status.showMessage(txt + "   – Umschalt+Mausrad/Pfeile: verschieben, Umschalt+linke Maus: drehen, "
                                      "Umschalt+rechte Maus: schieben, S = weiter, Umschalt+S = Seite")
        self._update_lock_badge()
        self._update_silhouette()
        self.view.update()

    # ---------------- Bohrbild ----------------
    def _arrow_item(self, tip, d, length):
        """Kleiner Kegel-Pfeil, Spitze bei tip, zeigt entgegen d (d = vom Pfeil weg nach außen)."""
        md = gl.MeshData.cylinder(rows=1, cols=16, radius=[0.0, length * 0.32], length=length)
        P = md.vertexes()
        a = np.cross([0, 0, 1.0], d)
        if np.linalg.norm(a) < 1e-9:
            R = np.eye(3) if d[2] > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            a /= np.linalg.norm(a)
            c = float(np.clip(d[2], -1, 1))
            sn = math.sqrt(max(0.0, 1 - c * c))
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = np.eye(3) + sn * K + (1 - c) * K @ K
        V = P @ R.T + np.asarray(tip, float)
        md2 = gl.MeshData(vertexes=V.astype(np.float32), faces=md.faces())
        it = gl.GLMeshItem(meshdata=md2, color=(0.55, 0.55, 0.6, 1.0), smooth=False, shader="shaded",
                           glOptions=GL_ONTOP)
        it.setDepthValue(21)
        return it

    def _build_pattern_items(self):
        for it in getattr(self, "pattern_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.pattern_items = []
        self.pattern_callouts = []
        self._sync_callouts()
        if self.model is None or not self.a_pattern.isChecked():
            return
        m = self.model
        groups, (o, d, e1, e2) = core.hole_pattern(m, self.wcs)
        z_ok = abs(self.wcs.R[2] @ d) > 0.999
        if not groups:
            self.status.showMessage("Keine Bohrungen um die Mittelachse gefunden.")
            return
        # gegen den Uhrzeigersinn so, wie man gerade auf das Teil schaut
        q = self.view.opts["rotation"]
        tc = q.conjugated().rotatedVector(QtGui.QVector3D(0, 0, 1))
        if np.array([tc.x(), tc.y(), tc.z()]) @ d < 0:
            e2 = -e2
            for g in groups:
                g.angles = [round(-a, 4) % 360.0 for a in g.angles]
        V = m.tris.reshape(-1, 3)
        w = V - o
        R_out = float(np.linalg.norm(w - np.outer(w @ d, d), axis=1).max())
        R_lab = R_out * 1.15 + m.size * 0.03
        col_line = (0.10, 0.30, 0.75, 1.0)
        col_ref = (0.45, 0.45, 0.45, 1.0)
        f_big = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Weight.Bold)
        f_small = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.DemiBold)
        txtcol = QtGui.QColor("#0d2f8a")
        items = []

        def dirv(th):
            t = math.radians(th)
            return math.cos(t) * e1 + math.sin(t) * e2

        def line(pts, col=col_line, width=1.6, mode="lines"):
            it = gl.GLLinePlotItem(pos=np.array(pts), color=col, width=width, antialias=True,
                                   mode=mode, glOptions=GL_ONTOP)
            items.append(it)

        def text(p, t, font=f_big, color=txtcol, center=None):
            kw = {}
            if center is not None:
                kw["alignment"] = self._align_for(np.asarray(p, float), center)
            items.append(gl.GLTextItem(pos=np.asarray(p, float), text=t, color=color, font=font, **kw))

        holes = m._holes
        summary = []
        radial_done = set()
        ref_name = core.frame_name(e1, self.wcs)
        for g in groups:
            c = o + g.level * d
            def dtxt_of(h):
                hh = holes[h]
                return hh.thread if hh.thread else "Ø" + "/".join(fmt(x) for x in hh.diameters)
            dtxt = dtxt_of(g.holes[0])
            if g.kind == "axial":
                t = np.linspace(0, 2 * math.pi, 145)
                circ = [c + g.radius * (math.cos(a) * e1 + math.sin(a) * e2) for a in t]
                line(circ, width=1.4, mode="line_strip")
                for th in g.angles:
                    u = dirv(th)
                    line([c + g.radius * u, c + R_lab * u], width=1.2)
                    text(c + (R_lab + m.size * 0.01) * u, f"{fmt(th, 2)}°", center=c)
                # Gruppen-Beschriftung in der größten Lücke
                a = sorted(g.angles)
                gaps = [((a[(k + 1) % len(a)] - a[k]) % 360 or 360, a[k]) for k in range(len(a))]
                gap, start = max(gaps)
                mid = start + gap / 2
                info = f"LK Ø{fmt(2 * g.radius)} · {len(a)}× {dtxt}"
                pt = core.pitch_text(a)
                text(c + (R_lab * 1.12) * dirv(mid), info + (f" · {pt}" if pt else ""), f_small, center=c)
                summary.append(f"Stirnseite x={fmt(g.level)}: {info}, {pt or 'Einzelbohrung'} ("
                               + ", ".join(fmt(x, 2) + "°" for x in a) + ")")
            else:
                # Mantelbohrungen: kurzer Hinweisstrich mit Pfeil von der Bohrungsmitte nach außen
                # und Beschriftungsfeld
                L = m.size * 0.06
                offs = g.offsets or [0.0] * len(g.holes)
                for h, th, yo in zip(g.holes, g.angles, offs):
                    hh = holes[h]
                    ax = np.asarray(hh.axis_d, float)
                    ax = ax / (np.linalg.norm(ax) or 1.0)
                    p0 = np.asarray(hh.entry, float)
                    p1 = p0 + L * ax
                    line([p0, p1], col=(0.25, 0.35, 0.55, 1.0), width=1.2)
                    items.append(self._arrow_item(p0, ax, m.size * 0.012))
                    tl = f"  ∠{fmt(g.tilt, 2)}°" if g.kind == "inclined" else ""
                    if abs(yo) > 1e-4:
                        tl += f"  Y {fmt(yo)}"
                    self.pattern_callouts.append(self._callout(
                        p1, f"{fmt(th, 2)}°  {dtxt_of(h)}{tl}", bg=self._annot_bg(), fg=QtGui.QColor("#000000")))
                if g.kind == "inclined":
                    summary.append(f"Schräg ∠{fmt(g.tilt, 2)}° bei {fmt(g.level)} mm, Ø{fmt(2 * g.radius)}: "
                                   + ", ".join(fmt(x, 2) + "°" for x in g.angles))
                else:
                    summary.append(f"Radial bei {fmt(g.level)} mm: " + ", ".join(
                        fmt(x, 2) + "°" + (f" (Y {fmt(yo)})" if abs(yo) > 1e-4 else "")
                        for x, yo in zip(g.angles, g.offsets or [0.0] * len(g.angles))))
                continue
            # 0°-Bezug
            line([c, c + R_lab * 0.98 * e1], col=col_ref, width=1.0)
        for it in items:
            self.view.addItem(it)
        self.pattern_items = items
        self.pattern_summary = summary
        self._sync_callouts()
        if not getattr(self, "_pattern_msg_quiet", False):
            pass
        self.status.showMessage("Bohrbild: " + " | ".join(summary) + "   (0° = " + ref_name +
                                ", gegen den Uhrzeigersinn in der aktuellen Blickrichtung)"
                                + ("" if z_ok else "   ⚠ Nullpunkt-Z liegt nicht auf der Längsachse – für C0° "
                                   "„Drehmaschine“ und dann „C0° auf Auswahl legen“ verwenden"))

    # ---------------- Picking ----------------
    def _pick_face(self, pos):
        o, d = self.view.ray(pos)
        if o is None:
            return None
        hit, t = core.ray_hits(self.rtris.reshape(-1, 3, 3), o, d)
        if not hit.any():
            return None
        idx = np.where(hit)[0]
        if self.section is not None:
            P = o + np.outer(t[idx], d)
            n, dd = self._section_plane()
            idx = idx[P @ n <= dd + 1e-6]
            if len(idx) == 0:
                return None
        idx = idx[np.argsort(t[idx])]
        k = idx[0]
        if self.a_glass.isChecked():
            # im Glas-Modus: erste getroffene Innenfläche (Bohrung/Kanal) bevorzugen
            for j in idx:
                if self.model.faces[int(self.rface[j])].concavity == "1":
                    k = j
                    break
        return int(self.rface[k]), o + d * t[k], float(t[k]), o, d

    def _pick_vertex(self, pos, face_hit, radius_px=9):
        if self.model is None or len(getattr(self, "vertices", [])) == 0:
            return None
        S, ok = self.view.to_screen(self.vertices)
        dist = np.linalg.norm(S - np.array([pos.x(), pos.y()]), axis=1)
        dist[~ok] = 1e9
        cand = np.where(dist < radius_px)[0]
        if len(cand) == 0:
            return None
        o, d = self.view.ray(pos)
        P = self.vertices[cand]
        depth = (P - o) @ d
        if face_hit is not None:
            vis = depth <= face_hit[2] + self.model.size * 0.004
            if not vis.any():
                return None
            cand, P, depth = cand[vis], P[vis], depth[vis]
        k = int(np.argmin(dist[cand] + 1e-6 * depth))
        return P[k].copy()

    def _pick_edge(self, pos, face_hit):
        if len(self.edge_a) == 0:
            return None
        A, okA = self.view.to_screen(self.edge_a)
        B, okB = self.view.to_screen(self.edge_b)
        p = np.array([pos.x(), pos.y()])
        AB = B - A
        L2 = (AB ** 2).sum(axis=1).clip(1e-12)
        s = (((p - A) * AB).sum(axis=1) / L2).clip(0, 1)
        Q = A + AB * s[:, None]
        dist = np.linalg.norm(Q - p, axis=1)
        dist[~(okA & okB)] = 1e9
        cand = np.where(dist < 7)[0]
        if len(cand) == 0:
            return None
        o, d = self.view.ray(pos)
        P3 = self.edge_a[cand] + (self.edge_b[cand] - self.edge_a[cand]) * s[cand, None]
        depth = (P3 - o) @ d
        if face_hit is not None:
            tol = self.model.size * 0.004
            vis = depth <= face_hit[2] + tol
            if not vis.any():
                return None
            cand, depth, P3 = cand[vis], depth[vis], P3[vis]
            dist_c = dist[cand]
        else:
            dist_c = dist[cand]
        k = np.argmin(dist_c + 1e-6 * depth)
        return str(self.edge_ids[cand[k]]), P3[k]

    def on_click(self, pos, mods):
        if self.model is None or self.nav_mode:
            return
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        lines = self.a_lines.isChecked()
        if lines:
            # Linien-Modus: nur Eckpunkte und Kanten – Flächen werden nie gewählt
            occ = None if self.show_hidden else self._pick_face(pos)
            vh = self._pick_vertex(pos, occ)
            eh = None if vh is not None else self._pick_edge(pos, occ)
            if vh is None and eh is None:
                return
            if vh is not None:
                ent = core.Entity(self.model, "point", None, point=vh)
                ent.is_vertex = True
                pw = self.wcs.to_wcs(vh)
                ei = self._entry_at(vh)
                self.status.showMessage(self._entry_text(ei)[1] if ei is not None else
                                        f"Eckpunkt:  X {fmt(pw[0])}   Y {fmt(pw[1])}   Z {fmt(pw[2])}")
            else:
                ent = core.Entity(self.model, "edge", eh[0])
            keys = [e.key() for e in self.sel]
            if ent.key() in keys:
                self.sel.pop(keys.index(ent.key()))
            elif ctrl or len(self.sel) < 2:
                self.sel.append(ent)
            else:
                self.sel = [ent]
            self._clear_hover()
            self._refresh_selection()
            return
        fh = self._pick_face(pos)
        if shift:
            if fh is None:
                return
            ent = core.Entity(self.model, "point", None, point=fh[1])
            if len(self.sel) >= 2 and not ctrl:
                self.sel = []
            self.sel.append(ent)
            self._refresh_selection()
            return
        glass = self.a_glass.isChecked() and not self.a_lines.isChecked()
        lines = self.a_lines.isChecked()
        fh_occ = None if lines else fh          # im Linien-Modus verdeckt nichts
        vh = None if glass else self._pick_vertex(pos, fh_occ)
        eh = None if (vh is not None or glass) else self._pick_edge(pos, fh_occ)
        if vh is not None:
            ent = core.Entity(self.model, "point", None, point=vh)
            ent.is_vertex = True
            pw = self.wcs.to_wcs(vh)
        elif eh is not None:
            ent = core.Entity(self.model, "edge", eh[0])
        elif fh is not None:
            ent = core.Entity(self.model, "face", fh[0])
        else:
            ent = None
        if vh is not None:
            ei = self._entry_at(vh)
            self.status.showMessage(self._entry_text(ei)[1] if ei is not None else
                                    f"Eckpunkt:  X {fmt(pw[0])}   Y {fmt(pw[1])}   Z {fmt(pw[2])}")
        elif fh is not None:
            pw = self.wcs.to_wcs(fh[1])
            self.status.showMessage(f"Punkt (Nullpunkt-KS): X {fmt(pw[0])}  Y {fmt(pw[1])}  Z {fmt(pw[2])}")
        keys = [e.key() for e in self.sel]
        if ent is None:
            if not ctrl:
                self.sel = []
        elif ent.key() in keys:
            self.sel.pop(keys.index(ent.key()))          # nochmal anklicken = abwählen
        elif ctrl or len(self.sel) < 2:
            self.sel.append(ent)                         # 1. und 2. Element
        else:
            self.sel = [ent]                             # 3. Klick startet neue Messung
        self._refresh_selection()

    # ---------------- Dynamisches Hervorheben (Hover) ----------------
    def _on_hover(self, pos):
        active = (self.model is not None and not self.nav_mode
                  and (self.a_lines.isChecked() or self.opt.get("hover_on", True)))
        if not active:
            if self._hover_key is not None:
                self._clear_hover()
            return
        self._hover_pos = pos
        if not self._hover_timer.isActive():
            self._hover_timer.start(25)

    def _clear_hover(self):
        for it in getattr(self, "hover_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.hover_items = []
        self._hover_key = None
        self.view.unsetCursor()

    def _do_hover(self):
        pos = self._hover_pos
        if pos is None or self.model is None or self.nav_mode:
            return
        lines = self.a_lines.isChecked()
        glass = self.a_glass.isChecked() and not lines
        fh = self._pick_face(pos)
        occ = None if (lines and self.show_hidden) else fh
        vh = None if glass else self._pick_vertex(pos, occ)
        eh = None if (vh is not None or glass) else self._pick_edge(pos, occ)
        fi = None
        if vh is None and eh is None and not lines and fh is not None:
            fi = fh[0]
        if vh is not None:
            key = ("v", tuple(np.round(vh, 4)))
        elif eh is not None:
            key = ("e", eh[0])
        elif fi is not None:
            key = ("f", fi)
        else:
            key = None
        if key == self._hover_key:
            return
        self._clear_hover()
        if key is None:
            return
        self._hover_key = key
        sel_keys = {e.key() for e in self.sel}
        is_sel = (("face", fi) in sel_keys) or (eh is not None and ("edge", eh[0]) in sel_keys)
        col = self._rgba(self.opt["hover_sel_color"] if is_sel else self.opt["hover_color"])
        line_col = (col[0], col[1], col[2], 1.0)
        if vh is not None:
            it = gl.GLScatterPlotItem(pos=np.array([vh]), size=14, color=line_col, pxMode=True, glOptions=GL_ONTOP)
            pw = self.wcs.to_wcs(vh)
            tip = f"Eckpunkt  X {fmt(pw[0])}  Y {fmt(pw[1])}  Z {fmt(pw[2])}"
        elif eh is not None:
            e = self.model.edges[eh[0]]
            pts = np.repeat(e.points, 2, axis=0)[1:-1]
            it = gl.GLLinePlotItem(pos=pts, mode="lines", color=line_col,
                                   width=max(3.0, 2.0 * float(self.opt["hl_width"])), antialias=True,
                                   glOptions=GL_ONTOP)
            tip = f"{e.id}: " + (f"Kreis Ø {fmt(2 * e.radius)}" if e.etype == "CIRCLE" and e.radius
                                 else f"Gerade L {fmt(e.length)}")
        else:
            mask = self.rface == fi
            tri = self.rtris[mask]
            verts = tri.reshape(-1, 3).astype(np.float32)
            md = gl.MeshData(vertexes=verts, faces=np.arange(len(verts)).reshape(-1, 3),
                             vertexColors=np.tile(np.array(col, np.float32), (len(verts), 1)))
            md._vertexNormals = self.normals.reshape(-1, 3, 3)[mask].reshape(-1, 3)
            it = gl.GLMeshItem(meshdata=md, smooth=True, shader="s3d_cut", glOptions=GL_HOVER)
            f = self.model.faces[fi]
            tip = f"{f.id}: {core.FACE_TYPES_DE.get(f.ftype, f.ftype)}"
            d = core.Entity(self.model, "face", fi).diameter()
            if d:
                tip += f"  Ø {fmt(d)}"
        it.setDepthValue(30)
        self.view.addItem(it)
        self.hover_items = [it]
        if vh is not None or eh is not None:
            self.view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status.showMessage(tip)

    def on_dclick(self, pos):
        if self.model is None:
            return
        fh = self._pick_face(pos)
        if fh is not None:
            self.view.opts["center"] = QtGui.QVector3D(*fh[1])
            self.view.center_on_lock()
            self.view.update()
            self.status.showMessage("Drehpunkt gesetzt")

    def clear_selection(self):
        self.sel = []
        self._refresh_selection()

    # ---------------- Auswahl anzeigen ----------------
    def _clear_overlay(self):
        for it in self.items_overlay:
            self.view.removeItem(it)
        self.items_overlay = []
        self.sel_callouts = []
        self._sync_callouts()
        for it in getattr(self, "angle_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.angle_items = []

    def _refresh_selection(self):
        if self.model is None or self.mesh is None:
            return
        cols = self.base_colors.copy()
        edge_pts, pts = [], []
        for e in self.sel:
            if e.kind == "face":
                cols[self.vert_face == e.ref] = SEL_COLOR
            elif e.kind == "edge":
                p = e.edge.points
                edge_pts.append(np.repeat(p, 2, axis=0)[1:-1])
                if e.edge.etype == "CIRCLE" and e.edge.center is not None:
                    pts.append(e.edge.center)
            else:
                pts.append(e.point)
        self.meshdata.setVertexColors(cols)
        self.meshdata._vertexNormals = self.normals
        self.mesh.meshDataChanged()
        self._build_glass_highlight()
        self.sel_edges.setData(pos=np.concatenate(edge_pts) if edge_pts else np.zeros((0, 3)))
        self.sel_pts.setData(pos=np.array(pts) if pts else np.zeros((0, 3)))
        self._clear_overlay()
        if self.annot_types.get("only_selected"):
            self._build_annotations()
        self._show_selection_info()

    def _hole_faces(self, fi):
        for h in (self.holes or getattr(self.model, "_holes", []) or []):
            if fi in h.faces:
                return list(h.faces)
        return [fi]

    def _build_glass_highlight(self):
        if self.hi_mesh is not None:
            try:
                self.view.removeItem(self.hi_mesh)
            except Exception:
                pass
            self.hi_mesh = None
        if not (self.a_glass.isChecked() or self.a_lines.isChecked()) or self.model is None:
            return
        fcol = {}
        k = 0
        for e in self.sel:
            if e.kind != "face":
                continue
            col = GLASS_PALETTE[k % len(GLASS_PALETTE)]
            k += 1
            for fi in self._hole_faces(e.ref):
                fcol.setdefault(fi, col)
        if not fcol:
            return
        mask = np.isin(self.rface, list(fcol))
        tri = self.rtris[mask]
        verts = tri.reshape(-1, 3).astype(np.float32)
        alpha = 0.35 if self.a_lines.isChecked() else 1.0
        cols = np.repeat(np.array([[*fcol[int(f)], alpha] for f in self.rface[mask]], np.float32), 3, axis=0)
        nrm = self.normals.reshape(-1, 3, 3)[mask].reshape(-1, 3)
        md = gl.MeshData(vertexes=verts, faces=np.arange(len(verts)).reshape(-1, 3), vertexColors=cols)
        md._vertexNormals = nrm
        self.hi_mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader="s3d_cut",
                                     glOptions=GL_GLASS if self.a_lines.isChecked() else GL_SOLID)
        self.hi_mesh.setDepthValue(-5)
        self.view.addItem(self.hi_mesh)

    def _update_silhouette(self):
        if self.model is None or not self.a_lines.isChecked() or not hasattr(self, "sil_seg"):
            return
        q = self.view.opts["rotation"]
        tc = q.conjugated().rotatedVector(QtGui.QVector3D(0, 0, 1))
        t = np.array([tc.x(), tc.y(), tc.z()])
        f = self.tri_n @ t
        m = (f[self.sil_a] > 0) != (f[self.sil_b] > 0)
        seg = self.sil_seg[m]
        if self.section is not None and len(seg):
            n, dd = self._section_plane()
            seg = seg[((seg @ n) <= dd + 1e-6).all(axis=1)]
        self.sil_item.setData(pos=seg.reshape(-1, 3) if len(seg) else np.zeros((0, 3)))

    def _sil_tick(self):
        if self.a_lines.isChecked() and not self._sil_timer.isActive():
            self._sil_timer.start(30)

    def toggle_hidden_lines(self):
        self.show_hidden = not self.show_hidden
        self.settings.setValue("show_hidden", int(self.show_hidden))
        self.update_display()
        self.status.showMessage("Verdeckte Linien werden angezeigt" if self.show_hidden
                                else "Nur sichtbare Linien (verdeckte ausgeblendet)")

    def on_lines(self):
        if self.model is None:
            return
        if not self.a_lines.isChecked():
            self._clear_hover()
        self._build_glass_highlight()
        self.update_display()
        self.status.showMessage("Nur Linien: Linie/Eckpunkt unter der Maus wird rot, Klick rastet ein. Flächen werden hier nie gewählt (Y = zurück)"
                                if self.a_lines.isChecked() else "Normale Darstellung")

    def on_glass(self):
        if self.model is None:
            return
        self._build_glass_highlight()
        self.update_display()
        self.status.showMessage("Glas-Modus: Bohrung/Kanal anklicken = farbig, Strg+Klick = weitere Kanäle (eigene Farbe)"
                                if self.a_glass.isChecked() else "Normale Darstellung")

    def _axis_lines(self, ents, seg=None):
        """Mittellinien (Strich-Punkt) der gewählten Drehflächen, damit die Bohrungsmitte sichtbar ist."""
        out = []
        for e in ents:
            if e.kind != "face" or not isinstance(e.face.surf, core.RevSurf):
                continue
            s = e.face.surf
            ax = (e.face.triangles.reshape(-1, 3) - s.o) @ s.d
            lo, hi = float(ax.min()), float(ax.max())
            if seg is not None:
                for q in seg:
                    t = (np.asarray(q) - s.o) @ s.d
                    if np.linalg.norm(np.asarray(q) - s.o - t * s.d) < 1e-6:
                        lo, hi = min(lo, t), max(hi, t)
            ext = 0.15 * (hi - lo) + 1.0
            lo, hi = lo - ext, hi + ext
            L = hi - lo
            dash, gap, dot = 3.0, 1.0, 0.6
            pts, t = [], lo
            while t < hi:
                for seglen in (dash, dot):
                    t2 = min(t + seglen, hi)
                    pts += [s.o + t * s.d, s.o + t2 * s.d]
                    t = t2 + gap
                    if t >= hi:
                        break
            if pts:
                ln = gl.GLLinePlotItem(pos=np.array(pts), mode="lines", color=MEAS_COLOR, width=1.4,
                                       antialias=True, glOptions=GL_ONTOP)
                ln.setDepthValue(19)
                out.append(ln)
        return out

    def _arc_info(self, fi):
        """Kreisbogen eines Radius im Nullpunkt-KS (Drehmaschine: Z und X als Durchmesser)."""
        try:
            ap = core.arc_profile(self.model, fi)
        except Exception:
            return None
        w = self.wcs
        o, d = ap["o"], ap["d"]
        dz = w.dir_to_wcs(d)
        ow = w.to_wcs(o)
        lathe = abs(abs(dz[2]) - 1) < 1e-6 and math.hypot(ow[0], ow[1]) < 1e-4
        f = self.model.faces[fi]
        # Profil-Ebene: Seite der Fläche (bei Vollkreis +X des Nullpunkts)
        if f.face_angle is not None and f.face_angle < 180:
            V = f.triangles.reshape(-1, 3) - o
            u = V.mean(axis=0)
        else:
            u = w.R[0].copy()
        u = u - (u @ d) * d
        if np.linalg.norm(u) < 1e-9:
            u = core.any_perp(d)
        u = u / np.linalg.norm(u)
        P = lambda q: o + q[0] * d + q[1] * u
        out = {"r": ap["r"], "exact": ap["exact"], "concave": ap["concave"], "lathe": lathe,
               "p3": P(ap["c"]), "e3": [P(q) for q in ap["ends"]], "u": u, "o": o, "d": d,
               "c2": ap["c"], "ends2": ap["ends"]}
        if lathe:
            out["cz"] = float(w.to_wcs(o + ap["c"][0] * d)[2])
            out["cx"] = 2 * float(ap["c"][1])
            ends = [(float(w.to_wcs(o + q[0] * d)[2]), 2 * float(q[1])) for q in ap["ends"]]
            out["ends"] = sorted(ends, key=lambda t: -t[0])
        return out

    def _show_entry_overlay(self, i):
        """Eintrittspunkt: Bohrungsachse und Flankenlinie der Fläche (im Längsschnitt) durch den Punkt –
        beide schneiden sich genau im roten Punkt auf der Fläche."""
        m = self.model
        h = self.holes[i]
        p = np.asarray(h.entry, float)
        ax = core.unit(h.axis_d)                 # zeigt aus dem Bauteil heraus
        R = max(h.diameters) / 2 if h.diameters else m.size * 0.01
        Lout = max(R * 6, m.size * 0.08)
        segs_axis = [p + Lout * ax, p - max(h.depth, R) * ax]
        items = []
        fi, n = core.entry_surface(m, h)
        if n is not None:
            mo, md = core.main_axis(m)
            q = p - mo
            u = q - (q @ md) * md
            u = core.unit(u) if np.linalg.norm(u) > 1e-9 else core.any_perp(md)
            # Flankenlinie: Schnitt der Tangentialebene mit dem Längsschnitt (Ebene aus Längsachse und Radial)
            t = np.cross(n, np.cross(md, u))
            if np.linalg.norm(t) > 1e-9:
                t = core.unit(t)
                Lf = max(R * 5, m.size * 0.06)
                fl = gl.GLLinePlotItem(pos=np.array([p - Lf * t, p + Lf * t]), mode="lines",
                                       color=(0.85, 0.45, 0.0, 1.0), width=2.2, antialias=True, glOptions=GL_ONTOP)
                fl.setDepthValue(20)
                items.append(fl)
        al = gl.GLLinePlotItem(pos=np.array(segs_axis), mode="lines", color=(0.10, 0.30, 0.75, 1.0), width=2.0,
                               antialias=True, glOptions=GL_ONTOP)
        al.setDepthValue(20)
        dot = gl.GLScatterPlotItem(pos=np.array([p]), size=13, color=(0.86, 0.05, 0.05, 1.0), pxMode=True,
                                   glOptions=GL_ONTOP)
        dot.setDepthValue(22)
        items += [al, dot]
        for it in items:
            self.view.addItem(it)
        self.items_overlay = list(self.items_overlay) + items

    def _show_arc_overlay(self, fi):
        ap = self._arc_info(fi)
        if ap is None:
            return
        c3 = ap["p3"]
        items = []
        segs = []
        for q in ap["e3"]:
            segs += [c3, q]
        # Kreuz im Mittelpunkt
        u, d = ap["u"], ap["d"]
        k = max(ap["r"] * 0.25, self.model.size * 0.006)
        segs += [c3 - k * u, c3 + k * u, c3 - k * d, c3 + k * d]
        ln = gl.GLLinePlotItem(pos=np.array(segs), mode="lines", color=MEAS_COLOR, width=1.6, antialias=True,
                               glOptions=GL_ONTOP)
        ln.setDepthValue(20)
        # Bogen selbst im Profil
        c2, r = ap["c2"], ap["r"]
        if len(ap["ends2"]) >= 2:
            a0 = math.atan2(ap["ends2"][0][1] - c2[1], ap["ends2"][0][0] - c2[0])
            a1 = math.atan2(ap["ends2"][1][1] - c2[1], ap["ends2"][1][0] - c2[0])
            # Richtung so, dass der Bogen durch die Fläche läuft
            V = self.model.faces[fi].triangles.reshape(-1, 3) - ap["o"]
            av = V @ d
            rv = np.linalg.norm(V - np.outer(av, d), axis=1)
            am = math.atan2(rv.mean() - c2[1], av.mean() - c2[0])
            da = (a1 - a0) % (2 * math.pi)
            if (am - a0) % (2 * math.pi) > da:
                da -= 2 * math.pi
            t = a0 + np.linspace(0, da, 40)
            arc = [ap["o"] + (c2[0] + r * math.cos(x)) * d + (c2[1] + r * math.sin(x)) * u for x in t]
            al = gl.GLLinePlotItem(pos=np.array(arc), mode="line_strip", color=(1.0, 0.1, 0.1, 1.0), width=2.5,
                                   antialias=True, glOptions=GL_ONTOP)
            al.setDepthValue(20)
            items.append(al)
        dots = gl.GLScatterPlotItem(pos=np.array([c3] + list(ap["e3"])), size=8, color=MEAS_COLOR, pxMode=True,
                                    glOptions=GL_ONTOP)
        dots.setDepthValue(21)
        items += [ln, dots]
        for it in items:
            self.view.addItem(it)
        self.items_overlay = list(self.items_overlay) + items
        if ap["lathe"]:
            txt = f"R {fmt(ap['r'])}\nM  Z {fmt(ap['cz'])}\nM  X Ø {fmt(ap['cx'])}"
            self.status.showMessage(f"Radius R {fmt(ap['r'])} · Mittelpunkt Z {fmt(ap['cz'])} · X (Ø) {fmt(ap['cx'])}"
                                    + "  ·  " + "  ·  ".join(f"{'Anfang' if i == 0 else 'Ende'} Z {fmt(z)} Ø {fmt(x)}"
                                                         for i, (z, x) in enumerate(ap["ends"])))
        else:
            pw = self.wcs.to_wcs(c3)
            txt = f"R {fmt(ap['r'])}\nM  X {fmt(pw[0])} Y {fmt(pw[1])} Z {fmt(pw[2])}"
        self.sel_callouts = [self._callout(c3, txt)]
        self._sync_callouts()

    def _show_selection_info(self):
        n = len(self.sel)
        if n == 0:
            self.sel_info.setHtml("<i>Keine Auswahl.</i>")
            return
        if n == 1:
            self.sel_info.setHtml(self._entity_html(self.sel[0]))
            e = self.sel[0]
            if e.kind == "point":
                pw = self.wcs.to_wcs(e.point)
                ei = self._entry_at(e.point)
                if ei is not None:
                    self._show_entry_overlay(ei)
                    self.sel_callouts = [self._callout(e.point, self._entry_text(ei)[0],
                                                       bg=self._qc("#fffff0f0"), fg=self._qc("#ffb00020"))]
                else:
                    self.sel_callouts = [self._callout(e.point, f"X {fmt(pw[0])}\nY {fmt(pw[1])}\nZ {fmt(pw[2])}")]
                self._sync_callouts()
            elif e.kind == "edge" and str(e.ref).startswith("Ecke-"):
                ed = e.edge
                cw = self.wcs.to_wcs(ed.center)
                pw = self.wcs.to_wcs(ed.points[len(ed.points) // 2])
                self.sel_callouts = [self._callout(ed.points[len(ed.points) // 2],
                                                   f"Scharfe Ecke\nZ {fmt(cw[2])}\nØ {fmt(2 * ed.radius)}  (X {fmt(ed.radius)})",
                                                   bg=self._qc("#ffffe6e6"), fg=self._qc("#ffb00020"))]
                self._sync_callouts()
                self.status.showMessage(f"Scharfe Ecke: Z {fmt(cw[2])} · Ø {fmt(2 * ed.radius)} · "
                                        f"Punkt X {fmt(pw[0])} Y {fmt(pw[1])} Z {fmt(pw[2])}")
            elif self._thread_for_entity(e) is not None:
                self._show_thread_overlay(self._thread_for_entity(e))
            elif e.kind == "face" and e.face.ftype in ("TORUS", "SPHERE") and self._arc_info(e.ref) is not None:
                self._show_arc_overlay(e.ref)
            else:
                dia = e.diameter()
                if dia:
                    self.sel_callouts = [self._callout(self._entity_center(e), f"Ø {fmt(dia)}")]
                    self._sync_callouts()
        elif n == 2:
            rows, seg = core.measure(self.sel[0], self.sel[1], self.wcs)
            title = f"Messung: {self.sel[0].label()} ↔ {self.sel[1].label()}"
            html = html_table(rows, title)
            html += "<p style='color:#6b625a'><i>Werte für dX/dY/dZ beziehen sich auf den aktuellen Nullpunkt.</i></p>"
            self.sel_info.setHtml(html)
            import re as _re
            txt = _re.sub(r"<[^>]+>", "", f"{rows[0][0]}: {rows[0][1]}") if rows else ""
            self.status.showMessage(txt)
            items = []
            if seg is not None and np.linalg.norm(seg[1] - seg[0]) > 1e-9:
                ln = gl.GLLinePlotItem(pos=np.array(seg), color=MEAS_COLOR, width=2.5, antialias=True,
                                       glOptions=GL_ONTOP)
                ln.setDepthValue(20)
                dots = gl.GLScatterPlotItem(pos=np.array(seg), size=8, color=MEAS_COLOR, pxMode=True,
                                            glOptions=GL_ONTOP)
                items += [ln, dots]
                lpos = (seg[0] + seg[1]) / 2
            else:
                lpos = self._entity_center(self.sel[1])
            items += self._axis_lines(self.sel[:2], seg)
            if rows and lpos is not None:
                lines = [_re.sub(r"<[^>]+>", "", f"{k}: {v}") for k, v in rows[:3]]
                self.sel_callouts = [self._callout(lpos, "\n".join(lines))]
                self._sync_callouts()
            for it in items:
                self.view.addItem(it)
            self.items_overlay = items
            self._update_angle_overlay()
        else:
            rows = [(f"{i + 1}.", e.label()) for i, e in enumerate(self.sel)]
            html = html_table(rows, f"{n} Elemente gewählt")
            ang = self._hole_angles()
            if ang is not None:
                lst, info = ang
                html += html_table([(f"{fmt(a, 3)}°", lab) for a, lab, *_ in lst], "Winkellage der Bohrungen")
                html += f"<p style='color:#6b625a'><i>{info}</i></p>"
            try:
                o, z, txt = core.wcs_from_selection(self.model, self.sel, self.wcs)
                html += html_table([("Ergebnis", txt), ("Punkt (Nullpunkt-KS)", vtxt(self.wcs.to_wcs(o))),
                                    ("Achse", vtxt(self.wcs.dir_to_wcs(z)))], "Auswertung")
                html += "<p><i>„Nullpunkt → Nullpunkt aus Auswahl“ übernimmt diesen Punkt.</i></p>"
            except Exception as ex:
                html += f"<p><i>{ex}</i></p>"
            self.sel_info.setHtml(html)
        self._update_angle_overlay()
        if not getattr(self, "_keep_tab", False):
            self.tabs_right.setCurrentIndex(0)

    def _hole_angles(self):
        """Winkellage parallel liegender Bohrungen um ihre gemeinsame Mitte.
        0° = X-Achse des Nullpunkts, gezählt gegen den Uhrzeigersinn wie in der aktuellen Ansicht."""
        if self.model is None or len(self.sel) < 2:
            return None
        axes = []
        for e in self.sel:
            ax = e.axis()
            if ax is None:
                return None
            axes.append((e.primary().p, core.unit(ax[1]), e))
        d = axes[0][1]
        if any(core._angle(d, a[1]) > 0.05 for a in axes):
            return None
        pts = np.array([a[0] for a in axes])
        # Mitte: Hauptachse, wenn parallel; sonst Kreis durch die Achsen
        mo, md = core.main_axis(self.model)
        if core._angle(md, d) < 0.5:
            c = mo
        elif len(axes) >= 3:
            c, _, _ = core._fit_circle_3d(pts, d)
        else:
            c = pts.mean(axis=0)
        # Blickrichtung: gegen den Uhrzeigersinn so, wie man gerade draufschaut
        q = self.view.opts["rotation"]
        tc = q.conjugated().rotatedVector(QtGui.QVector3D(0, 0, 1))
        toward = np.array([tc.x(), tc.y(), tc.z()])
        n = d if d @ toward >= 0 else -d
        e1 = None
        for cand in (self.wcs.R[0], self.wcs.R[1], self.wcs.R[2]):
            if abs(cand @ n) < 0.9:
                e1 = core.unit(cand - (cand @ n) * n)
                break
        e2 = np.cross(n, e1)
        names = "XYZ"
        k = [abs(core.unit(self.wcs.R[i]) @ e1) for i in range(3)]
        ref = names[int(np.argmax(k))]
        lst = []
        for p, _, e in axes:
            v = p - c
            v = v - (v @ n) * n
            r = float(np.linalg.norm(v))
            th = round(math.degrees(math.atan2(v @ e2, v @ e1)), 4) % 360.0
            if abs(th - 360.0) < 1e-6:
                th = 0.0
            if r < 0.01:
                return None            # Element liegt auf der Mittelachse -> kein Winkel
            lst.append((th, e.label(), p, v / r, r, e))
        lst.sort(key=lambda x: x[0])
        info = (f"0° = +{ref}-Achse des Nullpunkts, gegen den Uhrzeigersinn in der aktuellen Blickrichtung. "
                f"Nach dem Drehen der Ansicht werden die Winkel neu ausgerichtet.")
        self._ang_center = (c, n)
        return lst, info

    def _align_for(self, p, c):
        """Textausrichtung so wählen, dass Beschriftungen vom Bauteil weg zeigen."""
        S, _ = self.view.to_screen(np.array([p, c], float))
        dx, dy = S[0] - S[1]
        L = math.hypot(dx, dy) or 1.0
        A = Qt.AlignmentFlag
        h = A.AlignRight if dx < -0.3 * L else (A.AlignLeft if dx > 0.3 * L else A.AlignHCenter)
        v = A.AlignBottom if dy < -0.3 * L else (A.AlignTop if dy > 0.3 * L else A.AlignVCenter)
        return h | v

    def _update_angle_overlay(self):
        for it in getattr(self, "angle_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.angle_items = []
        if len(self.sel) < 2:
            return
        res = self._hole_angles()
        if res is None:
            return
        lst, _ = res
        c, n = self._ang_center
        m = self.model
        V = m.tris.reshape(-1, 3)
        w = V - c
        R_out = float(np.linalg.norm(w - np.outer(w @ n, n), axis=1).max())
        R_lab = R_out * 1.12 + m.size * 0.02
        font = QtGui.QFont("Segoe UI", 13, QtGui.QFont.Weight.Bold)
        items = []
        for th, lab, p, u, r, e in lst:
            a = (p - c) @ n
            base = c + a * n
            q1 = base + r * u
            q2 = base + R_lab * u
            ln = gl.GLLinePlotItem(pos=np.array([q1, q2]), color=(0.10, 0.10, 0.10, 0.85), width=1.2,
                                   antialias=True, glOptions=GL_ONTOP)
            lp = base + (R_lab + m.size * 0.01) * u
            tx = gl.GLTextItem(pos=lp, text=f"{fmt(th, 2)}°", color=QtGui.QColor("#101010"), font=font,
                               alignment=self._align_for(lp, base))
            items += [ln, tx]
        for it in items:
            self.view.addItem(it)
        self.angle_items = items
        # Tabelle bei Blickwechsel aktualisieren
        if len(self.sel) >= 3:
            self._show_selection_text_only()

    def _show_selection_text_only(self):
        rows = [(f"{i + 1}.", e.label()) for i, e in enumerate(self.sel)]
        html = html_table(rows, f"{len(self.sel)} Elemente gewählt")
        ang = self._hole_angles()
        if ang is not None:
            lst, info = ang
            html += html_table([(f"{fmt(a, 3)}°", lab) for a, lab, *_ in lst], "Winkellage der Bohrungen")
            html += f"<p style='color:#6b625a'><i>{info}</i></p>"
        try:
            o, z, txt = core.wcs_from_selection(self.model, self.sel, self.wcs)
            html += html_table([("Ergebnis", txt), ("Punkt (Nullpunkt-KS)", vtxt(self.wcs.to_wcs(o))),
                                ("Achse", vtxt(self.wcs.dir_to_wcs(z)))], "Auswertung")
        except Exception:
            pass
        self.sel_info.setHtml(html)

    def _entity_center(self, e):
        if e.kind == "point":
            return e.point
        if e.kind == "edge":
            return e.edge.points[len(e.edge.points) // 2]
        T = e.face.triangles
        c = T.mean(axis=1)
        areas = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        return c[np.argmax(areas)]

    def _entity_html(self, e):
        m, w = self.model, self.wcs
        if e.kind == "point":
            pw = w.to_wcs(e.point)
            return html_table([("X", f"<b>{fmt(pw[0])}</b>"), ("Y", f"<b>{fmt(pw[1])}</b>"),
                               ("Z", f"<b>{fmt(pw[2])}</b>"),
                               ("Ø an dieser Stelle", fmt(2 * math.hypot(pw[0], pw[1]))
                                if w.name.startswith("Drehmaschine") else "–"),
                               ("Bezug", w.name),
                               ("Modell (CAD)", vtxt(e.point))], e.label())
        if e.kind == "edge":
            ed = e.edge
            rows = [("Typ", core.EDGE_TYPES_DE.get(ed.etype, ed.etype))]
            if ed.etype == "CIRCLE":
                rows += [("Durchmesser", f"<b>Ø {fmt(2 * ed.radius)} mm</b>"), ("Radius", fmt(ed.radius)),
                         ("Mittelpunkt", vtxt(w.to_wcs(ed.center))),
                         ("Achse (I/J/K)", vtxt(w.dir_to_wcs(ed.normal))),
                         ("Bogenlänge", fmt(ed.length))]
                if ed.length and ed.radius:
                    rows.append(("Öffnungswinkel", f"{fmt(math.degrees(ed.length / ed.radius), 2)}°"))
            else:
                rows += [("Länge", f"{fmt(ed.length)} mm"), ("Start", vtxt(w.to_wcs(ed.start))),
                         ("Ende", vtxt(w.to_wcs(ed.end))),
                         ("Richtung", vtxt(w.dir_to_wcs(core.unit(ed.end - ed.start))))]
            return html_table(rows, ed.id)
        f = e.face
        s = f.surf
        rows = [("Typ", core.FACE_TYPES_DE.get(f.ftype, f.ftype))]
        if isinstance(s, core.RevSurf):
            if s.kind == "CYLINDER":
                rows += [("Durchmesser", f"<b>Ø {fmt(2 * s.radius)} mm</b>"), ("Radius", fmt(s.radius))]
            elif s.kind == "CONE":
                rows.append(("Kegel-Halbwinkel", f"{fmt(s.half_angle, 3)}°  (Spitze {fmt(2 * s.half_angle, 3)}°)"))
                for r in (f.radius1, f.radius2):
                    if r is not None:
                        rows.append(("Ø", fmt(2 * r)))
            elif s.kind == "TORUS":
                ap = self._arc_info(e.ref)
                if ap is not None and ap["lathe"]:
                    rows += [("Radius", f"<b>R {fmt(ap['r'])} mm</b>"),
                             ("Mittelpunkt Z", f"<b>{fmt(ap['cz'])}</b>"),
                             ("Mittelpunkt X (Ø)", f"<b>{fmt(ap['cx'])}</b>"),
                             ("Mittelpunkt X (radial)", fmt(ap['cx'] / 2))]
                    for k, (z, x) in enumerate(ap["ends"]):
                        rows.append((("Anfang" if k == 0 else "Ende") + " Z / X(Ø)", f"{fmt(z)} / {fmt(x)}"))
                    rows.append(("Art", "Hohlkehle (Mittelpunkt außerhalb Material)" if ap["concave"]
                                 else "Rundung (Mittelpunkt im Material)"))
                    if not ap["exact"]:
                        rows.append(("Hinweis", "Mittelpunkt aus Tessellierung ausgeglichen"))
                elif ap is not None:
                    rows += [("Radius", f"<b>R {fmt(ap['r'])} mm</b>"),
                             ("Mittelpunkt", vtxt(w.to_wcs(ap['p3'])))]
                else:
                    rows += [("Rundung R", f"<b>R {fmt(s.r)} mm</b>"), ("Mittenkreis Ø", fmt(2 * s.major))]
            p = e.primary()
            rows += [("Achspunkt", vtxt(w.to_wcs(p.p))), ("Achse (I/J/K)", vtxt(w.dir_to_wcs(s.d)))]
        elif isinstance(s, core.PlaneSurf):
            nrm = core.outward_normal(m, e.ref, s.n)
            rows += [("Normale (nach außen)", vtxt(w.dir_to_wcs(nrm)))]
            pw = w.to_wcs(s.p)
            dz = float(w.dir_to_wcs(nrm)[2])
            if abs(abs(dz) - 1) < 1e-6:
                rows.append(("Lage Z (Nullpunkt-KS)", f"<b>{fmt(pw[2])}</b>"))
            rows.append(("Punkt auf Fläche", vtxt(pw)))
        if f.length is not None:
            rows.append(("Länge", f"{fmt(f.length)} mm"))
        if f.area is not None:
            rows.append(("Flächeninhalt", f"{fmt(f.area, 2)} mm²"))
        if f.concavity:
            rows.append(("Konkavität", {"1": "konkav (innen)", "0": "konvex (außen)"}.get(f.concavity, f.concavity)))
        ti = self._thread_for_entity(e)
        if ti is not None:
            rows += [("<b>Gewindebohrung</b>", "")] + self._thread_rows(ti)
        ft = m.features.get(f.feature)
        if ft:
            rows.append(("Feature", ft.name))
            rows += [(f"&nbsp;&nbsp;{k}", v) for k, v in ft.params.items()]
        return html_table(rows, f.id)

    # ---------------- Nullpunkt ----------------
    def _wcs_changed(self):
        self._build_pattern_items()
        self._build_box_items()
        if self.section is not None:
            self._apply_section()
        if self.cb_lock.currentIndex() in (2, 3, 4):
            self.on_lock(self.cb_lock.currentIndex())
        self._build_wcs_items()
        self._update_wcs_view()
        self._fill_holes()
        self._fill_lathe()
        self._refresh_selection()

    def _wcs_op(self, fn):
        if self.model is None:
            return
        fn()
        self.wcs.name = "Benutzer-Nullpunkt"
        self._wcs_changed()

    def wcs_from_selection(self):
        if self.model is None:
            return
        # Querbohrung gewählt, Z liegt auf der Längsachse → meist ist „C0° hierher“ gemeint
        if len(self.sel) == 1 and self.sel[0].kind == "face":
            h = core.hole_for_entity(self.model, self.sel[0])
            mo, md = core.main_axis(self.model)
            if h is not None and abs(core.unit(h.axis_d) @ md) < 0.1 and abs(self.wcs.R[2] @ md) > 0.999:
                box = QtWidgets.QMessageBox(self)
                box.setWindowTitle(APP_NAME)
                box.setText("Du hast eine Querbohrung gewählt. Was soll passieren?")
                b1 = box.addButton("Diese Bohrung = C0°  (Z bleibt Längsachse)", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                b2 = box.addButton("Neuer Nullpunkt in der Bohrung  (Z = Bohrungsachse)",
                                   QtWidgets.QMessageBox.ButtonRole.ActionRole)
                box.addButton("Abbrechen", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.exec()
                if box.clickedButton() is b1:
                    self.wcs_align_x()
                    return
                if box.clickedButton() is not b2:
                    return
        try:
            o, z, txt = core.wcs_from_selection(self.model, self.sel, self.wcs)
        except Exception as ex:
            QtWidgets.QMessageBox.information(self, APP_NAME, str(ex))
            return
        self.wcs.set(o, z, name=txt)
        self._wcs_changed()
        self.status.showMessage(f"Nullpunkt gesetzt: {txt}")

    def wcs_reset(self):
        self.wcs.reset()
        self._wcs_changed()

    def wcs_lathe(self):
        if self.model is None:
            return
        self.wcs.set_lathe(self.model)
        self._wcs_changed()

    def wcs_rotate_free(self):
        if self.model is None:
            return
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, APP_NAME, "C0° (X-Achse) um die Längsachse drehen um … Grad\n"
            "(positiv = gegen den Uhrzeigersinn mit Blick auf die Stirnseite)", 180.0, -360.0, 360.0, 3)
        if ok and abs(val) > 1e-9:
            self._wcs_op(lambda: self.wcs.rotate_z(val))
            self.status.showMessage(f"C0° um {fmt(val, 3)}° gedreht.")

    def _align_target(self, e):
        """Richtung vom Nullpunkt zum gewählten Element (für C0°)."""
        z = self.wcs.R[2]
        if e.kind == "face":
            h = core.hole_for_entity(self.model, e)
            if h is not None and abs(core.unit(h.axis_d) @ z) < 0.99:
                return np.asarray(h.entry, float) - self.wcs.origin      # radiale Bohrung: Eintrittsseite
            ax = e.axis()
            if ax is not None and abs(core.unit(ax[1]) @ z) > 0.99:
                return ax[0] - self.wcs.origin                             # achsparallele Bohrung: Bohrungsmitte
            return e.face.triangles.reshape(-1, 3).mean(axis=0) - self.wcs.origin
        if e.kind == "edge":
            ed = e.edge
            if ed.etype == "CIRCLE" and ed.center is not None:
                v = ed.center - self.wcs.origin
                if np.linalg.norm(v - (v @ z) * z) > 1e-6:
                    return v
            return ed.points.mean(axis=0) - self.wcs.origin
        return e.point - self.wcs.origin

    def wcs_align_x(self):
        if not self.sel:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Bitte zuerst die Bohrung, Kante oder den Punkt anklicken, der C0° werden soll.")
            return
        d = self._align_target(self.sel[0])
        if not self.wcs.align_x(d):
            QtWidgets.QMessageBox.information(self, APP_NAME, "Richtung liegt parallel zu Z.")
            return
        self._wcs_changed()
        self.status.showMessage(f"C0° liegt jetzt auf {self.sel[0].label()} – Bohrbild und Winkel neu berechnet.")

    def wcs_shift(self):
        dlg = ShiftDialog(self)
        if dlg.exec():
            self.wcs.shift(*dlg.values())
            self._wcs_changed()

    def _update_wcs_view(self):
        w = self.wcs
        rows = [("Art", w.name), ("Ursprung (Modell)", vtxt(w.origin)),
                ("X-Achse", vtxt(w.R[0])), ("Y-Achse", vtxt(w.R[1])), ("Z-Achse", vtxt(w.R[2]))]
        if self.model is not None:
            V = self.model.tris.reshape(-1, 3)
            P = (V - w.origin) @ w.R.T
            lo, hi = P.min(axis=0), P.max(axis=0)
            rows += [("Bereich X", f"{fmt(lo[0])} … {fmt(hi[0])}"),
                     ("Bereich Y", f"{fmt(lo[1])} … {fmt(hi[1])}"),
                     ("Bereich Z", f"{fmt(lo[2])} … {fmt(hi[2])}")]
        self.wcs_info.setHtml(html_table(rows, "Aktueller Nullpunkt"))

    # ---------------- Tabellen ----------------
    def _fill_tables(self):
        m = self.model
        u = m.units
        V = m.tris.reshape(-1, 3)
        Pw = (V - self.wcs.origin) @ self.wcs.R.T
        dims = Pw.max(axis=0) - Pw.min(axis=0)
        dmax = 2 * float(np.hypot(Pw[:, 0], Pw[:, 1]).max())
        pmap = {n: v for _, n, v in m.params}

        def like(*parts):
            """Parameter, deren Name einen der Begriffe enthält (unabhängig vom CAD-System)."""
            for part in parts:
                for _, n, v in m.params:
                    if part.upper() in n.upper() and v:
                        return v
            return "–"

        def pick(*names):
            for n in names:
                if pmap.get(n):
                    return pmap[n]
            return "–"

        rows = [
            ("Name", m.name),
            ("Benennung", like("PART_NAME_GE", "PART_NAME", "BENENNUNG")),
            ("Material", like("MATERIAL", "WERKSTOFF")),
            ("Ident", like("SAP_IDENT", "IDENT")),
            ("Item-ID", like("ITEM_ID", "PART_NO", "SACHNUMMER")),
            ("Länge (Z)", f"{fmt(dims[2])} {u}"),
            ("Ø max", f"{fmt(dmax)} {u}"),
            ("Höhe (X)", f"{fmt(dims[0])} {u}"),
            ("Breite (Y)", f"{fmt(dims[1])} {u}"),
            ("Rohteil (L×B×H)", "{} × {} × {} {}".format(
                fmt(m.info.get("length")), fmt(m.info.get("width")), fmt(m.info.get("height")), u)),
            ("Volumen", f"{fmt(core._float(m.info.get('volume')), 1)} {u}³"),
            ("Oberfläche", f"{fmt(core._float(m.info.get('area')), 1)} {u}²"),
            ("Masse", f"{fmt(core._float(m.info.get('mass')), 3)} kg"),
            ("Flächen", len(m.faces)),
            ("Kanten", len(m.edges)),
            ("Features", len(m.features)),
            ("Quellsystem", m.system),
            ("Exportdatum", m.date),
            ("Datei", os.path.basename(m.path)),
        ]
        self.model_table.set_rows(rows)
        self.param_table.set_rows([(n, v, t) for t, n, v in sorted(m.params, key=lambda x: x[1].lower())])
        self.filter_params(self.param_filter.text())

        frows = []
        for f in m.faces:
            frows.append((f.id, core.FACE_TYPES_DE.get(f.ftype, f.ftype), self._dia_text(f), fmt(f.length),
                          fmt(f.area, 2), m.features[f.feature].name if f.feature in m.features else ""))
        self.face_table.set_rows(frows)

        self.feat_tree.clear()
        groups = {}
        for ft in m.features.values():
            groups.setdefault(ft.ftype or "Sonstige", []).append(ft)
        for gname in sorted(groups):
            g = QtWidgets.QTreeWidgetItem([f"{gname} ({len(groups[gname])})", ""])
            self.feat_tree.addTopLevelItem(g)
            for ft in groups[gname]:
                it = QtWidgets.QTreeWidgetItem([ft.name, ft.ftype])
                it.setData(0, Qt.ItemDataRole.UserRole, ft.id)
                for k, v in ft.params.items():
                    it.addChild(QtWidgets.QTreeWidgetItem([k, v]))
                g.addChild(it)
        self.feat_tree.resizeColumnToContents(0)

        trows = []
        for ft in m.features.values():
            p = ft.params
            desc = p.get("thread-description") or p.get("thread-size")
            if desc or "THREAD" in ft.ftype.upper():
                trows.append((ft.name, desc or "–", fmt(core._float(p.get("tapped-diameter"))),
                              fmt(core._float(p.get("thread-depth"))), fmt(core._float(p.get("thread-pitch")))))
        self.thread_table.set_rows(trows)

    def _hole_machine_cols(self, h):
        """Maschinenwerte (X Ø, Y, Z, C, B) und Senkung/Spitze einer Bohrung für die Tabelle."""
        try:
            ei = core.entry_info(self.model, h, self.wcs)
            hd = core.hole_details(self.model, h)
        except Exception:
            return ("",) * 9
        sk = "; ".join(f"{a} Ø{fmt(d)}" + (f" {fmt(w, 3)}°" if w else "") + f" T {fmt(t)}"
                       for a, d, w, t in hd["sinks"])
        tip = f"{fmt(hd['tip'][0], 3)}° L {fmt(hd['tip'][1])}" if hd["tip"] else ("durch" if h.through else "flach")
        return (fmt(ei["dia"]), fmt(ei["y"]), fmt(ei["z"]), fmt(ei["c"], 3), fmt(ei["b"], 3),
                fmt(ei["depth_x0"]) if ei["depth_x0"] is not None else "", sk, tip,
                fmt(hd["full"]))

    def _fill_holes(self):
        rows = []
        ang = {}
        if self.model is not None:
            self.model._holes = self.holes
            groups, (o, dm, e1, e2) = core.hole_pattern(self.model, self.wcs)
            for g in groups:
                for h, th in zip(g.holes, g.angles):
                    ang[h] = (th, g.radius if g.kind in ("axial", "inclined") else None)
        for i, h in enumerate(self.holes):
            e = self.wcs.to_wcs(h.entry)
            d = self.wcs.dir_to_wcs(h.axis_d)
            th, rr = ang.get(i, (None, None))
            ti = core.thread_info(self.model, h) if h.thread else None
            rows.append((i + 1, " / ".join(fmt(x) for x in h.diameters), h.thread,
                         fmt(ti["length"]) if ti else "", fmt(ti["core_end"]) if ti else "",
                         fmt(e[0]), fmt(e[1]), fmt(e[2]), fmt(d[0], 4), fmt(d[1], 4), fmt(d[2], 4),
                         fmt(h.depth), "durch" if h.through else "Sackloch",
                         fmt(th, 3) if th is not None else "", fmt(rr) if rr is not None else "",
                         *self._hole_machine_cols(h),
                         ", ".join(h.features)))
        self.hole_table.set_rows(rows)

    def export_holes(self):
        if not self.holes:
            return
        base = os.path.splitext(self.model.path)[0] + "_Bohrungen.csv"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Bohrungstabelle speichern", base, "CSV (*.csv)")
        if not p:
            return
        hdr = [self.hole_table.horizontalHeaderItem(c).text() for c in range(self.hole_table.columnCount())]
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(f"# Nullpunkt: {self.wcs.name}; Ursprung (Modell): {vtxt(self.wcs.origin)}\n")
            fh.write(";".join(hdr) + "\n")
            for r in range(self.hole_table.rowCount()):
                fh.write(";".join(self.hole_table.item(r, c).text() for c in range(len(hdr))) + "\n")
        self.status.showMessage(f"Gespeichert: {p}")

    @staticmethod
    def _dia_text(f):
        s = f.surf
        if isinstance(s, core.RevSurf) and s.kind == "CYLINDER":
            return f"Ø {fmt(2 * s.radius)}"
        if f.ftype == "CONE":
            parts = [fmt(2 * r) for r in (f.radius1, f.radius2) if r]
            return ("Ø " + " / ".join(parts)) if parts else ""
        if isinstance(s, core.RevSurf) and s.kind == "TORUS":
            return f"R {fmt(s.r)}"
        return ""

    def filter_params(self, text):
        t = text.lower().strip()
        for r in range(self.param_table.rowCount()):
            row = " ".join(self.param_table.item(r, c).text().lower() for c in range(self.param_table.columnCount()))
            self.param_table.setRowHidden(r, bool(t) and t not in row)

    def on_face_row(self):
        rows = self.face_table.selectionModel().selectedRows()
        if rows and self.model:
            self.sel = [core.Entity(self.model, "face", r.row()) for r in rows]
            self._refresh_selection()

    def on_feature_row(self):
        items = self.feat_tree.selectedItems()
        if not items or self.model is None:
            return
        fid = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not fid:
            return
        self.sel = [core.Entity(self.model, "face", i) for i, f in enumerate(self.model.faces) if f.feature == fid]
        self._refresh_selection()

    # ---------------- Drehteil: Maßliste und Kontur ----------------
    def _build_lathe_tab(self):
        tabs = QtWidgets.QTabWidget()
        self.lathe_tabs = tabs
        # Maßliste
        w1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(w1)
        l1.setContentsMargins(2, 2, 2, 2)
        top = QtWidgets.QHBoxLayout()
        self.dim_filter = QtWidgets.QComboBox()
        self.dim_filter.addItems(["Alle", "Durchmesser", "Planflächen", "Fasen / Kegel", "Radien"])
        self.dim_filter.currentIndexChanged.connect(self._fill_lathe)
        top.addWidget(QtWidgets.QLabel("Zeigen:"))
        top.addWidget(self.dim_filter, 1)
        b = QtWidgets.QPushButton("Als CSV speichern …")
        b.clicked.connect(lambda: self._export_table(self.dim_table, "_Massliste.csv", "Maßliste speichern"))
        top.addWidget(b)
        l1.addLayout(top)
        self.dim_table = InfoTable(["Art", "Maß", "Z von", "Z bis", "Länge", "ab links", "ab rechts", "Info"])
        self.dim_table.itemSelectionChanged.connect(lambda: self._lathe_row_select(self.dim_table, self._dim_rows))
        l1.addWidget(self.dim_table, 1)
        lab = QtWidgets.QLabel("Z = Längsachse im aktuellen Nullpunkt. „ab links“ / „ab rechts“ = Abstand von der "
                               "Stirnfläche am kleinsten bzw. größten Z – wie in der Zeichnung bemaßt. "
                               "Zeile anklicken = Fläche zeigen.")
        lab.setWordWrap(True)
        l1.addWidget(lab)
        tabs.addTab(w1, "Maßliste")
        # Kontur
        w2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(w2)
        l2.setContentsMargins(2, 2, 2, 2)
        top = QtWidgets.QHBoxLayout()
        self.cont_sel = QtWidgets.QComboBox()
        self.cont_sel.addItems(["Außenkontur", "Innenkontur"])
        self.cont_sel.currentIndexChanged.connect(self._fill_lathe)
        top.addWidget(self.cont_sel, 1)
        self.cont_rev = QtWidgets.QCheckBox("umkehren")
        self.cont_rev.setToolTip("Reihenfolge umdrehen (von der anderen Seite her)")
        self.cont_rev.toggled.connect(self._fill_lathe)
        top.addWidget(self.cont_rev)
        self.cont_show = QtWidgets.QCheckBox("im Bild")
        self.cont_show.setToolTip("Kontur als Linie im 3D-Bild zeigen")
        self.cont_show.toggled.connect(self._show_contour_overlay)
        top.addWidget(self.cont_show)
        l2.addLayout(top)
        # Ausgabe passend zur eigenen Maschine/Steuerung
        st = QtWidgets.QHBoxLayout()
        self.cont_dialect = QtWidgets.QComboBox()
        self.cont_dialect.addItem("ISO/DIN: Fanuc, Mazak EIA, DMG MORI (Fanuc/MAPPS), DIN PLUS – Bogen mit R", "iso")
        self.cont_dialect.addItem("Siemens Sinumerik 840D – Bogen mit CR=", "siemens")
        self.cont_dialect.setCurrentIndex(max(0, self.cont_dialect.findData(self.settings.value("cont_dialect", "iso"))))
        self.cont_dialect.currentIndexChanged.connect(self._lathe_opts_changed)
        st.addWidget(QtWidgets.QLabel("Steuerung:"))
        st.addWidget(self.cont_dialect, 1)
        l2.addLayout(st)
        st2 = QtWidgets.QHBoxLayout()
        self.cont_xrad = QtWidgets.QCheckBox("X als Radius")
        self.cont_xrad.setChecked(str(self.settings.value("cont_xrad", "false")).lower() in ("1", "true"))
        self.cont_xrad.setToolTip("Für Steuerungen, die X im Radius programmieren (sonst Durchmesser)")
        self.cont_front = QtWidgets.QCheckBox("Werkzeug vor der Drehmitte (G2/G3 tauschen)")
        self.cont_front.setChecked(str(self.settings.value("cont_front", "false")).lower() in ("1", "true"))
        self.cont_front.setToolTip("Standard: Werkzeug hinter der Drehmitte (Schrägbett) – G2 = im Uhrzeiger "
                                   "bei Blick Z nach rechts, X nach oben.\nHaken setzen bei Werkzeug vor der "
                                   "Drehmitte (z. B. Flachbett, Revolver vorne).")
        for cb in (self.cont_xrad, self.cont_front):
            cb.toggled.connect(self._lathe_opts_changed)
            st2.addWidget(cb)
        st2.addStretch(1)
        l2.addLayout(st2)
        self.cont_table = InfoTable(["Nr", "Element", "Z", "X (Ø)", "R", "Mitte Z", "Mitte X (Ø)", "Drehsinn", "Info"])
        self.cont_table.itemSelectionChanged.connect(lambda: self._lathe_row_select(self.cont_table, self._cont_rows))
        l2.addWidget(self.cont_table, 1)
        bl = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Als CSV speichern …")
        b1.clicked.connect(lambda: self._export_table(self.cont_table, "_Kontur.csv", "Kontur speichern"))
        b2 = QtWidgets.QPushButton("Als DXF speichern …")
        b2.clicked.connect(self.export_contour_dxf)
        b3 = QtWidgets.QPushButton("Als G-Code speichern …")
        b3.setToolTip("Konturprogramm (G0/G1/G2/G3) im gewählten Steuerungsformat – nur Geometrie")
        b3.clicked.connect(self.export_contour_gcode)
        bl.addWidget(b1)
        bl.addWidget(b2)
        bl.addWidget(b3)
        l2.addLayout(bl)
        lab = QtWidgets.QLabel("Endpunkt je Element. Z bezieht sich auf den aktuellen Nullpunkt. "
                               "Steuerung, X-Angabe und Werkzeuglage oben passend zur eigenen Maschine einstellen.")
        lab.setWordWrap(True)
        l2.addWidget(lab)
        tabs.addTab(w2, "Kontur")
        # Einstiche / Freistiche
        w3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(w3)
        l3.setContentsMargins(2, 2, 2, 2)
        self.groove_table = InfoTable(["Art", "Z von", "Z bis", "Breite", "Tiefe", "an Ø", "Grund Ø", "Radien im Grund",
                                       "Hinweis"])
        self.groove_table.itemSelectionChanged.connect(lambda: self._lathe_row_select(self.groove_table,
                                                                                     self._groove_rows))
        l3.addWidget(self.groove_table, 1)
        b = QtWidgets.QPushButton("Als CSV speichern …")
        b.clicked.connect(lambda: self._export_table(self.groove_table, "_Einstiche.csv", "Einstiche speichern"))
        l3.addWidget(b)
        lab = QtWidgets.QLabel("Einstich = Vertiefung zwischen zwei gleich hohen Stellen, Freistich = am Absatz. "
                               "Breite auf Höhe der niedrigeren Kante, Tiefe ab dieser Kante. Zeile anklicken = Flächen zeigen.")
        lab.setWordWrap(True)
        l3.addWidget(lab)
        tabs.addTab(w3, "Einstiche")
        # Werkzeug-Erreichbarkeit
        w4 = QtWidgets.QWidget()
        l4 = QtWidgets.QVBoxLayout(w4)
        l4.setContentsMargins(2, 2, 2, 2)
        top = QtWidgets.QHBoxLayout()
        self.tool_side = QtWidgets.QComboBox()
        self.tool_side.addItems(["Außenkontur", "Innenkontur"])
        self.tool_dir = QtWidgets.QComboBox()
        self.tool_dir.addItems(["Vorschub −Z", "Vorschub +Z"])
        self.tool_ins = QtWidgets.QComboBox()
        for nm, e, kr in core.STD_INSERTS:
            self.tool_ins.addItem(f"{nm}  (κ' {fmt(core.insert_clearance(e, kr), 1)}°)", nm)
        self.tool_ins.setCurrentIndex(max(0, self.tool_ins.findData(self.settings.value("tool_ins", "DNMG 55° κ93°"))))
        for cb in (self.tool_side, self.tool_dir, self.tool_ins):
            cb.currentIndexChanged.connect(self._fill_tool_check)
            top.addWidget(cb)
        l4.addLayout(top)
        self.tool_table = InfoTable(["Nr", "Element", "Z von", "Z bis", "fällt in −Z", "fällt in +Z",
                                     "Platte −Z (mind.)", "Platte +Z (mind.)", "Schneidenradius max"])
        self.tool_table.itemSelectionChanged.connect(lambda: self._lathe_row_select(self.tool_table, self._tool_rows))
        l4.addWidget(self.tool_table, 1)
        bl = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Nicht erreichbare Stellen zeigen")
        b.setToolTip("Wählt alle Flächen aus, die mit der gewählten Platte in der gewählten Richtung nicht gehen")
        b.clicked.connect(self._show_tool_problems)
        bl.addWidget(b)
        b = QtWidgets.QPushButton("Als CSV speichern …")
        b.clicked.connect(lambda: self._export_table(self.tool_table, "_Werkzeug.csv", "Werkzeugprüfung speichern"))
        bl.addWidget(b)
        l4.addLayout(bl)
        self.tool_summary = QtWidgets.QLabel("")
        self.tool_summary.setWordWrap(True)
        l4.addWidget(self.tool_summary)
        lab = QtWidgets.QLabel("„fällt“ = so steil läuft die Kontur in Vorschubrichtung vom Werkzeug weg. Die Platte "
                               "braucht einen Nebeneinstellwinkel κ' = 180° − κr − ε, der größer ist (+2° Sicherheit). "
                               "90° = Planfläche/Absatz: nur aus der Gegenrichtung oder mit Einstechwerkzeug. "
                               "Hohlkehlen begrenzen den Schneidenradius. Rot = mit der gewählten Platte nicht erreichbar.")
        lab.setWordWrap(True)
        l4.addWidget(lab)
        tabs.addTab(w4, "Werkzeug")
        # Fräsmerkmale
        w5 = QtWidgets.QWidget()
        l5 = QtWidgets.QVBoxLayout(w5)
        l5.setContentsMargins(2, 2, 2, 2)
        self.mill_table = InfoTable(["Art", "Breite / SW", "Tiefe ab Ø", "an Ø", "Z von", "Z bis", "Länge", "C°",
                                     "Boden ab Mitte", "Enden"])
        self.mill_table.itemSelectionChanged.connect(lambda: self._lathe_row_select(self.mill_table, self._mill_rows))
        l5.addWidget(self.mill_table, 1)
        b = QtWidgets.QPushButton("Als CSV speichern …")
        b.clicked.connect(lambda: self._export_table(self.mill_table, "_Fraesen.csv", "Fräsmerkmale speichern"))
        l5.addWidget(b)
        lab = QtWidgets.QLabel("Nut: Boden mit zwei Seitenwänden – Breite, Tiefe ab Außendurchmesser (Mitte Nut). "
                               "Schlüsselfläche: zwei gegenüberliegende Flächen – SW. C° = Richtung der Bodennormalen "
                               "(Werkzeug von dort). Zeile anklicken = Flächen zeigen.")
        lab.setWordWrap(True)
        l5.addWidget(lab)
        tabs.addTab(w5, "Fräsen")
        self._groove_rows, self._tool_rows, self._mill_rows = [], [], []
        self._dim_rows, self._cont_rows, self._lathe = [], [], None
        self.contour_items = []
        return tabs

    def _lathe_opts_changed(self, *_):
        self.settings.setValue("cont_dialect", self.cont_dialect.currentData())
        self.settings.setValue("cont_xrad", self.cont_xrad.isChecked())
        self.settings.setValue("cont_front", self.cont_front.isChecked())
        self._fill_lathe()

    def _lathe_ok(self):
        if self.model is None:
            return False
        mo, md = core.main_axis(self.model)
        return abs(abs(self.wcs.dir_to_wcs(md)[2]) - 1) < 1e-6

    def _fill_lathe(self, *_):
        if not hasattr(self, "dim_table"):
            return
        self._dim_rows, self._cont_rows = [], []
        if self.model is None:
            self.dim_table.set_rows([]); self.cont_table.set_rows([])
            self._lathe = None
            self._fill_extra_lathe()
            return
        if not self._lathe_ok():
            self.dim_table.set_rows([("–", "Nullpunkt-Z liegt nicht auf der Längsachse – Reiter Nullpunkt → "
                                            "„Drehmaschine“ wählen", "", "", "", "", "", "")])
            self.cont_table.set_rows([])
            self._lathe = None
            self._fill_extra_lathe()
            return
        try:
            dims = core.dimension_list(self.model, self.wcs)
            self._lathe = core.lathe_contours(self.model)
        except Exception as ex:
            self.dim_table.set_rows([("Fehler", str(ex), "", "", "", "", "", "")])
            return
        flt = self.dim_filter.currentText()
        want = {"Durchmesser": ("Durchmesser",), "Planflächen": ("Planfläche",),
                "Fasen / Kegel": ("Fase", "Kegel"), "Radien": ("Radius",)}.get(flt)
        self._dim_rows = [d for d in dims if want is None or d["typ"] in want]
        V = self.model.tris.reshape(-1, 3)
        zz = (V - self.wcs.origin) @ self.wcs.R[2]
        zmin, zmax = float(zz.min()), float(zz.max())

        def span(a, b):
            a, b = sorted((a, b))
            return fmt(a) if abs(b - a) < 1e-6 else f"{fmt(a)} … {fmt(b)}"
        self.dim_table.set_rows([(d["typ"], d["wert"], fmt(d["z0"]), fmt(d["z1"]),
                                  fmt(d["L"]) if d["L"] else "",
                                  span(d["z0"] - zmin, d["z1"] - zmin), span(zmax - d["z1"], zmax - d["z0"]),
                                  d["info"]) for d in self._dim_rows])
        hi = self.dim_table.horizontalHeaderItem(5)
        if hi is not None:
            hi.setToolTip(f"Abstand von der Stirnfläche bei Z {fmt(zmin)}")
            self.dim_table.horizontalHeaderItem(6).setToolTip(f"Abstand von der Stirnfläche bei Z {fmt(zmax)}")
        if self._lathe:
            elems = self._lathe["outer" if self.cont_sel.currentIndex() == 0 else "inner"]
            if self.cont_rev.isChecked():
                elems = [(s_, not f_) for s_, f_ in reversed(elems)]
            self._cont_rows = core.contour_rows(self.model, self.wcs, elems)
            xr = self.cont_xrad.isChecked()
            k = 0.5 if xr else 1.0
            swap = self.cont_front.isChecked()
            rows = []
            for i, r in enumerate(self._cont_rows):
                d = r.get("dir", "")
                if d and swap:
                    d = "gegen Uhrzeiger (G3)" if "G2" in d else "im Uhrzeiger (G2)"
                    d += " – getauscht"
                rows.append((i, r["elem"], fmt(r["z"]), fmt(r["x"] * k), fmt(r["r"]) if "r" in r else "",
                             fmt(r["cz"]) if "cz" in r else "", fmt(r["cx"] * k) if "cx" in r else "",
                             d, r.get("info", "")))
            self.cont_table.set_rows(rows)
            hx = "X (R)" if xr else "X (Ø)"
            self.cont_table.setHorizontalHeaderLabels(["Nr", "Element", "Z", hx, "R", "Mitte Z",
                                                       "Mitte " + hx, "Drehsinn", "Info"])
        self._fill_extra_lathe()
        self._show_contour_overlay()

    def _fill_extra_lathe(self):
        """Einstiche, Werkzeugprüfung und Fräsmerkmale."""
        self._groove_rows, self._mill_rows = [], []
        if self.model is None or not self._lathe:
            for t in (self.groove_table, self.mill_table, self.tool_table):
                t.set_rows([])
            return
        try:
            g = core.grooves(self.model, self.wcs, self._lathe["outer"]) + \
                core.grooves(self.model, self.wcs, self._lathe["inner"], inner=True)
        except Exception as ex:
            g = []
            self.status.showMessage(f"Einstiche: {ex}")
        for x in g:
            if x["z0"] > x["z1"]:
                x["z0"], x["z1"] = x["z1"], x["z0"]
        g.sort(key=lambda x: x["z0"])
        self._groove_rows = g
        self.groove_table.set_rows([(x["kind"], fmt(x["z0"]), fmt(x["z1"]), fmt(x["width"]), fmt(x["depth"]),
                                     fmt(x["dia"]), fmt(x["bottom"]),
                                     ", ".join("R " + fmt(r) for r in x["radii"]), x["note"]) for x in g]
                                   or [("–", "keine Einstiche gefunden", "", "", "", "", "", "", "")])
        try:
            mills = core.milling_features(self.model, self.wcs)
        except Exception as ex:
            mills = []
            self.status.showMessage(f"Fräsmerkmale: {ex}")
        self._mill_rows = mills
        rows = []
        for f in mills:
            c = fmt(f["c"], 2) + (f" / {fmt(f['c2'], 2)}" if "c2" in f else "")
            rows.append((f["kind"], ("SW " if f["kind"] == "Schlüsselfläche" else "") + (fmt(f["width"]) if f["width"] else "–"),
                         fmt(f["depth"]), fmt(f["dia"]), fmt(f["z0"]), fmt(f["z1"]), fmt(f["length"]), c,
                         fmt(f["h"]), f["ends"]))
        self.mill_table.set_rows(rows or [("–", "keine Fräsmerkmale gefunden", "", "", "", "", "", "", "", "")])
        self._fill_tool_check()

    def _fill_tool_check(self, *_):
        if not hasattr(self, "tool_table"):
            return
        self._tool_rows = []
        if self.model is None or not self._lathe:
            self.tool_table.set_rows([])
            self.tool_summary.setText("")
            return
        self.settings.setValue("tool_ins", self.tool_ins.currentData())
        inner = self.tool_side.currentIndex() == 1
        elems = self._lathe["inner" if inner else "outer"]
        try:
            rows = core.tool_check(self.model, self.wcs, elems, inner=inner)
        except Exception as ex:
            self.tool_table.set_rows([("Fehler", str(ex))])
            return
        self._tool_rows = rows

        def best(lst):
            return lst[0] if lst else "keine – Gegenrichtung/Einstechen"
        self.tool_table.set_rows([(r["nr"], r["elem"], fmt(r["z0"]), fmt(r["z1"]), fmt(r["dive_m"], 1) + "°",
                                   fmt(r["dive_p"], 1) + "°", best(r["tools_m"]), best(r["tools_p"]),
                                   ("≤ R " + fmt(r["rmax"])) if r["rmax"] else "") for r in rows])
        key = "tools_m" if self.tool_dir.currentIndex() == 0 else "tools_p"
        nm = self.tool_ins.currentData()
        bad = [i for i, r in enumerate(rows) if nm not in r[key]]
        red = QtGui.QColor(255, 205, 205)
        for i in bad:
            for c in range(self.tool_table.columnCount()):
                it = self.tool_table.item(i, c)
                if it is not None:
                    it.setBackground(red)
        rmax = [r["rmax"] for r in rows if r["rmax"]]
        txt = (f"{nm}, {self.tool_dir.currentText()}: " +
               ("ganze Kontur erreichbar." if not bad else
                f"nicht erreichbar: Element {', '.join(str(rows[i]['nr']) for i in bad)}."))
        if rmax:
            txt += f"  Schneidenradius höchstens R {fmt(min(rmax))} (Hohlkehle)."
        gw = [x["width"] for x in self._groove_rows if not x["kind"].endswith("innen")] if self._groove_rows else []
        if gw and not inner:
            txt += f"  Einstechbreite höchstens {fmt(min(gw))} mm."
        self.tool_summary.setText(txt)

    def _show_tool_problems(self):
        if not self._tool_rows:
            return
        key = "tools_m" if self.tool_dir.currentIndex() == 0 else "tools_p"
        nm = self.tool_ins.currentData()
        faces = [r["face"] for r in self._tool_rows if nm not in r[key]]
        if not faces:
            self.status.showMessage("Alles erreichbar – nichts zu markieren.")
            return
        self.sel = [core.Entity(self.model, "face", f) for f in faces]
        self._keep_tab = True
        try:
            self._refresh_selection()
        finally:
            self._keep_tab = False
        self.status.showMessage(f"{len(faces)} Fläche(n) mit {nm} in {self.tool_dir.currentText()} nicht erreichbar.")

    def _lathe_row_select(self, table, rows):
        sel = table.selectionModel().selectedRows()
        if not sel or self.model is None:
            return
        ents = []
        for r in sel:
            if r.row() >= len(rows):
                continue
            row = rows[r.row()]
            for fi in row.get("faces") or ([row["face"]] if row.get("face") is not None else []):
                ents.append(core.Entity(self.model, "face", fi))
        if ents:
            self.sel = ents[:2] if len(ents) <= 2 else ents
            self._keep_tab = True
            try:
                self._refresh_selection()
            finally:
                self._keep_tab = False

    def _contour_polyline(self, elems):
        mo, md = core.main_axis(self.model)
        u = self.wcs.R[0] - (self.wcs.R[0] @ md) * md
        u = u / (np.linalg.norm(u) or 1.0)
        P = lambda a, r: mo + a * md + r * u
        pts = []
        for s_, f_ in elems:
            q0, q1 = core._oriented(s_, f_)
            if s_["kind"] == "ARC":
                c = s_["c"]
                a0 = math.atan2(q0[1] - c[1], q0[0] - c[0])
                a1 = math.atan2(q1[1] - c[1], q1[0] - c[0])
                am = math.atan2(s_["mid"][1] - c[1], s_["mid"][0] - c[0])
                da = (a1 - a0) % (2 * math.pi)
                if (am - a0) % (2 * math.pi) > da:
                    da -= 2 * math.pi
                for t in np.linspace(0, da, 24):
                    pts.append(P(c[0] + s_["r"] * math.cos(a0 + t), c[1] + s_["r"] * math.sin(a0 + t)))
            else:
                pts += [P(*q0), P(*q1)]
        return np.array(pts)

    def _show_contour_overlay(self, *_):
        for it in getattr(self, "contour_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.contour_items = []
        if self.model is None or not getattr(self, "cont_show", None) or not self.cont_show.isChecked():
            return
        if not self._lathe:
            return
        for key, col in (("outer", (0.10, 0.45, 0.95, 1.0)), ("inner", (0.05, 0.65, 0.40, 1.0))):
            el = self._lathe.get(key) or []
            if not el:
                continue
            pts = self._contour_polyline(el)
            it = gl.GLLinePlotItem(pos=pts, mode="line_strip", color=col, width=3.0, antialias=True,
                                   glOptions=GL_ONTOP)
            it.setDepthValue(25)
            self.view.addItem(it)
            self.contour_items.append(it)

    def _export_table(self, table, suffix, title):
        if self.model is None or table.rowCount() == 0:
            return
        base = os.path.splitext(self.model.path)[0] + suffix
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, base, "CSV (*.csv)")
        if not p:
            return
        hdr = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(f"# {os.path.basename(self.model.path)} – Nullpunkt: {self.wcs.name}\n")
            fh.write(";".join(hdr) + "\n")
            for r in range(table.rowCount()):
                fh.write(";".join((table.item(r, c).text() if table.item(r, c) else "")
                                  for c in range(len(hdr))) + "\n")
        self.status.showMessage(f"Gespeichert: {p}")

    def export_contour_gcode(self):
        if self.model is None or not self._cont_rows:
            return
        dia = self.cont_dialect.currentData()
        side = "Aussen" if self.cont_sel.currentIndex() == 0 else "Innen"
        base = os.path.splitext(self.model.path)[0] + f"_Kontur_{side}" + (".mpf" if dia == "siemens" else ".nc")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kontur als G-Code speichern", base,
                                                     "G-Code (*.nc *.mpf *.spf *.eia *.txt);;Alle Dateien (*)")
        if not p:
            return
        txt = core.contour_gcode(self._cont_rows, dia, self.cont_xrad.isChecked(), self.cont_front.isChecked(),
                                 name=f"{side}kontur {os.path.basename(self.model.path)}")
        with open(p, "w", encoding="ascii", errors="replace", newline="\r\n") as fh:
            fh.write(txt)
        self.status.showMessage(f"G-Code gespeichert: {p}")

    def export_contour_dxf(self):
        """Halbschnitt-Kontur als DXF (R12): X-Achse = Z, Y-Achse = Radius."""
        if self.model is None or not self._lathe:
            return
        base = os.path.splitext(self.model.path)[0] + "_Kontur.dxf"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kontur als DXF speichern", base, "DXF (*.dxf)")
        if not p:
            return
        mo, md = core.main_axis(self.model)
        Z = lambda a: float(self.wcs.to_wcs(mo + a * md)[2])
        zs = 1.0 if self.wcs.dir_to_wcs(md)[2] >= 0 else -1.0
        ent = []
        for key, layer in (("outer", "AUSSEN"), ("inner", "INNEN")):
            for s_, f_ in self._lathe.get(key) or []:
                q0, q1 = core._oriented(s_, f_)
                if s_["kind"] == "ARC":
                    c = s_["c"]
                    cz, cr = Z(c[0]), c[1]
                    ang = lambda q: math.degrees(math.atan2(q[1] - cr, Z(q[0]) - cz))
                    a0, a1, am = ang(q0), ang(q1), ang(s_["mid"])
                    # DXF-Bögen laufen gegen den Uhrzeigersinn von Start nach Ende
                    ccw_ok = (am - a0) % 360 <= (a1 - a0) % 360
                    st, en = (a0, a1) if ccw_ok else (a1, a0)
                    ent.append(f"0\nARC\n8\n{layer}\n10\n{cz:.6f}\n20\n{cr:.6f}\n30\n0\n40\n{s_['r']:.6f}"
                               f"\n50\n{st % 360:.6f}\n51\n{en % 360:.6f}")
                else:
                    ent.append(f"0\nLINE\n8\n{layer}\n10\n{Z(q0[0]):.6f}\n20\n{q0[1]:.6f}\n30\n0"
                               f"\n11\n{Z(q1[0]):.6f}\n21\n{q1[1]:.6f}\n31\n0")
        zmin = min(Z(s_["p0"][0]) for s_ in self._lathe["segs"])
        zmax = max(Z(s_["p1"][0]) for s_ in self._lathe["segs"])
        ent.append(f"0\nLINE\n8\nACHSE\n10\n{min(zmin, zmax) - 5:.6f}\n20\n0\n30\n0\n11\n{max(zmin, zmax) + 5:.6f}"
                   f"\n21\n0\n31\n0")
        txt = "0\nSECTION\n2\nENTITIES\n" + "\n".join(ent) + "\n0\nENDSEC\n0\nEOF\n"
        with open(p, "w", encoding="ascii", errors="replace") as fh:
            fh.write(txt)
        self.status.showMessage(f"DXF gespeichert: {p}  (X-Achse = Z, Y-Achse = Radius)")

    def on_hole_row(self):
        rows = self.hole_table.selectionModel().selectedRows()
        if not rows or self.model is None:
            return
        sel = []
        for r in rows:
            h = self.holes[r.row()]
            # größten Zylinder der Bohrung als Stellvertreter
            cyl = [i for i in h.faces if self.model.faces[i].ftype == "CYLINDER"]
            cyl.sort(key=lambda i: -self.model.faces[i].surf.radius)
            sel += [core.Entity(self.model, "face", i) for i in cyl[:1]]
        self.sel = sel
        self._refresh_selection()
        self.tabs_right.setCurrentIndex(2)


def main():
    fmt_gl = QtGui.QSurfaceFormat()
    fmt_gl.setSamples(8)
    fmt_gl.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt_gl)
    QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    base = os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
    for ico in (os.path.join(getattr(sys, "_MEIPASS", base), "S3D_Viewer.ico"), os.path.join(base, "S3D_Viewer.ico")):
        if os.path.isfile(ico):
            app.setWindowIcon(QtGui.QIcon(ico))
            break
    try:   # eigenes Taskleisten-Symbol unter Windows
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("privat.S3DViewer")
    except Exception:
        pass
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.load(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
