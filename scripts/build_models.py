"""Parametric 3D models for monuments that have no usable scanned/hand-made GLB.

    python scripts/build_models.py            # builds every monument listed in BUILDERS
    python scripts/build_models.py konark-sun-temple

Each model is assembled from primitives (frustums, revolved domes, tori, arcades) using
*documented* overall dimensions (heights, diameters, tier counts, orientation of the main
entrance). They are architectural massing models: proportions and layout are right, carved
detail is not. Every GLB carries asset.extras {source: "parametric", ...} and the viewer
labels it "parametric approximation" — it is never presented as a survey or scan.

Convention: metres; built Z-up with +Y = north, then rotated to glTF Y-up with north = -Z
(what core/viewer3d.py expects).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "models3d"

# ------------------------------------------------------------------ palette
PAL = {
    "redstone": (166, 74, 52), "redstone_d": (128, 54, 38), "buff": (214, 190, 150), "buff_d": (178, 152, 112),
    "granite": (150, 146, 138), "granite_d": (112, 108, 102), "granite_l": (184, 178, 168),
    "stucco": (226, 220, 206), "marble": (238, 236, 230), "basalt": (96, 88, 80), "basalt_d": (62, 56, 50),
    "khondalite": (128, 104, 84), "khondalite_d": (98, 78, 62), "void": (34, 30, 28), "gold": (212, 168, 72),
    "water": (70, 112, 128), "ground": (196, 186, 164), "brick": (176, 120, 88), "grass": (126, 140, 96),
    "poly_r": (190, 74, 60), "poly_b": (64, 104, 160), "poly_y": (222, 180, 70), "poly_g": (88, 140, 96),
    "poly_w": (232, 226, 212),
}


class Model:
    def __init__(self):
        self.parts: dict[str, list[trimesh.Trimesh]] = {}

    def add(self, mesh: trimesh.Trimesh, mat: str) -> trimesh.Trimesh:
        self.parts.setdefault(mat, []).append(mesh)
        return mesh

    # ---- primitives (all positioned by base-centre unless stated) ----
    def box(self, w, d, h, x=0, y=0, z=0, mat="stucco", rot=0.0):
        m = trimesh.creation.box(extents=[w, d, h])
        if rot:
            m.apply_transform(trimesh.transformations.rotation_matrix(rot, [0, 0, 1]))
        m.apply_translation([x, y, z + h / 2])
        return self.add(m, mat)

    def frustum(self, w0, d0, w1, d1, h, x=0, y=0, z=0, mat="stucco"):
        p = []
        for (w, d, zz) in ((w0, d0, 0), (w1, d1, h)):
            p += [[sx * w / 2, sy * d / 2, zz] for sx in (-1, 1) for sy in (-1, 1)]
        m = trimesh.convex.convex_hull(np.array(p))
        m.apply_translation([x, y, z])
        return self.add(m, mat)

    def cyl(self, r, h, x=0, y=0, z=0, mat="stucco", sections=32, axis="z"):
        """axis='z': base-centred vertical. axis='x'/'y': horizontal, centred at (x, y, z)."""
        m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
        if axis == "z":
            m.apply_translation([x, y, z + h / 2])
        else:
            m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0] if axis == "x" else [1, 0, 0]))
            m.apply_translation([x, y, z])
        return self.add(m, mat)

    def cone(self, r0, r1, h, x=0, y=0, z=0, mat="stucco", sections=32):
        return self.revolve([(r0, 0), (r1, h)], x, y, z, mat, sections, close=True)

    def revolve(self, prof, x=0, y=0, z=0, mat="stucco", sections=40, close=True):
        pts = [(0.0, prof[0][1])] + list(prof) + ([(0.0, prof[-1][1])] if close else [])
        m = trimesh.creation.revolve(np.array(pts, float), sections=sections)
        m.apply_translation([x, y, z])
        return self.add(m, mat)

    def dome(self, r, x=0, y=0, z=0, mat="stucco", rise=1.0, sections=40):
        prof = [(r * math.cos(t), r * rise * math.sin(t)) for t in np.linspace(0, math.pi / 2, 14)]
        prof[-1] = (0.0, r * rise)
        return self.revolve(prof, x, y, z, mat, sections)

    def onion(self, r, x=0, y=0, z=0, mat="stucco", h=None, sections=40):
        h = h or r * 1.9
        prof = [(r * 0.82, 0), (r * 1.0, h * 0.18), (r * 1.04, h * 0.33), (r * 0.95, h * 0.5), (r * 0.72, h * 0.66),
                (r * 0.42, h * 0.8), (r * 0.16, h * 0.92), (0.0, h)]
        self.revolve(prof, x, y, z, mat, sections)
        self.cone(r * 0.06, r * 0.01, h * 0.35, x, y, z + h * 0.95, "gold", 12)

    def torus(self, R, r, x=0, y=0, z=0, mat="stucco", axis="x"):
        m = trimesh.creation.torus(major_radius=R, minor_radius=r, major_sections=48, minor_sections=10)
        if axis == "x":
            m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        elif axis == "y":
            m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        m.apply_translation([x, y, z])
        return self.add(m, mat)

    # ---- composites ----
    def chhatri(self, r, x, y, z, mat="stucco", dome_mat=None, h=None):
        h = h or r * 1.6
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.box(r * .22, r * .22, h, x + sx * r * .7, y + sy * r * .7, z, mat)
        self.box(r * 2.1, r * 2.1, r * .25, x, y, z + h, mat)
        self.dome(r * .9, x, y, z + h + r * .25, dome_mat or mat)
        self.cone(r * .08, 0.01, r * .6, x, y, z + h + r * 1.1, "gold", 8)

    def arcade(self, length, h, x, y, z, face, n, depth=0.25, mat="void", w_frac=0.6):
        """Dark arch openings on a wall face. face: 'n','s','e','w'."""
        step = length / n
        for i in range(n):
            o = -length / 2 + step * (i + .5)
            w = step * w_frac
            if face in ("n", "s"):
                sy = 1 if face == "n" else -1
                self.box(w, depth, h, x + o, y + sy * depth / 2, z, mat)
                self.cyl(w / 2, depth, x + o, y + sy * depth / 2, z + h, mat, 16, axis="y")
            else:
                sx = 1 if face == "e" else -1
                self.box(depth, w, h, x + sx * depth / 2, y + o, z, mat)
                self.cyl(w / 2, depth, x + sx * depth / 2, y + o, z + h, mat, 16, axis="x")

    def crenels(self, length, x, y, z, along="x", size=0.8, mat="redstone"):
        n = int(length / (size * 2))
        for i in range(n):
            o = -length / 2 + (i + .5) * length / n
            if along == "x":
                self.box(size, size * .8, size * 1.1, x + o, y, z, mat)
            else:
                self.box(size * .8, size, size * 1.1, x, y + o, z, mat)

    def tiered(self, w, d, h, tiers, x=0, y=0, z=0, mats=("granite",), shrink=0.72, ratio=None):
        """Stepped pyramidal tower (vimana / gopuram). Returns top (z, w, d)."""
        th = h / tiers
        for i in range(tiers):
            f0 = 1 - (1 - shrink) * i / tiers
            f1 = 1 - (1 - shrink) * (i + 1) / tiers
            mat = mats[i % len(mats)]
            self.frustum(w * f0, d * f0, w * f1 * .96, d * f1 * .96, th * .78, x, y, z + i * th, mat)
            self.box(w * f1 * 1.0, d * f1 * 1.0, th * .22, x, y, z + i * th + th * .78, mat)
            # miniature pavilions along the ledge
            k = max(2, int(3 + (tiers - i) / 2))
            for j in range(k):
                o = -w * f1 / 2 + (j + .5) * w * f1 / k
                for sy in (-1, 1):
                    self.box(w * f1 / k * .45, th * .3, th * .28, x + o, y + sy * d * f1 / 2, z + (i + 1) * th, mat)
        return z + h, w * shrink, d * shrink

    def ground(self, w, d, mat="ground"):
        self.box(w, d, 0.4, 0, 0, -0.4, mat)

    # ---- export ----
    def export(self, path: Path, extras: dict, yaw_deg: float = 0.0):
        scene = trimesh.Scene()
        for mat, meshes in self.parts.items():
            m = trimesh.util.concatenate(meshes)
            c = PAL[mat]
            rough = 0.35 if mat in ("gold", "water", "marble") else 0.88
            metal = 0.7 if mat == "gold" else 0.0
            lin = [((v / 255 + 0.055) / 1.055) ** 2.4 for v in c]          # glTF colours are linear, palette is sRGB
            m.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
                name=mat, baseColorFactor=[*lin, 1.0], roughnessFactor=rough, metallicFactor=metal))
            if yaw_deg:  # orient the main entrance to its real compass direction
                m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw_deg), [0, 0, 1]))
            m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))  # Z-up → Y-up, +Y→-Z
            scene.add_geometry(m, node_name=mat, geom_name=mat)
        path.write_bytes(_inject_extras(scene.export(file_type="glb"), extras))


def _inject_extras(glb: bytes, extras: dict) -> bytes:
    """Write provenance into asset.extras of the GLB JSON chunk."""
    import struct
    jlen = struct.unpack("<I", glb[12:16])[0]
    js = json.loads(glb[20:20 + jlen])
    js.setdefault("asset", {})["extras"] = extras
    js["asset"]["generator"] = "ChronicleHeritage parametric builder"
    new = json.dumps(js, separators=(",", ":")).encode()
    new += b" " * ((4 - len(new) % 4) % 4)
    rest = glb[20 + jlen:]
    body = struct.pack("<I", len(new)) + b"JSON" + new + rest
    return glb[:8] + struct.pack("<I", 12 + len(body)) + body


# ====================================================================== monuments
def golconda(M: Model):
    """Granite hill fort: fortified hill, concentric curtain walls, Bala Hisar pavilion on the summit."""
    M.ground(260, 260)
    rng = np.random.default_rng(7)
    pts = []
    for r, z in ((95, 0), (80, 12), (62, 30), (44, 46), (26, 58), (12, 62)):
        for a in np.linspace(0, 2 * math.pi, 22, endpoint=False):
            rr = r * (1 + rng.uniform(-.08, .08))
            pts.append([rr * math.cos(a), rr * math.sin(a) * .85, z])
    M.add(trimesh.convex.convex_hull(np.array(pts)), "granite")
    for i in range(10):  # boulders
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(30, 80)
        M.add(trimesh.creation.icosphere(1, rng.uniform(4, 9)).apply_translation(
            [r * math.cos(a), r * math.sin(a) * .85, 60 - r * .7]), "granite_d")
    for r, z, h in ((112, 0, 9), (70, 26, 7)):  # curtain walls with bastions
        n = 28
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            p0 = np.array([r * math.cos(a0), r * math.sin(a0) * .85])
            p1 = np.array([r * math.cos(a1), r * math.sin(a1) * .85])
            mid, L = (p0 + p1) / 2, np.linalg.norm(p1 - p0)
            ang = math.atan2(*(p1 - p0)[::-1])
            M.box(L, 2.6, h, mid[0], mid[1], z, "granite_l", rot=ang)
            if k % 4 == 0:
                M.cyl(5, h + 3, p0[0], p0[1], z, "granite_l", 16)
    # Fateh Darwaza (east gate)
    M.box(16, 10, 16, 112, 0, 0, "granite_l")
    M.arcade(8, 9, 117.1, 0, 0, "e", 1, depth=.6)
    # Bala Hisar summit pavilion: two-storey arcaded block + small domes
    z = 62
    M.box(30, 18, 9, 0, 0, z, "stucco")
    M.box(26, 14, 7, 0, 0, z + 9, "stucco")
    M.arcade(26, 5, 0, -9.1, z + 1, "s", 7, depth=.5)
    M.arcade(22, 4, 0, -7.1, z + 10, "s", 6, depth=.5)
    for sx in (-1, 1):
        M.dome(3, sx * 10, 0, z + 16, "stucco")
    M.box(8, 8, 4, 30, 18, 52, "stucco")  # Baradari-like pavilion on slope
    M.dome(2.5, 30, 18, 56, "stucco")


def qutb_shahi(M: Model):
    """Tomb of Sultan Muhammad Quli Qutb Shah: raised terrace, arcaded cube, bulbous dome, corner minarets."""
    M.ground(120, 120, "grass")
    M.box(65, 65, 4, mat="granite")             # terrace
    M.box(45, 45, 2, 0, 0, 4, "granite_l")
    s, h = 32, 18
    M.box(s, s, h, 0, 0, 6, "granite_l")
    for f, (x, y) in {"n": (0, s / 2), "s": (0, -s / 2), "e": (s / 2, 0), "w": (-s / 2, 0)}.items():
        M.arcade(s - 4, 9, x, y, 7, f, 5, depth=.6)
    M.box(s + 2, s + 2, 1.4, 0, 0, 6 + h, "stucco")                 # parapet
    for sx in (-1, 1):
        for sy in (-1, 1):
            M.cyl(1.4, 8, sx * s / 2, sy * s / 2, 6 + h, "stucco", 16)
            M.onion(1.5, sx * s / 2, sy * s / 2, 14 + h, "stucco")
    M.cyl(11, 6, 0, 0, 6 + h + 1.4, "granite_l", 40)                  # drum with petal band
    M.cyl(12, 1.2, 0, 0, 6 + h + 6.6, "stucco", 40)
    M.onion(12.5, 0, 0, 6 + h + 7.8, "granite_l", h=22)


def red_fort(M: Model):
    """Lahori Gate (west) with flanking octagonal towers and crenellated curtain wall."""
    M.ground(240, 120)
    M.box(220, 30, 1.5, 0, -20, 0, "water")                            # dry moat
    L, H = 220, 18
    for side in (-1, 1):
        M.box(L / 2 - 22, 12, H, side * (L / 4 + 11), 0, 0, "redstone")
        M.crenels(L / 2 - 22, side * (L / 4 + 11), -5.6, H, "x", 1.2)
        for k in range(3):
            x = side * (30 + k * 30)
            M.cyl(7, H + 4, x, -2, 0, "redstone", 8)
            M.chhatri(3.2, x, -2, H + 4, "redstone", "marble")
    # gate block
    M.box(30, 20, 26, 0, 0, 0, "redstone")
    M.box(14, 1.2, 15, 0, -10.3, 0, "redstone_d")
    M.arcade(10, 11, 0, -10.9, 0, "s", 1, depth=.6)
    M.arcade(26, 3, 0, -10.2, 19, "s", 7, depth=.4, w_frac=.5)
    M.crenels(30, 0, -9.6, 26, "x", 1.1)
    for sx in (-1, 1):                                                   # octagonal flanking towers
        M.cyl(5.5, 30, sx * 18, -6, 0, "redstone", 8)
        M.chhatri(4, sx * 18, -6, 30, "redstone", "marble")
    for k in range(7):                                                   # row of small chhatris over gate
        M.chhatri(1.4, -9 + k * 3, -8, 26 + 1.3, "marble", "marble")
    # barbican in front of gate
    M.box(40, 6, 12, 0, -28, 0, "redstone")
    M.crenels(40, 0, -30, 12, "x", 1)


def hampi(M: Model):
    """Virupaksha Temple: ~50 m east gopuram (9 tiers), inner courtyard, smaller inner gopuram, main vimana."""
    M.ground(200, 120, "ground")
    M.box(170, 70, 6, 0, 0, 0, "granite")                                # enclosure walls (outline)
    M.box(166, 66, 6.2, 0, 0, 0, "ground")
    # east gopuram (main): granite base 2 storeys + brick/stucco 9 tiers
    gx = 85
    M.box(12, 30, 10, gx, 0, 0, "granite")
    M.arcade(8, 8, gx + 6, 0, 0, "e", 1, depth=.6)
    top, w, d = M.tiered(12, 30, 38, 9, gx, 0, 10, ("stucco", "buff"), shrink=.55)
    M.box(w * .7, d * 1.0, 3, gx, 0, top, "buff")                         # wagon-vault crown
    M.cyl(1.6, d * 1.05, gx, 0, top + 3, "buff", 20, axis="y")
    for k in range(9):
        M.cone(.5, .05, 2.4, gx, -d / 2 + (k + .5) * d / 9, top + 4.4, "gold", 8)
    # inner gopuram
    top, w, d = M.tiered(9, 18, 18, 5, 35, 0, 6, ("stucco", "buff"), shrink=.6)
    M.box(w * .7, d, 2, 35, 0, top, "buff")
    # mandapa + sanctum vimana
    M.box(36, 26, 6, -10, 0, 0, "granite")
    for i in range(6):
        for j in range(4):
            M.cyl(.45, 5, -24 + i * 5.5, -9 + j * 6, 6, "granite_l", 8)
    M.box(38, 28, 1, -10, 0, 11, "granite")
    top, w, d = M.tiered(12, 12, 14, 4, -40, 0, 0, ("granite", "stucco"), shrink=.5)
    M.dome(w * .55, -40, 0, top, "stucco", rise=.9)
    M.cone(.4, .05, 2.5, -40, 0, top + w * .5, "gold", 8)


def ajanta(M: Model):
    """Horseshoe-shaped basalt gorge of the Waghur river; ~30 rock-cut caves on the concave scarp face."""
    M.ground(320, 260, "grass")
    M.box(300, 16, .8, 0, -95, 0, "water")                      # river leaving the gorge (south)
    R, H, n = 70, 40, 46
    a0, a1 = math.radians(200), math.radians(-20)                # arc opens to the south
    for k in range(n):
        a = a0 + (a1 - a0) * (k + .5) / n
        da = abs(a1 - a0) / n
        # solid plateau segment from the scarp face outward (gives a hollow horseshoe without booleans)
        r_in, r_out = R, R + 70
        pts = []
        for rr in (r_in, r_out):
            for aa in (a - da / 2 - .002, a + da / 2 + .002):
                for zz in (0, H):
                    pts.append([rr * math.cos(aa), rr * math.sin(aa), zz])
        M.add(trimesh.convex.convex_hull(np.array(pts)), "basalt")
        M.add(trimesh.convex.convex_hull(np.array([[p[0], p[1], H if p[2] else H + .6] for p in pts])), "grass")
    for k in range(40):                                          # narrow river following the bend
        a = a0 + (a1 - a0) * (k + .5) / 40
        M.box(9, 10, .6, (R - 30) * math.cos(a), (R - 30) * math.sin(a), 0, "water", rot=a + math.pi / 2)
    # cave fronts on the concave face
    for k in range(28):
        a = a0 + (a1 - a0) * (k + .5) / 28
        ux, uy = math.cos(a), math.sin(a)                       # outward normal; caves face -u
        x, y = (R - .6) * ux, (R - .6) * uy
        rot = a + math.pi / 2
        z0 = 10 + (k % 4) * 1.5
        M.box(12, 3, .8, (R - 1.5) * ux, (R - 1.5) * uy, z0 - .8, "basalt_d", rot=rot)       # ledge path
        if k in (8, 17, 24):                                    # chaitya halls with horseshoe window
            M.box(9, 1.2, 12, x, y, z0, "basalt_d", rot=rot)
            M.box(3.2, 1.6, 6, x - ux * .3, y - uy * .3, z0, "void", rot=rot)
            M.box(4.5, 1.6, 4.5, x - ux * .3, y - uy * .3, z0 + 7, "void", rot=rot)
        else:                                                    # viharas: dark verandah between pillars
            M.box(10, 1.6, 5.5, x, y, z0, "void", rot=rot)
            for pnum in range(4):
                o = -3.6 + pnum * 2.4
                M.box(.7, .7, 5.5, x - ux * .9 + math.cos(rot) * o, y - uy * .9 + math.sin(rot) * o, z0, "basalt", rot=rot)


def ellora(M: Model):
    """Kailasa Temple (Cave 16): monolith carved out of a pit; entrance on the west, ~32 m vimana."""
    M.ground(200, 160, "grass")
    W, D, H = 90, 50, 32                                       # pit (rock walls on N, S, E; open-ish west screen)
    M.box(W + 40, 20, H, 0, D / 2 + 10, 0, "basalt")
    M.box(W + 40, 20, H, 0, -D / 2 - 10, 0, "basalt")
    M.box(20, D, H, W / 2 + 10, 0, 0, "basalt")
    M.box(W + 40, 2, 1, 0, D / 2 + 1, H, "grass")
    for f, yy in (("s", D / 2), ("n", -D / 2)):              # cloister cells cut into side walls
        M.arcade(W - 10, 5, 5, yy - (0.3 if f == "s" else -0.3), 4, f, 12, depth=.6)
    M.box(D * .9, 1, 1, 0, 0, 0, "basalt")  # pit floor marker
    # west gopuram screen wall + gateway
    M.box(8, D, 14, -W / 2 + 4, 0, 0, "basalt")
    M.arcade(8, 8, -W / 2, 0, 0, "w", 1, depth=.6)
    M.box(6, 8, 4, -W / 2 + 4, 0, 14, "basalt_d")
    # Nandi mandapa + bridge
    M.box(9, 9, 10, -22, 0, 0, "basalt")
    M.tiered(9, 9, 5, 2, -22, 0, 10, ("basalt_d",), shrink=.6)
    for sy in (-1, 1):
        M.cyl(1.0, 15, -22, sy * 13, 0, "basalt_d", 16)        # dhvajastambhas
        M.box(6, 3.5, 3.4, -22, sy * 17, 0, "basalt_d")        # life-size elephants (blocks)
    # main temple: high plinth with elephant frieze, mandapa, vimana
    M.box(52, 30, 8, 12, 0, 0, "basalt")
    for k in range(14):
        M.box(2.6, 1.2, 3.5, -12 + k * 3.6, -15.5, 1, "basalt_d")
        M.box(2.6, 1.2, 3.5, -12 + k * 3.6, 15.5, 1, "basalt_d")
    M.box(22, 22, 9, 0, 0, 8, "basalt")                         # mandapa
    M.arcade(18, 6, 0, -11.1, 8, "s", 4, depth=.4)
    M.arcade(18, 6, 0, 11.1, 8, "n", 4, depth=.4)
    M.box(24, 24, 1.2, 0, 0, 17, "basalt_d")
    top, w, d = M.tiered(18, 18, 18, 4, 24, 0, 8, ("basalt", "basalt_d"), shrink=.45)
    M.revolve([(w * .75, 0), (w * .8, 1.5), (w * .55, 3.5), (0, 4.2)], 24, 0, top, "basalt", 8)
    M.cone(.4, .05, 2.5, 24, 0, top + 4.2, "basalt_d", 8)


def konark(M: Model):
    """Konark Sun Temple: chariot plinth with 24 wheels, surviving pyramidal jagamohana (~39 m), ruined deul, 7 horses (east)."""
    M.ground(160, 120, "grass")
    L, Wd = 74, 40
    M.box(L, Wd, 5, 0, 0, 0, "khondalite_d")
    M.box(L - 4, Wd - 4, 3, 0, 0, 5, "khondalite")
    for k in range(12):                                         # 12 pairs of wheels (~3 m dia)
        x = -L / 2 + 5 + k * (L - 10) / 11
        for sy in (-1, 1):
            y = sy * (Wd / 2 + .5)
            M.torus(2.9, .32, x, y, 3.2, "khondalite_d", axis="y")
            M.cyl(.55, 1.2, x, y, 3.2, "khondalite", 12, axis="y")
            for s in range(8):
                a = s * math.pi / 8
                M.box(.22, .3, 5.6, x, y, .4, "khondalite", rot=0).apply_transform(
                    trimesh.transformations.rotation_matrix(a, [0, 1, 0], [x, y, 3.2]))
    # jagamohana (assembly hall) – three tiers of pidhas
    jx = 14
    M.box(30, 30, 12, jx, 0, 8, "khondalite")
    M.arcade(10, 8, jx + 15, 0, 8, "e", 1, depth=.6)
    z = 20
    for t, (w, hh) in enumerate(((32, 3), (28, 3), (24, 3), (20, 3), (17, 2.6), (14, 2.6))):
        M.frustum(w, w, w - 2.5, w - 2.5, hh, jx, 0, z, "khondalite" if t % 2 == 0 else "khondalite_d")
        z += hh + .5
    M.frustum(13, 13, 9, 9, 3, jx, 0, z, "khondalite")
    M.revolve([(5.6, 0), (6, 1.4), (4, 3), (0, 3.4)], jx, 0, z + 3, "khondalite", 16)    # amalaka
    M.cone(.6, .1, 3, jx, 0, z + 6.4, "khondalite_d", 8)
    # ruined deul (sanctum) – collapsed stub
    M.box(26, 26, 14, -20, 0, 8, "khondalite_d")
    M.frustum(24, 24, 18, 18, 5, -20, 0, 22, "khondalite")
    # natya mandapa platform (east) and 7 horses
    M.box(18, 18, 6, 52, 0, 0, "khondalite")
    for k in range(7):
        y = -9 + k * 3
        M.box(4.2, 1.3, 2.6, L / 2 + 6, y, 2.4, "khondalite")
        M.box(1.4, 1.1, 2.0, L / 2 + 8, y, 4.2, "khondalite")
        for lx in (-1.5, 1.5):
            M.box(.5, .5, 2.4, L / 2 + 6 + lx, y, 0, "khondalite_d")


def sanchi(M: Model):
    """Great Stupa: 36.6 m dome, 16.5 m high, terrace, harmika + triple chhatra, vedika railing, 4 toranas."""
    M.ground(110, 110, "grass")
    R = 18.3
    M.cyl(R + 2, 4.3, 0, 0, 0, "buff_d", 64)                    # medhi (raised terrace)
    M.cyl(R + 3.2, .9, 0, 0, 4.3, "buff", 64)                    # terrace railing band
    M.dome(R, 0, 0, 3.6, "buff", rise=12.9 / R, sections=64)
    M.box(5, 5, 2.4, 0, 0, 16.4, "buff_d")                      # harmika
    M.cyl(.25, 7, 0, 0, 16.4, "buff_d", 8)
    for i, (r, z) in enumerate(((3.0, 19.0), (2.4, 20.6), (1.8, 22.0))):
        M.cyl(r, .35, 0, 0, z, "buff_d", 24)
    # ground railing (vedika) ring with posts
    Rv = R + 9
    M.torus(Rv, .35, 0, 0, 3.0, "buff_d", axis="z")
    M.torus(Rv, .35, 0, 0, 2.0, "buff_d", axis="z")
    for k in range(72):
        a = 2 * math.pi * k / 72
        M.box(.5, .5, 3.2, Rv * math.cos(a), Rv * math.sin(a), 0, "buff_d", rot=a)
    # four toranas (gateways) at cardinal points, ~8.5 m
    for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        cx, cy = (Rv + 2) * math.cos(a), (Rv + 2) * math.sin(a)
        tx, ty = -math.sin(a), math.cos(a)
        for s in (-1, 1):
            M.box(.9, .9, 6.5, cx + tx * s * 2.2, cy + ty * s * 2.2, 0, "buff_d", rot=a)
        for i, z in enumerate((6.3, 7.4, 8.5)):
            M.box(.9 if abs(tx) < .5 else 7.6, 7.6 if abs(tx) < .5 else .9, .55, cx, cy, z, "buff")


def brihadeeswara(M: Model):
    """Brihadisvara (Thanjavur): ~66 m 13-tier square vimana, octagonal cupola, mandapas and Nandi to the east."""
    M.ground(240, 140, "ground")
    M.box(240 - 20, 120, 6, 0, 0, 0, "granite_l")              # enclosure (cloister walls outline)
    M.box(236 - 20, 116, 6.2, 0, 0, 0, "ground")
    # sanctum base (2 storeys) + 13 tiers
    vx = -40
    M.box(30, 30, 16, vx, 0, 0, "granite")
    for f, (x, y) in {"n": (vx, 15), "s": (vx, -15), "e": (vx + 15, 0)}.items():
        M.arcade(22, 5, x, y, 4, f, 3, depth=.5, mat="granite_d")
    M.box(32, 32, 1.4, vx, 0, 16, "granite_d")
    top, w, d = M.tiered(28, 28, 44, 13, vx, 0, 17.4, ("granite", "granite_l"), shrink=.32)
    M.cyl(w * .55, 1.5, vx, 0, top, "granite_d", 8)
    M.revolve([(w * .9, 0), (w * 1.0, 2), (w * .8, 4.2), (w * .3, 5.4), (0, 5.6)], vx, 0, top + 1.5, "granite_l", 8)  # octagonal sikhara
    M.cone(.6, .1, 3.6, vx, 0, top + 7, "gold", 12)                                    # kalasha
    # mandapas toward the east
    M.box(40, 24, 11, -6, 0, 0, "granite")
    M.box(42, 26, 1.2, -6, 0, 11, "granite_d")
    M.box(20, 18, 8, 26, 0, 0, "granite")
    M.box(22, 20, 1, 26, 0, 8, "granite_d")
    # Nandi mandapa
    M.box(10, 10, .8, 60, 0, 0, "granite_d")
    for sx in (-1, 1):
        for sy in (-1, 1):
            M.box(.7, .7, 6, 60 + sx * 4, sy * 4, .8, "granite_l")
    M.box(11, 11, 1, 60, 0, 6.8, "granite")
    M.box(5, 2.5, 3.4, 60, 0, .8, "granite_d")
    # Keralantakan & Rajarajan gopurams (east entrances)
    for gx, h, tiers in ((95, 26, 5), (78, 20, 3)):
        M.box(10, 22, 8, gx, 0, 0, "granite")
        M.arcade(6, 6, gx + 5, 0, 0, "e", 1, depth=.6)
        top, w, d = M.tiered(10, 22, h - 8, tiers, gx, 0, 8, ("stucco", "granite_l"), shrink=.55)
        M.box(w * .7, d, 2, gx, 0, top, "stucco")


def fatehpur(M: Model):
    """Buland Darwaza (~54 m above ground incl. steps): half-octagonal iwan facade facing south, chhatris on top."""
    M.ground(140, 120, "ground")
    # flight of steps (~13 m) facing south
    for k in range(12):
        M.box(48 - k * .8, 22 - k * 1.5, 1.1, 0, -24 + k * .75, k * 1.1, "redstone_d")
    z0 = 13
    W, D, H = 40, 30, 41
    M.box(W, D, H, 0, 6, z0, "redstone")
    # half-octagonal projecting front
    for ang, dx in ((0, 0), (math.radians(45), 1), (math.radians(-45), -1)):
        x = dx * 15.5
        y = -9 - (0 if dx == 0 else -4.2)
        M.box(14 if dx == 0 else 12, 3, H, x, y, z0, "redstone", rot=ang)
    # great iwan (arched recess) + inner doorway
    M.box(14, 2, 30, 0, -10.6, z0, "redstone_d")
    M.cyl(7, 2, 0, -10.6, z0 + 30, "redstone_d", 24, axis="y")
    M.box(6, 2.2, 10, 0, -10.9, z0, "void")
    # marble inlay bands and arched panels
    for z in (z0 + 33, z0 + 38):
        M.box(42, 31.2, .6, 0, 6, z, "marble")
    M.arcade(30, 4, 0, -9.0, z0 + 2, "s", 3, depth=.3)
    M.box(40, 30, 2.5, 0, 6, z0 + H, "redstone_d")              # parapet / crenellation band
    M.crenels(40, 0, -9, z0 + H + 2.5, "x", .9, "redstone")
    for k in range(13):                                          # row of small chhatris
        M.chhatri(1.2, -18 + k * 3, -6, z0 + H + 2.5, "redstone", "marble")
    for sx in (-1, 1):
        M.chhatri(3.2, sx * 18, 16, z0 + H + 2.5, "redstone", "marble")
        M.cyl(2.6, H + 2.5, sx * 21.5, -6, z0, "redstone", 8)      # corner turrets
        M.chhatri(2.4, sx * 21.5, -6, z0 + H + 2.5, "redstone", "marble")
    # mosque courtyard wall behind (north)
    M.box(110, 4, 12, 0, 40, 0, "redstone")


def meenakshi(M: Model):
    """Meenakshi Amman Temple: walled rectangle with 4 tall outer gopurams (south ~52 m), golden vimanas, lotus tank."""
    M.ground(300, 280, "ground")
    Wx, Wy = 254, 237
    for (x, y, w, d) in ((0, Wy / 2, Wx, 3), (0, -Wy / 2, Wx, 3), (Wx / 2, 0, 3, Wy), (-Wx / 2, 0, 3, Wy)):
        M.box(w, d, 9, x, y, 0, "poly_w")
    poly = ("poly_w", "poly_r", "poly_y", "poly_b", "poly_g")
    gop = {"s": (0, -Wy / 2, 52, 11), "n": (0, Wy / 2, 49, 10), "e": (Wx / 2, 0, 46, 9), "w": (-Wx / 2, 0, 47, 9)}
    for f, (x, y, h, tiers) in gop.items():
        along_x = f in ("n", "s")
        w, d = (34, 16) if along_x else (16, 34)
        M.box(w, d, 9, x, y, 0, "granite")
        if along_x:
            M.arcade(6, 7, x, y + (d / 2 if f == "n" else -d / 2), 0, f, 1, depth=.6)
        else:
            M.arcade(6, 7, x + (w / 2 if f == "e" else -w / 2), y, 0, f, 1, depth=.6)
        top, tw, td = M.tiered(w, d, h - 9, tiers, x, y, 9, poly, shrink=.5)
        M.box(tw * (.6 if along_x else 1), td * (1 if along_x else .6), 2.5, x, y, top, "poly_y")
        for k in range(9):
            o = -(tw if along_x else td) / 2 + (k + .5) * (tw if along_x else td) / 9
            M.cone(.4, .05, 2, x + (o if along_x else 0), y + (0 if along_x else o), top + 2.5, "gold", 8)
    # inner enclosure & smaller gopurams
    M.box(130, 110, 8, 10, 10, 0, "poly_w")
    for (x, y) in ((10, -45), (10, 65), (75, 10)):
        M.tiered(14, 9, 22, 5, x, y, 8, poly, shrink=.55)
    # golden vimanas over the sanctums
    for (x, y) in ((-5, 25), (35, 0)):
        M.box(10, 10, 9, x, y, 8, "poly_w")
        top, w, d = M.tiered(9, 9, 7, 3, x, y, 17, ("gold",), shrink=.55)
        M.dome(w * .6, x, y, top, "gold", rise=.9)
    # Potramarai Kulam (golden lotus tank) with steps and pillared corridor
    M.box(50, 36, .6, -60, -20, 0, "granite_l")
    M.box(40, 26, .7, -60, -20, 0, "water")
    M.cyl(1.2, 7, -60, -20, .5, "gold", 12)
    for k in range(10):
        for sy in (-1, 1):
            M.cyl(.5, 5, -82 + k * 4.8, -20 + sy * 18.5, 0, "poly_w", 8)
    # thousand-pillar hall block
    M.box(60, 45, 6, -60, 55, 0, "granite_l")


BUILDERS = {
    "golconda-fort": (golconda, "Golconda Fort"), "qutb-shahi-tombs": (qutb_shahi, "Qutb Shahi Tombs — tomb of Muhammad Quli Qutb Shah"),
    "red-fort": (red_fort, "Red Fort — Lahori Gate"), "hampi": (hampi, "Hampi — Virupaksha Temple"),
    "ajanta-caves": (ajanta, "Ajanta Caves — horseshoe scarp"), "ellora-caves": (ellora, "Ellora — Kailasa Temple (Cave 16)"),
    "konark-sun-temple": (konark, "Konark Sun Temple"), "sanchi-stupa": (sanchi, "Sanchi — Great Stupa"),
    "brihadeeswara-temple": (brihadeeswara, "Brihadisvara Temple, Thanjavur"),
    "fatehpur-sikri": (fatehpur, "Fatehpur Sikri — Buland Darwaza"), "meenakshi-amman-temple": (meenakshi, "Meenakshi Amman Temple"),
}


# Builders model the main entrance on the south (-Y) face unless noted; YAW rotates it to the real bearing.
YAW = {"red-fort": -90.0}   # Lahori Gate faces west
# Initial camera direction (three.js Y-up, +Z = south) for sites seen best from above
VIEW = {"ellora-caves": [-0.35, 0.9, 0.35], "ajanta-caves": [0.15, 0.75, 0.75]}


def build(slug: str) -> Path:
    fn, title = BUILDERS[slug]
    M = Model()
    fn(M)
    path = OUT / f"{slug}.glb"
    M.export(path, {"title": title, "source": "parametric",
                    "author": "ChronicleHeritage parametric builder (scripts/build_models.py)",
                    "license": "Project-generated", "note": "Massing model from documented dimensions; approximate"},
             yaw_deg=YAW.get(slug, 0.0))
    calib = {"north_deg": 0, "frame": {"w": 1.0, "h": 1.0}, "source": "parametric"}
    if slug in VIEW:
        calib["view"] = VIEW[slug]
    (OUT / f"{slug}.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (sys.argv[1:] or BUILDERS):
        p = build(s)
        print(f"{s:24s} {p.stat().st_size / 1024:8.0f} KB")
