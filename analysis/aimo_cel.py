"""Mark layer of the report's figures: flat editorial monograph on white paper.

Every panel opens with a 0.6 pt top rule, a square letter tag and a bold title with a gray subtitle; there are no
frames, cast shadows, screentone or gleams. Fills are single flat steps of one family with a 0.55 pt outline in the
family's stroke tone; pale washes (alpha <= 0.14 over paper) mark only a contrast zone or a confidence band. Every
color comes from aimo_palette.py; a mark has three levels: pale (fill pale, line body), emphasized (fill mid, line
rim) and accent (fill body, at most a few per figure).
"""
import colorsys

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

from aimo_palette import HUES, INK, MUTED, PAPER, hue, light, line, mid, rim, spec, wash

LINE = HUES['slate'][0]
TICK = HUES['slate'][1]
FAINT = HUES['slate'][2]
RULE = 0.6                                        # pt, the panel top rule
EDGE = 0.55                                       # pt, outline of bars, cells and cards
BOLD = {'fontweight': 'bold'}


def deeper(c, dl=0.10):
    h, l, s = colorsys.rgb_to_hls(*to_rgb(c))
    return to_hex(colorsys.hls_to_rgb(h, max(0.0, l - dl), s))


def ramp(role, level=0):
    """(fill, line) of a mark."""
    if level == 0:
        return light(role), line(role)
    if level == 1:
        return mid(role), rim(role)
    return line(role), rim(role)


def shades(role):
    """Five depth steps of one family, pale to deep."""
    return [wash(role), light(role), spec(role), mid(role), line(role)]


def depth_of(v, lo, hi, n=5):
    t = (v - lo) / (hi - lo) if hi > lo else 0.0
    return int(min(n - 1, max(0, np.floor(t * n))))


def tint(role, alpha=0.14):
    """A pale wash of the family's mid step at alpha <= 0.14 over paper, for contrast zones and bands."""
    return (*to_rgb(mid(role)), min(alpha, 0.14))


def hatched(patch, color, lw=0.3, n=5, kind='/'):
    patch.set_hatch(kind * n)
    patch.set_hatchcolor(color)
    patch.set_hatch_linewidth(lw)
    return patch


# ---------------------------------------------------------------- canvas and panels (inches)
def canvas(fig, z=0):
    fx = fig.add_axes([0, 0, 1, 1], zorder=z)
    w, h = fig.get_size_inches()
    fx.set_xlim(0, w)
    fx.set_ylim(0, h)
    fx.axis('off')
    fx.patch.set_alpha(0)
    return fx


def axes(fig, x0, y0, x1, y1, z=1):
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x0 / fw, y0 / fh, (x1 - x0) / fw, (y1 - y0) / fh], zorder=z)
    ax.patch.set_facecolor('none')
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def sheet(fig, fx, x, w, letter, title, sub=None, size=7.0, sub_size=6.4, head=0.20, y=None, bottom=0.0):
    """An editorial panel head across [x, x + w], at the top of the figure or with its top edge at y for a lower
    deck: a square letter tag, a bold title and a gray subtitle. Registers the panel body (below the head, down to
    bottom) for figcheck and returns its top y."""
    H = fig.get_size_inches()[1] if y is None else y
    s, yc = 0.108, H - 0.085
    fx.plot([x, x + w], [H - 0.01] * 2, color=INK, lw=RULE, zorder=9, solid_capstyle='butt')
    fx.add_patch(Rectangle((x, yc - s / 2), s, s, fc=INK, ec='none', zorder=9, gid='tab'))
    fx.text(x + s / 2, yc - 0.004, letter, fontsize=size - 0.4, color=PAPER, ha='center', va='center', zorder=9.1,
            gid='tab', **BOLD)
    r = fig.canvas.get_renderer()
    t = fx.text(x + s + 0.045, yc, title, fontsize=size, color=INK, ha='left', va='center', zorder=9, **BOLD)
    if sub:
        x1 = x + s + 0.045 + t.get_window_extent(r).width / fig.dpi + 0.05
        fx.text(x1, yc, sub, fontsize=sub_size, color=MUTED, ha='left', va='center', zorder=9)
    top = H - head
    fig._cel_sheets = getattr(fig, '_cel_sheets', []) + [(x - 0.03, bottom, w + 0.06, top - bottom)]
    return top


def card(ax, x, y, w, h, role='slate', fc=PAPER, lw=EDGE, z=3, ec=None, ls='-'):
    """A flat box in data units of ax: one fill, one outline."""
    p = Rectangle((x, y), w, h, fc=fc, ec=ec or line(role), lw=lw, ls=ls, zorder=z, joinstyle='miter')
    ax.add_patch(p)
    return p


# ---------------------------------------------------------------- unit conversions
def _axes_inches(ax):
    fw, fh = ax.figure.get_size_inches()
    b = ax.get_position()
    return b.width * fw, b.height * fh


def pt_dx(ax, pt):
    x0, x1 = ax.get_xlim()
    return pt / 72 * abs(x1 - x0) / _axes_inches(ax)[0]


def pt_dy(ax, pt):
    y0, y1 = ax.get_ylim()
    return pt / 72 * abs(y1 - y0) / _axes_inches(ax)[1]


# ---------------------------------------------------------------- marks
def _closed(pts):
    pts = np.vstack([pts, pts[:1]])
    codes = np.full(len(pts), Path.LINETO, dtype=Path.code_type)
    codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
    return Path(pts, codes)


_T = np.radians(np.linspace(0, 360, 49)[:-1])
DISC = _closed(np.column_stack([np.cos(_T), np.sin(_T)]))
SQUARE = _closed(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.82)
DIAMOND = _closed(np.array([[0, -1], [1, 0], [0, 1], [-1, 0]]) * 1.05)


def _marks(ax, xs, ys, shape, d, fc, ec, lw, z, transform=None, gid=None):
    xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
    col = PathCollection([shape], sizes=[(d / 2) ** 2], offsets=np.column_stack([xs, ys]),
                         offset_transform=transform or ax.transData, facecolors=[fc], edgecolors=[ec],
                         linewidths=[lw], zorder=z, gid=gid)
    col.set_transform(IdentityTransform())
    ax.add_collection(col, autolim=False)
    return col


def discs(ax, xs, ys, role, level=1, d=5.0, lw=0.6, z=5, transform=None, colors=None, hollow=False, gid=None):
    """Flat round marks with a 0.6 pt outline."""
    f, l = colors or ramp(role, level)
    return _marks(ax, xs, ys, DISC, d, PAPER if hollow else f, l, lw, z, transform, gid)


def squares(ax, xs, ys, role, level=1, d=4.6, lw=0.6, z=5, transform=None, gid=None):
    f, l = ramp(role, level)
    return _marks(ax, xs, ys, SQUARE, d, f, l, lw, z, transform, gid)


def diamonds(ax, xs, ys, role, level=1, d=5.0, lw=0.6, z=5, transform=None, gid=None):
    f, l = ramp(role, level)
    return _marks(ax, xs, ys, DIAMOND, d, f, l, lw, z, transform, gid)


def ring(ax, x, y, d=8.5, role='slate', z=6.6, lw=0.6, transform=None):
    return _marks(ax, [x], [y], DISC, d, 'none', line(role), lw, z, transform)


def chip(ax, x, y, text, role, size=6.4, ha='left', va='center', transform=None, bold=False, z=7, pad=0.26,
         face=PAPER, color=None):
    """An academic label tag: square frame lw 0.45 in the family's stroke tone, paper face."""
    return ax.text(x, y, text, fontsize=size, color=color or rim(role), ha=ha, va=va, zorder=z,
                   transform=transform or ax.transData, **(BOLD if bold else {}),
                   bbox=dict(boxstyle=f'square,pad={pad}', fc=face, ec=line(role), lw=0.45))


def hbar(ax, y, lo, hi, role, level=0, h=3.4, lw=EDGE, z=3, colors=None):
    f, l = colors or ramp(role, level)
    hy = pt_dy(ax, h) / 2
    ax.add_patch(Rectangle((lo, y - hy), hi - lo, 2 * hy, fc=f, ec=l, lw=lw, zorder=z, joinstyle='miter'))


def vbar(ax, x, lo, hi, role, level=0, w=4.0, lw=EDGE, z=3, colors=None):
    f, l = colors or ramp(role, level)
    hx = pt_dx(ax, w) / 2
    ax.add_patch(Rectangle((x - hx, lo), 2 * hx, hi - lo, fc=f, ec=l, lw=lw, zorder=z, joinstyle='miter'))


def cline(ax, x, y, role, level=0, lw=1.0, under=1.6, z=3, ls='-', color=None, gid=None):
    """A line over a white underlay so it reads cleanly across grids and other traces."""
    ax.plot(x, y, color=PAPER, lw=lw + under, solid_capstyle='round', solid_joinstyle='round', zorder=z, gid=gid)
    ax.plot(x, y, color=color or (line(role) if level == 0 else rim(role)), lw=lw, ls=ls, solid_capstyle='round',
            solid_joinstyle='round', zorder=z + 0.02, gid=gid)


def leader(ax, x0, y0, x1, y1, role='slate', z=6.4):
    ax.plot([x0, x1], [y0, y1], color=deeper(line(role), 0.08), lw=0.45, zorder=z, solid_capstyle='butt')


def rule(fx, x0, x1, y, color=None, lw=0.45, z=1.5, ls='-'):
    """A light horizontal reference rule (gridline) that labels may sit beside."""
    fx.plot([x0, x1], [y, y], color=color or FAINT, lw=lw, ls=ls, zorder=z, solid_capstyle='butt', gid='texture')


def tag_line(fx, x, y, color=INK, w=0.10, lw=0.9, ls='-', z=4):
    fx.add_line(Line2D([x, x + w], [y, y], color=color, lw=lw, ls=ls, zorder=z))
