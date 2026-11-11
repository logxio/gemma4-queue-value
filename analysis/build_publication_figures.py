"""Three print-size plates: persistence and queue crossing, Gemma sandbox and trajectory X-ray, stopping value.

Run with /opt/homebrew/bin/python3.13. Every canvas is 6.85 in wide, the \\textwidth of the A4 build, so one canvas
point prints as one point. Every number is read from evidence/results, evidence/gemma_a100, evidence/data or the
hashed ATIF traces at run time; the provenance sidecar records each source hash and the plotted values.
"""
from pathlib import Path
import sys, json, csv, glob, hashlib, re, shutil
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import LineCollection
from scipy.interpolate import PchipInterpolator
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
PREVIEW = Path('/tmp/gemma4_fig_preview_s5')
BASE = Path(__file__).resolve().parent
ARC = BASE
sys.path.insert(0, str(BASE))
import aimo_palette as pal
import aimo_cel as sheet
import cel
import figcheck
cel.HUES.update(pal.HUES)          # crescent discs and screentone in this paper's mineral palette
figcheck.MIN = 6.5                 # print floor: 1 canvas pt = 1 printed pt at \textwidth

W = 6.85
FS, FSS = 7.2, 6.6
INK, MUTED, PAPER = pal.INK, pal.MUTED, pal.PAPER
SOURCES, CHECKS, DATA = {}, {}, {}
NOTES = [
 'Canvas width 6.85 in equals the LaTeX \\textwidth (A4, 18 mm margins); figures are included at width=\\textwidth, so text prints at its set size (7.2 pt body, 6.6 pt ticks).',
 'Llama persistence is recomputed with equal issue weight over issues with at least three attempts and 1000 issue-bootstrap draws. The pooled 1.05 is nebius_restart.json all-pair q/p for Llama 3 8B; 91 is the largest attempt count per issue in the Llama collection CSV.',
 'The 465-issue transfer matrix holds four agent systems; cells are P(column succeeds | row fails). The diagonal has no same-system repeat and is hatched.',
 'Queue stations are training fresh yields per 1k calls by language (queue_restart_decision.json), ordered high to low. Retry levels are restart_value.json compute terciles with edges 49 and 80 calls.',
 'Budget ribbons are the 250-replicate grid recorded at 5 to 75 percent and carried at zero to 100 percent. The crossing peak whisker is the grid 95 percent band at 35 percent; the retry-ordering diamond is the separate 1000-replicate 50 percent run (+49.7 [27.2, 72.0]) reported in Table 2.',
 'Gemma per-task counts 11, 14, 7, 44 (3, 2, 1, 0 of 3 rounds) come from full76_official.jsonl. Calibrated 127/129 and private gold 100 percent are source-reported in the dated forensics note and are not outputs of this experiment.',
 '7,157.4 s is the sum of event intervals preceding schema rejections, not identified causal time. The +10.71 replay compares a 600 s cap plus a 3-strike schema breaker against 60-minute timeouts with zero late recovery.',
 'Figure 3 uses Y/(lambda C) so that systems with different yields share one value field; positive value lies below the diagonal. The +13.8 selector difference is decision_value.json repairs_forfeited_selecting_by_cv_auc_instead_of_cv_value_sum4.',
]

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': FS, 'axes.unicode_minus': True,
                     'pdf.fonttype': 42, 'hatch.linewidth': .25, 'xtick.labelsize': FSS, 'ytick.labelsize': FSS,
                     'axes.labelsize': FSS, 'text.color': INK, 'axes.labelcolor': INK})


# ---------------------------------------------------------------- sources
def raw(p):
    p = Path(p); p = p if p.is_absolute() else ROOT / p
    b = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = hashlib.sha256(b).hexdigest()
    return b

def js(n): return json.loads(raw('evidence/results/' + n + '.json'))
def rows(p): return list(csv.DictReader(raw(p).decode().splitlines()))


# ---------------------------------------------------------------- drawing helpers
def T(ax, x, y, s, size=FS, color=INK, bold=False, **kw):
    kw.setdefault('va', 'center'); kw.setdefault('zorder', 8)
    return ax.text(x, y, s, fontsize=size, color=color, fontweight='bold' if bold else 'normal', **kw)

def card(ax, x0, y0, x1, y1, role='slate', fc=PAPER, z=6, lw=.5):
    """A clean flat framed card in inch units of the canvas, without cast shadows."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=pal.line(role), lw=lw, zorder=z, joinstyle='miter'))

def bead(ax, x, y, role, d=4.6, level=1):
    sheet.discs(ax, [x], [y], role, d=d + 1.4, colors=(PAPER, PAPER), lw=0, z=4.9)
    return sheet.discs(ax, [x], [y], role, level=level, d=d, lw=.55, z=5.1)

def diamond_mark(ax, x, y, role, d=5.2, level=1):
    f_col, l_col = sheet.ramp(role, level)
    sheet._marks(ax, [x], [y], sheet.DIAMOND, d + 1.4, PAPER, PAPER, 0, 4.9)
    return sheet.diamonds(ax, [x], [y], role, level=level, d=d, lw=.55, z=5.1)

def width(fig, t):
    return t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi

def legend_row(fig, fx, x, y, items, gap=.12, size=FSS):
    """items: (kind, role, label, extra) placed left to right, measured so nothing touches."""
    for kind, role, label, extra in items:
        if kind == 'bar':
            p = Rectangle((x, y - .03), .1, .06, fc=extra or pal.mid(role), ec=pal.line(role), lw=.4, zorder=6)
            fx.add_patch(p)
        elif kind == 'keel':
            p = Polygon([(x, y + .03), (x + .1, y + .03), (x + .07, y - .03), (x + .03, y - .03)], fc=pal.light(role),
                        ec=pal.line(role), lw=.4, zorder=6)
            sheet.hatched(p, pal.line(role), lw=.25, n=3); fx.add_patch(p)
        elif kind == 'disc':
            bead(fx, x + .05, y, role, 4.4)
        elif kind == 'square':
            sheet.squares(fx, [x + .05], [y], role, 1, d=4.4)
        elif kind == 'diamond':
            diamond_mark(fx, x + .05, y, role, 5.0)
        elif kind == 'tick':
            fx.plot([x + .05] * 2, [y - .035, y + .035], color=pal.line(role), lw=.8, zorder=6)
        elif kind == 'line':
            fx.plot([x, x + .14], [y, y], color=pal.line(role), lw=.9, ls=extra or '-', zorder=6)
        t = T(fx, x + (.18 if kind == 'line' else .14), y, label, size, pal.rim(role) if role != 'ink' else INK)
        x += (.18 if kind == 'line' else .14) + width(fig, t) + gap
    return x

def pchip(x, y, n=240):
    xx = np.linspace(x[0], x[-1], n)
    return xx, PchipInterpolator(x, y)(xx)

def axes(fig, x0, y0, x1, y1, xlim, ylim):
    a = sheet.axes(fig, x0, y0, x1, y1)
    a.set(xlim=xlim, ylim=ylim)
    cel.axes_style(a)
    a.tick_params(labelsize=FSS, length=2.0, pad=1.6)
    return a

def figure(h):
    f = plt.figure(figsize=(W, h), dpi=300, facecolor='white')
    return f, sheet.canvas(f)

def head(f, fx, x, w, letter, title, sub=None, y=None, bottom=0.0):
    """Clean panel header without top horizontal rule."""
    H = f.get_size_inches()[1] if y is None else y
    s, yc = 0.108, H - 0.080
    fx.add_patch(Rectangle((x, yc - s / 2), s, s, fc=INK, ec='none', zorder=9, gid='tab'))
    fx.text(x + s / 2, yc - 0.004, letter, fontsize=FS, color=PAPER, ha='center', va='center', zorder=9.1,
            gid='tab', fontweight='bold')
    r = f.canvas.get_renderer()
    t = fx.text(x + s + 0.045, yc, title, fontsize=FS + .4, color=INK, ha='left', va='center', zorder=9, fontweight='bold')
    if sub:
        x1 = x + s + 0.045 + t.get_window_extent(r).width / f.dpi + 0.05
        fx.text(x1, yc, sub, fontsize=FSS, color=MUTED, ha='left', va='center', zorder=9)
    top = H - 0.20
    f._cel_sheets = getattr(f, '_cel_sheets', []) + [(x - 0.03, bottom, w + 0.06, top - bottom)]
    return top

def save(f, name):
    hits = figcheck.report(f, name)
    CHECKS[name] = {'hits': hits, 'hit_count': len(hits)}
    f.savefig(OUT / (name + '.pdf'), facecolor='white')
    f.savefig(OUT / (name + '.png'), dpi=300, facecolor='white')
    a = np.asarray(Image.open(OUT / (name + '.png')).convert('RGB')).astype(int)
    green = int(((a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])) >= 15).sum())
    ink = (a.sum(axis=2) < 740)
    cols, rws = ink.any(axis=0), ink.any(axis=1)
    CHECKS[name].update(green_px=green, pdf_bytes=(OUT / (name + '.pdf')).stat().st_size,
                        size_in=list(f.get_size_inches().round(3)), png_px=list(a.shape[1::-1]),
                        ink_bbox_fraction=[round(cols.mean(), 3), round(rws.mean(), 3)])
    plt.close(f)


# ================================================================ Figure 1
def persistence():
    r = js('restart_value'); oh = js('openhands_queue_restart'); au = js('gemma4_sandbox_and_schema_audit')
    out = []
    def entry(name, n, v, ci): out.append(dict(name=name, n=n, **{k: v[k] for k in ('p', 'q', 'q_over_p')}, ci=ci))
    entry('Qwen3.8 27B', r['primary']['attempts'], r['primary'], r['primary']['bootstrap_95'])
    v = oh['statistics']['equal_issue']
    entry('Qwen3 Coder 480B', oh['attempts'], {k: v[k]['point'] for k in v}, {k: v[k]['ci95'] for k in v})
    entry('Gemma 4 31B', au['observed']['attempts'], au['observed']['rates'], au['observed']['rate_bootstrap_95ci'])
    groups = defaultdict(lambda: defaultdict(list))
    for t in rows('evidence/results/nebius_attempt_costs.csv'):
        groups[t['model_name']][t['instance_id']].append(t['target'] == 'True')
    rng = np.random.default_rng(20261109)
    for model in ('8b', '70b', '405b'):
        gg = groups['swe-agent-llama-' + model]
        g = np.array([(len(v), sum(v)) for v in gg.values() if len(v) >= 3], float)
        n, s = g.T; pv = s / n; joint = (n - s) * s / (n * (n - 1)); fv = 1 - pv
        def st(idx):
            p = pv[idx].mean(); q = joint[idx].mean() / fv[idx].mean(); return p, q, q / p
        v = st(np.arange(len(n)))
        b = np.array([st(rng.integers(0, len(n), len(n))) for _ in range(1000)])
        ci = dict(zip(('p', 'q', 'q_over_p'), np.percentile(b, [2.5, 97.5], axis=0).T.tolist()))
        entry('Llama 3 ' + model.upper(), sum(map(len, gg.values())), dict(zip(('p', 'q', 'q_over_p'), v)), ci)
    tries = max(len(v) for m in groups.values() for v in m.values())
    pooled = js('nebius_restart')['models']['swe-agent-llama-8b']['q_over_p']
    DATA['persistence'] = out; DATA['llama_max_tries'] = tries; DATA['llama8b_pooled_q_over_p'] = pooled
    return out, tries, pooled


def fig1():
    pp, tries, pooled = persistence()
    q = js('queue_restart_decision'); r = js('restart_value'); cross = js('cross_system_restart')
    oh = js('openhands_queue_restart'); neb = js('nebius_queue_restart'); conf = js('confirmation_delivery')
    H = 2.92
    f, fx = figure(H)
    xa, wa, xb = 0.0, 3.28, 3.42
    top = head(f, fx, xa, wa, 'a', 'Persistence and failure transfer', f'{sum(v["n"] for v in pp):,} trajectories')
    head(f, fx, xb, W - xb, 'b', 'Queue crossing', 'Qwen3.8 27B held-out replay')

    # ---- a, upper: p and q per configuration, q/p chips
    b0, b1, vmax = 1.18, 2.50, .65
    bx = lambda v: b0 + v / vmax * (b1 - b0)
    yh = top - .07
    T(fx, 0, yh, 'Configuration', FSS, MUTED)
    T(fx, 1.10, yh, 'n', FSS, MUTED, ha='right')
    legend_row(f, fx, b0 - .02, yh, [('bar', 'sapphire', 'first-try p', None), ('bar', 'cobalt', 'retry after failure q', None)], gap=.08)
    T(fx, 2.80, yh, 'q/p', FSS, MUTED, ha='center')
    pitch = .158
    ys = [yh - .16 - i * pitch for i in range(len(pp))]
    for v, y in zip(pp, ys):
        T(fx, 0, y, v['name'], FS)
        T(fx, 1.10, y, f'{v["n"]:,}', FSS, MUTED, ha='right')
        for off, k, role in ((.032, 'p', 'sapphire'), (-.032, 'q', 'cobalt')):
            fx.add_patch(Rectangle((b0, y + off - .019), bx(v[k]) - b0, .038, fc=pal.mid(role), ec=pal.line(role), lw=.4, zorder=3))
            if k == 'q':
                lo, hi = (bx(c) for c in v['ci']['q'])
                fx.plot([lo, hi], [y + off] * 2, color=pal.rim(role), lw=.6, zorder=4)
                for e in (lo, hi): fx.plot([e, e], [y + off - .022, y + off + .022], color=pal.rim(role), lw=.5, zorder=4)
        sheet.chip(fx, 2.80, y, f'{v["q_over_p"]:.3f}', 'amber', size=FSS, ha='center', face=pal.wash('amber'), pad=.22)
    yr = ys[-1] - .105
    fx.plot([b0, b1], [yr, yr], color=cel.LINE, lw=.55, zorder=2)
    for t in (0, .2, .4, .6):
        fx.plot([bx(t)] * 2, [yr, yr - .025], color=cel.LINE, lw=.45)
        T(fx, bx(t), yr - .075, f'{t:g}', FSS, INK, ha='center')
    for t in (0, .2, .4, .6):
        fx.plot([bx(t)] * 2, [yr, ys[0] + .07], color=pal.light('slate'), lw=.3, zorder=1, gid='texture')

    # ---- a, lower: 4 x 4 failure transfer
    systems = ['sweagent', 'agentless', 'openhands', 'mini']; short = ['SWE-agent', 'Agentless', 'OpenHands', 'Mini']
    mt = yr - .2
    i8 = next(i for i, v in enumerate(pp) if v['name'].endswith('8B'))
    eq8 = pp[i8]['q_over_p']
    lines = [('Llama 8B pooling', FSS, pal.rim('coral'), True),
             (f'pooled pairs  {pooled:.2f}', FSS, INK, False),
             (f'equal weight  {eq8:.3f}', FSS, pal.rim('amber'), True)]
    probe = [T(fx, 0, 0, s, sz, bold=bd) for s, sz, _, bd in lines]
    c1 = 3.25; c0 = c1 - max(width(f, t) for t in probe) - .14
    for t in probe: t.remove()
    cx0, ch = .54, .20; cw = (c0 - .08 - cx0) / 4
    T(fx, cx0 + 2 * cw, mt + .005, 'Transfer P(col | row fails)', FSS, MUTED, ha='center')
    cy = [mt - .25 - i * ch for i in range(4)]
    for j, s in enumerate(short): T(fx, cx0 + (j + .5) * cw, mt - .085, s, FSS, INK, ha='center')
    shades = sheet.shades('sapphire'); steps = [0, .15, .3, .45, .6, .75]
    mat = {}
    for i, s in enumerate(systems):
        T(fx, cx0 - .04, cy[i], short[i], FSS, INK, ha='right')
        for j, t in enumerate(systems):
            x = cx0 + j * cw
            if s == t:
                p = Rectangle((x, cy[i] - ch / 2), cw, ch, fc=pal.wash('slate'), ec=PAPER, lw=.8, zorder=2)
                sheet.hatched(p, pal.light('slate'), lw=.3, n=3); fx.add_patch(p); continue
            v = cross['pairs'][s + '->' + t]['p_b_given_a_failed']; mat[s + '->' + t] = v
            k = min(4, int(np.searchsorted(steps, v, side='right') - 1))
            fx.add_patch(Rectangle((x, cy[i] - ch / 2), cw, ch, fc=shades[k], ec=PAPER, lw=.8, zorder=2))
            T(fx, x + cw / 2, cy[i], f'{v:.2f}', FSS, PAPER if k >= 3 else INK, ha='center')
    kb = cy[-1] - ch / 2 - .13; kw_ = 4 * cw / 5
    for k in range(5):
        fx.add_patch(Rectangle((cx0 + k * kw_, kb), kw_, .055, fc=shades[k], ec=pal.line('sapphire'), lw=.3, zorder=2))
    for k in range(6):
        T(fx, cx0 + k * kw_, kb - .07, f'{steps[k]:.2f}'.lstrip('0') if k else '0', FSS, MUTED, ha='center')
    T(fx, cx0 - .04, kb + .0275, 'P', FSS, MUTED, ha='right')

    # Llama 8B pooling trap card, centered and compact
    ct, cb = cy[0] + ch / 2, cy[-1] - ch / 2
    card(fx, c0, cb, c1, ct, 'coral')
    cm = (c0 + c1) / 2
    step = (ct - cb - .08) / len(lines); ly = ct - .04 - step / 2
    for s, sz, col, bd in lines:
        T(fx, cm, ly, s, sz, col, bd, ha='center'); ly -= step
    fx.plot([2.97, 3.15, 3.15], [ys[i8], ys[i8], ct], color=pal.line('coral'), lw=.5, ls=(0, (1.5, 1)), zorder=5)

    # ---- b, upper: fresh-yield decay against first-failure retry levels
    cats = sorted([(k, v) for k, v in q['train_fresh_per_1000_calls'].items() if k != '__global__'], key=lambda t: -t[1])
    xx = np.arange(len(cats)); yy = np.array([v for _, v in cats])
    au = axes(f, xb + .34, 1.56, W - .03, top - .04, (-.4, len(cats) - .6), (0, 18.2))
    rr = {v['group']: v['restart_per_1k_calls'] for v in r['failure_groups']['rows'] if v['family'] == 'compute_tercile'}
    e0, e1 = r['failure_groups']['compute_tercile_edges_calls']
    gx, gy = pchip(xx, yy)
    low = rr['low']; sel = gy <= low
    wedge = Polygon(np.c_[np.r_[gx[sel], gx[sel][::-1]], np.r_[gy[sel], np.full(sel.sum(), low)]], fc=sheet.tint('amber', .14), ec='none', zorder=1)
    cel.toned(wedge, pal.spec('amber'), 6, .55); au.add_patch(wedge)
    tiers = [('low', 'cobalt', f'≤{e0:g} calls  {rr["low"]:.2f}', 'above'), ('mid', 'amethyst', f'{e0 + 1:g} to {e1:g} calls  {rr["mid"]:.2f}', 'above'),
             ('high', 'slate', f'>{e1:g} calls  {rr["high"]:.2f}', 'below')]
    for g, role, lab, where in tiers:
        au.plot([-.4, len(cats) - .6], [rr[g]] * 2, color=pal.line(role), lw=.75, ls=(0, (3, 1.6)), zorder=2)
        T(au, -.2, rr[g] + (.95 if where == 'above' else -.95), lab, FSS, pal.rim(role))
    sheet.cline(au, gx, gy, 'sapphire', lw=1.1, z=3)
    for x, y in zip(xx, yy): bead(au, x, y, 'sapphire', 4.2)
    au.set_xticks(xx); au.set_xticklabels([k for k, _ in cats], fontsize=FSS - .1)
    au.tick_params(axis='x', length=1.6, pad=1.2)
    au.set_yticks([0, 4, 8, 12, 16]); au.set_ylabel('Repairs per 1k calls', labelpad=2)
    names_ = [k for k, _ in cats]
    i = names_.index('scala'); T(au, xx[i] + .10, 6.85, f'Scala {yy[i]:.2f}', FSS, pal.rim('sapphire'), ha='right')
    au.plot([xx[i], xx[i]], [yy[i] + .55, 6.00], color=pal.line('sapphire'), lw=.4, zorder=4)
    i = names_.index('kotlin'); T(au, xx[i] - .25, 1.15, f'Kotlin {yy[i]:.2f}', FSS, pal.rim('sapphire'), ha='right')
    dec = q['decomposition']['0.3']; bud = q['budgets']['0.3']['adaptive']['minus_deferred_rate']
    a_ = dec['adaptive_only_restart']; d_ = dec['deferred_rate_only_fresh']
    t = sheet.chip(au, len(cats) - .75, 17.8,
 f'30% budget: +{bud["mean"]:.1f} repairs\n{a_["attempts"]:.0f} retries ({a_["solved"]:.0f}) vs {d_["attempts"]:.0f} fresh ({d_["solved"]:.0f})',
               'amber', size=FSS, ha='right', va='top', face=PAPER, pad=.32, color=INK); t.set_multialignment('center')
    DATA['queue'] = {'fresh_stations': cats, 'retry_rates': rr, 'decomposition_30': dec, 'crossing_30': bud}

    # ---- b, lower: budget decomposition ribbons
    cv = q['curve']; fr = np.array(cv['fractions']) * 100
    cross_m = np.array(cv['adaptive_minus_deferred_rate']['mean']); order_m = np.array(cv['deferred_rate_minus_ranked_breadth']['mean'])
    ad = axes(f, xb + .34, .30, W - .09, 1.20, (5, 100), (-2, 86))
    for m, role in ((order_m, 'amethyst'), (cross_m, 'cobalt')):
        x_, y_ = pchip(fr, m)
        y_ = np.maximum(y_, 0)
        ad.fill_between(x_, 0, y_, color=sheet.tint(role, .14), lw=0, zorder=1)
        sheet.cline(ad, x_, y_, role, lw=.9, z=3)
    ad.plot([fr[-1], 100], [0, 0], color=pal.line('slate'), lw=.6, ls=(0, (1, 1.2)), zorder=2)
    T(ad, 88, 6, 'saturated', FSS, MUTED, ha='center')
    k35 = int(np.argmax(cross_m)); lo35, hi35 = cv['adaptive_minus_deferred_rate']['lo'][k35], cv['adaptive_minus_deferred_rate']['hi'][k35]
    ad.plot([fr[k35]] * 2, [lo35, hi35], color=pal.rim('cobalt'), lw=.6, zorder=4)
    for e in (lo35, hi35): ad.plot([fr[k35] - .9, fr[k35] + .9], [e, e], color=pal.rim('cobalt'), lw=.5, zorder=4)
    bead(ad, fr[k35], cross_m[k35], 'cobalt', 4.6, 2)
    T(ad, fr[k35] - 2.2, cross_m[k35] + 2, f'{fr[k35]:.0f}%  +{cross_m[k35]:.1f}', FSS, pal.rim('cobalt'), ha='right')
    o50 = q['budgets']['0.5']['deferred_rate']['minus_ranked_breadth']
    ad.plot([50, 50], o50['paired_95'], color=pal.rim('amethyst'), lw=.6, zorder=4)
    for e in o50['paired_95']: ad.plot([49.1, 50.9], [e, e], color=pal.rim('amethyst'), lw=.5, zorder=4)
    diamond_mark(ad, 50, o50['mean'], 'amethyst', 5.6, 2)
    T(ad, 50, o50['paired_95'][1] + 6, f'50%  +{o50["mean"]:.1f}', FSS, pal.rim('amethyst'), ha='center')
    T(ad, 6.5, 74, 'Queue crossing', FSS, pal.rim('cobalt'), bold=True)
    T(ad, 6.5, 60, 'Retry ordering', FSS, pal.rim('amethyst'), bold=True)
    ad.set_xticks([5, 20, 35, 50, 75, 100]); ad.set_yticks([0, 40, 80])
    ad.set_xlabel('Total call budget (%)', labelpad=1)
    ad.set_ylabel('Extra repairs', labelpad=2)
    k35n = int(np.argmax(neb['curve']['adaptive_minus_deferred_rate']['mean']))
    n35 = neb['curve']['adaptive_minus_deferred_rate']['mean'][k35n]
    ohq = oh['statistics']['equal_issue']['q_over_p']['point']
    rb = conf['local_evidence']['assistant_calls_budget_0.3']
    t = sheet.chip(ad, 99, 82,
               f'Llama 70B (35%): +{n35 + 1e-9:.1f}\n480B ranked: +{rb["adaptive_minus_plain_breadth"]["mean"]:.1f}',
               'slate', size=FSS, ha='right', va='top', face=PAPER, pad=.3, color=INK); t.set_multialignment('center')
    DATA['budget_curve'] = {'fractions': cv['fractions'], 'crossing': cv['adaptive_minus_deferred_rate'], 'ordering': cv['deferred_rate_minus_ranked_breadth'],
                            'ordering_50_run': o50, 'llama70b_peak': n35, 'openhands_q_over_p': ohq, 'openhands_30': rb}
    DATA['transfer_matrix'] = mat
    save(f, 'fig1_persistence_and_queue_crossing')


# ================================================================ Figure 2
def trace(task, rnd):
    au = AUDIT
    want = next(t['sha256'] for t in au['trace_manifest'] if t['task'] == task and t['round'] == rnd)
    p = ROOT / f'evidence/gemma_a100/traces/r{rnd}_trace_{task}.json'
    b = raw(p)
    assert hashlib.sha256(b).hexdigest() == want, f'{task} R{rnd} trace hash differs from the audit manifest'
    return json.loads(b)

def obs(s): return str(s.get('observation', ''))
SCHEMA = 'mandatory input parameters'

def recovery_chain(tr):
    """The recorded repair loop of the healthy run: first failed edit, the read that follows, the exact edit, the
    passing test run, each with its elapsed time."""
    st = [s for s in tr['steps'] if s.get('tool_calls')]
    name = lambda s: s['tool_calls'][0]['function_name']
    i0 = next(i for i, s in enumerate(st) if name(s) == 'edit_file' and '"error"' in obs(s))
    i1 = next(i for i in range(i0 + 1, len(st)) if name(st[i]) == 'read_file')
    i2 = next(i for i in range(i1 + 1, len(st)) if name(st[i]) == 'edit_file' and '"exact"' in obs(st[i]))
    i3 = next(i for i in range(i2 + 1, len(st)) if '[100%]' in obs(st[i]))
    passed = re.search(r'stdout": "([.]+)', obs(st[i3])).group(1).count('.')
    t = lambda s: s['extra']['elapsed_s']
    return [(t(st[i0]), 'edit mismatch'), (t(st[i1]), 'read context'), (t(st[i2]), 'exact edit'), (t(st[i3]), f'{passed} tests pass')]


def fig2():
    au = AUDIT
    ref = json.loads(raw('evidence/gemma_a100/reference_patch_check_129.json'))
    attempts = [json.loads(l) for l in raw('evidence/gemma_a100/full76_official.jsonl').decode().splitlines()]
    by = defaultdict(list)
    for t in attempts: by[t['task']].append(t)
    hist = Counter(sum(t['resolved'] for t in v) for v in by.values())
    ceiling = ('127', str(ref['total']))
    gold = '100'
    att = {(r['task'], int(r['round'])): r for r in rows('evidence/results/gemma4_sandbox_and_schema_audit_attempts.csv')}
    H = 3.50
    f, fx = figure(H)
    ya_bot = 2.08
    top = head(f, fx, 0, W, 'a', 'Sandbox evaluability waterline',
               f'{ref["total"]} public tasks, {au["observed"]["attempts"]} Gemma 4 31B rollouts', bottom=ya_bot)
    blocked = [t for t in ref['tasks'] if not t['reference_patch_resolved']]
    n_wheel = sum(1 for t in blocked if t['missing_modules']); n_env = len(blocked) - n_wheel
    ly = top - .08
    legend_row(f, fx, .42, ly, [('bar', 'sapphire', f'3 of 3 rounds ({hist[3]})', sheet.shades('sapphire')[4]),
                                ('bar', 'sapphire', f'2 of 3 ({hist[2]})', sheet.shades('sapphire')[3]),
                                ('bar', 'sapphire', f'1 of 3 ({hist[1]})', sheet.shades('sapphire')[2]),
                                ('bar', 'sapphire', f'0 of 3 ({hist[0]})', sheet.shades('sapphire')[1]),
                                ('keel', 'coral', f'missing test wheels ({n_wheel})', None),
                                ('keel', 'burgundy', f'environment mismatch ({n_env})', None)], gap=.13)
    repos = ['fastapi', 'rich', 'requests', 'httpx']
    x0, x1, gap = .42, 5.40, .07
    pitch = (x1 - x0 - gap * (len(repos) - 1)) / len(ref['tasks'])
    water = 2.50; hgt = lambda c: .07 + .1 * c
    dots = [2.99 + k * .043 for k in range(3)]
    cursor = x0; spans = {}
    for repo in repos:
        rr = [t for t in ref['tasks'] if t['task'].split('_')[0] == repo]
        rr.sort(key=lambda t: (not t['reference_patch_resolved'], -sum(z['resolved'] for z in by.get(t['task'], [])),
                               bool(not t['missing_modules']), t['task']))
        start = cursor
        for t in rr:
            x = cursor + pitch / 2; hw = pitch * .40; cursor += pitch
            if t['reference_patch_resolved']:
                c = sum(z['resolved'] for z in by[t['task']]); h = hgt(c)
                fx.add_patch(Polygon([(x - hw, water), (x - hw, water + h - .03), (x, water + h), (x + hw, water + h - .03), (x + hw, water)],
                                     fc=sheet.shades('sapphire')[c + 1], ec=pal.line('sapphire'), lw=.3, zorder=3))
                for z in by[t['task']]:
                    yv = dots[z['round'] - 1]
                    if z['resolved']: sheet.squares(fx, [x], [yv], 'cobalt', 2, d=2.1, lw=.2)
                    else: sheet.squares(fx, [x], [yv], 'slate', 0, d=2.1, lw=.2).set_facecolor(PAPER)
            else:
                role = 'coral' if t['missing_modules'] else 'burgundy'; depth = .17 if role == 'coral' else .27
                p = Polygon([(x - hw, water), (x + hw, water), (x + hw * .25, water - depth), (x - hw * .25, water - depth)],
                            fc=pal.light(role), ec=pal.line(role), lw=.3, zorder=3)
                sheet.hatched(p, pal.line(role), lw=.2, n=3); fx.add_patch(p)
        spans[repo] = (start, cursor); cursor += gap
    fx.plot([x0 - .02, x1 + .02], [water] * 2, color=pal.rim('sapphire'), lw=.9, zorder=4)
    for c in range(4):
        fx.plot([x0 - .06, x0 - .025], [water + hgt(c)] * 2, color=cel.LINE, lw=.45)
        T(fx, x0 - .08, water + hgt(c), f'{c}/3', FSS, INK, ha='right')
    fx.plot([x0 - .045] * 2, [water, water + hgt(3)], color=cel.LINE, lw=.55)
    T(fx, x0 - .08, dots[1], 'R1-3', FSS, MUTED, ha='right')
    T(fx, x0 - .08, water - .13, 'blocked', FSS, pal.rim('coral'), ha='right')
    yl = water - .335
    for repo in ('fastapi', 'rich'):
        a, b = spans[repo]; n = round((b - a) / pitch)
        T(fx, (a + b) / 2, yl, f'{repo}  {n}', FSS, INK, ha='center')
    a, _ = spans['requests']; _, b = spans['httpx']
    T(fx, (a + b) / 2, yl, f'requests {round((spans["requests"][1] - a) / pitch)}, httpx {round((b - spans["httpx"][0]) / pitch)}', FSS, INK, ha='center')
    ev = au['denominator_distortion']['observed_judgable']; zf = au['denominator_distortion']['zero_imputed_53_unrun_tasks']
    k0, k1, kt, kb = 5.53, W - .03, dots[-1] + .02, ya_bot + .05
    card(fx, k0, kb, k1, kt, 'sapphire')
    km = (k0 + k1) / 2
    lines = [(f'Evaluable {ref["resolved"]}/{ref["total"]}', pal.rim('sapphire'), True),
             (f'p {ev["p"]:.3f}   q/p {ev["q_over_p"]:.3f}', INK, False),
             (f'Zero-filled  p {zf["p"]:.3f}', pal.rim('slate'), True),
             (f'Calibrated {ceiling[0]}/{ceiling[1]}', MUTED, False)]
    step = (kt - kb - .08) / len(lines); yv = kt - .04 - step / 2
    for s, col, bd in lines:
        T(fx, km, yv, s, FSS, col, bd, ha='center'); yv -= step
    fx.plot([k0 + .08, k1 - .08], [kt - .04 - 2 * step] * 2, color=pal.light('slate'), lw=.4, zorder=7)
    DATA['gemma_success_histogram'] = dict(hist); DATA['reference'] = {'blocked_wheels': n_wheel, 'blocked_env': n_env, 'ceiling': ceiling, 'gold_percent': gold}

    # ---- b: paired ATIF trajectory X-ray on one wall-clock ruler
    btop = head(f, fx, 0, W, 'b', 'Paired ATIF v1.7 trajectory X-ray', 'recorded event times on one 605 s ruler', y=ya_bot - .02)
    t0x, t1x = 1.10, 5.48
    tx = lambda t: t0x + t / 605 * (t1x - t0x)
    yax = .585
    lanes = {'good': 1.355, 'r2943': (1.14, 1.025), 'r3882': (.80, .685)}
    good = trace('rich_3905', 1)
    chain = recovery_chain(good)
    gr = att[('rich_3905', 1)]
    T(fx, 0, lanes['good'] + .05, 'rich_3905 R1', FS, pal.rim('sapphire'), True)
    T(fx, 0, lanes['good'] - .05, f'resolved, {float(gr["seconds"]):.1f} s', FSS, MUTED)
    ts = [s['extra']['elapsed_s'] for s in good['steps'] if s.get('tool_calls')]
    errs = [s['extra']['elapsed_s'] for s in good['steps'] if s.get('tool_calls') and '"error"' in obs(s)]
    fx.add_collection(LineCollection([[(tx(t), lanes['good'] - .05), (tx(t), lanes['good'] + .05)] for t in ts if t not in errs],
                                     colors=pal.line('cobalt'), linewidths=.45, zorder=3, gid='texture'))
    fx.add_collection(LineCollection([[(tx(t), lanes['good'] - .05), (tx(t), lanes['good'] + .05)] for t in errs],
                                     colors=pal.line('coral'), linewidths=.6, zorder=3, gid='texture'))
    # magnified recovery chain above the lane
    cy_ = 1.665
    lo_t, hi_t = chain[0][0], chain[-1][0]
    probe = [T(fx, 0, 0, f'{t:.1f} s  {lab}', FSS) for t, lab in chain]
    ws = [width(f, p) + .08 for p in probe]
    for p in probe: p.remove()
    gapc = .17; span = sum(ws) + gapc * (len(ws) - 1); cx = t1x - span
    loupe = Polygon([(tx(lo_t) - .01, lanes['good'] + .065), (tx(hi_t) + .01, lanes['good'] + .065), (cx + span, cy_ - .075), (cx, cy_ - .075)],
                    fc=sheet.tint('cobalt', .10), ec='none', zorder=1, gid='texture')
    fx.add_patch(loupe)
    for (t, lab), w_ in zip(chain, ws):
        role = 'coral' if lab == 'edit mismatch' else 'cobalt'
        sheet.chip(fx, cx + w_ / 2, cy_, f'{t:.1f} s  {lab}', role, size=FSS, ha='center', face=PAPER if role == 'cobalt' else pal.wash('coral'), pad=.25)
        if lab != chain[-1][1]:
            fx.annotate('', xy=(cx + w_ + gapc - .025, cy_), xytext=(cx + w_ + .025, cy_),
                        arrowprops=dict(arrowstyle='-|>', lw=.5, color=pal.line('slate'), mutation_scale=4.5), zorder=6)
        cx += w_ + gapc
    T(fx, 5.58, lanes['good'], f'{gr["schema_errors"]} schema rejects', FSS, pal.rim('sapphire'))
    T(fx, 5.58, cy_, f'magnified {lo_t:.0f} to {hi_t:.0f} s', FSS, MUTED)

    cases = {}
    for key, task in (('r2943', 'rich_2943'), ('r3882', 'rich_3882')):
        tr = trace(task, 3); case = next(c for c in au['cases'] if c['task'] == task and c['round'] == 3)
        yu, yl_ = lanes[key]; ym = (yu + yl_) / 2
        T(fx, 0, ym + .095, f'{task} R3', FS, pal.rim('coral'), True)
        T(fx, 0, ym, f'{case["llm_calls"]} LLM calls', FSS, INK)
        T(fx, 0, ym - .09, f'{case["tool_calls"]}/100 counted tools', FSS, INK)
        sch, ok = [], []
        for s in tr['steps']:
            if not s.get('tool_calls'): continue
            t = s.get('extra', {}).get('elapsed_s')
            if t is None: continue
            (sch if SCHEMA in obs(s) else ok).append(t)
        fx.add_collection(LineCollection([[(tx(t), yu - .042), (tx(t), yu + .042)] for t in ok], colors=pal.line('cobalt'), linewidths=.5, zorder=3, gid='texture'))
        fx.add_collection(LineCollection([[(tx(t), yl_ - .042), (tx(t), yl_ + .042)] for t in sch], colors=pal.line('coral'), linewidths=.32, zorder=3, gid='texture'))
        fz = case['fuse_schema_3']
        fx.plot([tx(fz)] * 2, [yl_ - .065, yu + .065], color=pal.line('amber'), lw=.9, zorder=4)
        T(fx, 5.58, yu, f'{case["schema_errors"]} uncounted rejects', FSS, pal.rim('coral'))
        T(fx, 5.58, yl_, f'3rd reject {fz:.1f} s, ends {case["seconds"]:.1f} s', FSS, pal.rim('amber'))
        cases[task] = {'schema_times': sch, 'tool_times': ok, 'fuse_schema_3': fz, 'case': case}
    for t in range(0, 601, 100):
        fx.plot([tx(t)] * 2, [yax + .03, 1.42], color=pal.wash('slate') if t else pal.light('slate'), lw=.35, zorder=.5, gid='texture')
        fx.plot([tx(t)] * 2, [yax, yax - .025], color=cel.LINE, lw=.45)
        T(fx, tx(t), yax - .07, f'{t} s' if t == 600 else str(t), FSS, INK, ha='center')
    fx.plot([t0x, tx(605)], [yax, yax], color=cel.LINE, lw=.55)
    fx.plot([tx(600)] * 2, [yax + .02, 1.43], color=pal.line('coral'), lw=1.5, zorder=4.5)
    T(fx, tx(600) - .04, 1.43, '600 s cap', FSS, pal.rim('coral'), ha='right', va='bottom')
    
    legend_row(f, fx, 0, btop - .08, [('tick', 'cobalt', 'counted tool call', None), ('tick', 'coral', 'schema reject', None),
                                       ('tick', 'amber', '3-strike breaker', None)], gap=.1)

    # crystal cards: the payload and the replay (centered and concise)
    first = next(obs(s) for s in trace('rich_3882', 3)['steps'] if SCHEMA in obs(s))
    m = re.search(r'Invoking `(\w+)\(\)` failed as the following mandatory input parameters are not present:\\\\n(\w+)', first)
    tool, param = m.groups()
    gc = au['ghost_costs']
    rp = au['queue']['assumed_60min_sensitivity']['0.0']['paired_schema3_minus_60_solved']
    c0, c1, cb, ct = 0, 3.37, .045, .43
    card(fx, c0 + .01, cb, c1, ct, 'coral', fc=pal.wash('coral'))
    cm = (c0 + .01 + c1) / 2
    T(fx, cm, ct - .08, 'ATIF schema rejection (rich_3882 R3)', FSS, pal.rim('coral'), True, ha='center')
    T(fx, cm, ct - .19, f'{tool}(): missing {param}', FSS - .1, INK, family='DejaVu Sans Mono', ha='center')
    T(fx, cm, ct - .295, f'{gc["schema_rejected_calls"]:,} uncounted calls across {gc["preceding_event_intervals_seconds"]:,.1f} s', FSS, INK, ha='center')
    d0 = 3.49
    card(fx, d0, cb, W - .03, ct, 'amber', fc=pal.wash('amber'))
    dm = (d0 + W - .03) / 2
    T(fx, dm, ct - .08, '12 h queue replay (10-min cap + 3-strike breaker)', FSS, pal.rim('amber'), True, ha='center')
    T(fx, dm, ct - .19, f'+{rp["mean"]:.2f} solved tasks', FS + 1.2, pal.rim('amber'), True, ha='center')
    T(fx, dm, ct - .295, f'95% CI [{rp["task_bootstrap_95ci"][0]:.1f}, {rp["task_bootstrap_95ci"][1]:.1f}] vs 60-min timeout', FSS, INK, ha='center')
    DATA['xray'] = {'recovery_chain': chain, 'cases': cases, 'payload': {'tool': tool, 'param': param}, 'ghost_costs': gc, 'replay_12h': rp}
    save(f, 'fig2_gemma_xray_and_sandbox_waterline')


# ================================================================ Figure 3
def value_rows():
    grid = rows('evidence/results/decision_value_grid.csv')
    base = {cp: json.loads(raw(f'evidence/data/runtime_rows_{cp}.json')) for cp in (5, 10, 15)}
    out = []
    for r in grid:
        cp = int(r['checkpoint']); split = r['set']
        rr = [s for s in base[cp] if s['system'] == r['system'] and (
            (split == 'heldout' and s['split'] == 'verified' and hashlib.sha256((s['instance_id'] + ':train').encode()).digest()[0] >= 153)
            or (split != 'heldout' and s['split'] == split))]
        assert len(rr) == int(r['n'])
        C = sum(s['full'] for s in rr); lam = sum(s['resolved'] for s in rr) / C
        R = float(r['saved_calls']); Y = float(r['sacrificed'])
        assert abs(.5 * C * (lam * R - Y) / (C - R) - float(r['identity_ratio'])) < .001
        out.append(dict(r, C=C, lam=lam, x=R / C, yn=Y / (lam * C), gain=float(r['replay_gain'])))
    DATA['value_coordinates'] = out
    return out


def fig3():
    dv = js('decision_value'); vr = value_rows()
    H = 3.30
    f, fx = figure(H)
    xa1 = 3.35; xb = 3.55
    top = head(f, fx, 0, xa1, 'a', 'Stopping-value manifold', f'{len(vr)} policies, three benchmarks')
    ax = axes(f, .40, .34, xa1 - .04, top - .05, (-.012, .70), (-.02, .875))
    X, Y = np.meshgrid(np.linspace(0, .70, 220), np.linspace(-.02, .875, 220)); Z = (X - Y) / (1 - X)
    levels = [-2, -.5, -.25, 0, .25, .5, 2]
    cols = [pal.mix(pal.wash('coral'), pal.light('coral'), .55), pal.mix(PAPER, pal.wash('coral'), .9), pal.mix(PAPER, pal.wash('coral'), .45),
            pal.mix(PAPER, pal.wash('cobalt'), .5), pal.wash('cobalt'), pal.mix(pal.wash('cobalt'), pal.light('cobalt'), .55)]
    ax.contourf(X, Y, Z, levels=levels, colors=cols, zorder=.1)
    cs = ax.contour(X, Y, Z, levels=[-.5, -.25, .25, .5], colors=[pal.spec('slate')], linewidths=.4, zorder=.2)
    ax.clabel(cs, fmt=lambda v: f'{v:+.2f}', fontsize=FSS, inline=True, inline_spacing=2,
                     manual=[(.665, .5 * .665 + .5), (.665, .75 * .665 + .25), (.56, 1.25 * .56 - .25), (.645, 1.5 * .645 - .5)])
    ax.plot([0, .70], [0, .70], color=pal.rim('slate'), lw=.8, ls=(0, (3, 1.6)), zorder=3)
    gmax = max(abs(r['gain']) for r in vr)
    for r in sorted(vr, key=lambda r: -abs(r['gain'])):
        role = 'cobalt' if r['gain'] >= 0 else 'coral'
        d = 2.6 + 9.5 * np.sqrt(abs(r['gain']) / gmax)
        if r['set'] == 'heldout': bead(ax, r['x'], r['yn'], role, d)
        else:
            diamond_mark(ax, r['x'], r['yn'], role, d)
    ax.set_xticks([0, .2, .4, .6]); ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_xlabel('Saved calls  $R/C$', labelpad=1); ax.set_ylabel(r'Sacrificed repairs  $Y/(\lambda C)$', labelpad=2)
    T(ax, .52, .085, 'positive value', FSS, pal.rim('cobalt'), bold=True, ha='center')
    T(ax, .52, .035, r'below break-even $Y/R=\lambda$', FSS, pal.rim('cobalt'), ha='center')
    T(ax, .38, .82, 'negative value', FSS, pal.rim('coral'), bold=True, ha='center')
    ml = next(r for r in vr if r['set'] == 'multilingual' and r['system'] == 'mini' and r['checkpoint'] == '5' and r['family'] == 'source')
    sheet.ring(ax, ml['x'], ml['yn'], d=12, role='amber', lw=.7)
    c1 = dv['C1_value_identity']['ratio_form']
    mls = dv['C3_breakeven_screen']['selected']['mini']['multilingual']
    t = sheet.chip(ax, .012, .86,
                   r'$\Delta G = B(\lambda R - Y)/(C - R)$' + f'\n$R^2 = {c1["r2_identity_as_prediction"]:.3f}$, sign match {c1["sign_agreement"] * 100:.1f}%\nMultilingual drift: {mls["replay_gain"]:.2f}',
                   'amber', size=FSS, ha='left', va='top', face=PAPER, pad=.35, color=INK); t.set_multialignment('center')
    ax.plot([ml['x'], .05], [ml['yn'] + .02, .655], color=pal.line('amber'), lw=.5, zorder=4)
    lx, ly = .44, .275
    legend_items = [('heldout', 'held-out Verified'), ('transfer', 'Lite, Multilingual transfer')]
    for k, (kind, lab) in enumerate(legend_items):
        y_ = ly - k * .045
        if kind == 'heldout': bead(ax, lx, y_, 'cobalt', 4.6)
        else: diamond_mark(ax, lx, y_, 'cobalt', 5.0)
        T(ax, lx + .018, y_, lab, FSS, INK)
    T(ax, lx - .008, ly - .095, f'area ∝ |replay gain|, max {gmax:.0f}', FSS, MUTED)
    DATA['multilingual_transfer'] = ml

    # ---- b: AUC against calls saved per stop
    btop = head(f, fx, xb, W - xb, 'b', 'AUC orthogonality', 'held-out Verified policies', bottom=1.76)
    ab = axes(f, xb + .36, 2.06, W - .03, btop - .05, (.40, .80), (-4, 104))
    ab.axvspan(.48, .52, color=pal.wash('slate'), zorder=.2)
    T(ab, .50, 96, 'chance', FSS, MUTED, ha='center')
    hv = [r for r in vr if r['set'] == 'heldout']
    for r in sorted(hv, key=lambda r: -abs(r['gain'])):
        bead(ab, float(r['auc']), float(r['saved_calls']) / max(1, float(r['stops'])), 'cobalt' if r['gain'] >= 0 else 'coral',
             2.6 + 6.5 * np.sqrt(abs(r['gain']) / gmax))
    oh = dv['C3_breakeven_screen']['selected']['openhands']['heldout']
    cps = oh['saved_calls'] / oh['stops']
    sheet.ring(ab, oh['auc'], cps, d=13, role='amber', lw=.8)
    t = sheet.chip(ab, .795, 101, f'OpenHands gate (AUC {oh["auc"]:.3f})\n{oh["stops"]} stops, {oh["saved_calls"]:,.0f} calls: +{oh["replay_gain"]:.2f}',
               'amber', size=FSS, ha='right', va='top', face=PAPER, pad=.3, color=INK); t.set_multialignment('center')
    ab.plot([oh['auc'] + .006, .58], [cps + 5, 79], color=pal.line('amber'), lw=.5, zorder=4)
    ab.set_xticks([.4, .5, .6, .7]); ab.set_yticks([0, 25, 50, 75, 100])
    ab.set_xlabel('Held-out ROC AUC', labelpad=1); ab.set_ylabel('Calls saved per stop', labelpad=2)

    # ---- c: four-system prospective selector bridge
    ctop = head(f, fx, xb, W - xb, 'c', 'Prospective selector bridge', 'train-time choice, held-out repairs', y=1.74)
    per = dv['C2_auc_vs_value']['per_system']
    names = {'sweagent': 'SWE-agent', 'agentless': 'Agentless 1.5', 'openhands': 'OpenHands', 'mini': 'Mini SWE Agent'}
    ac = axes(f, xb + .80, .33, W - .03, ctop - .17, (-4.2, 4.2), (-.6, 3.55))
    ac.spines['left'].set_position(('data', -4.2))
    ac.set_yticks([3, 2, 1, 0]); ac.set_yticklabels([names[s] for s in per], fontsize=FSS)
    ac.tick_params(axis='y', length=2, pad=2)
    ac.axvline(0, color=pal.light('slate'), lw=.5, zorder=.5)
    tot_auc = sum(v['selected_by_cv_auc'][2] for v in per.values()); tot_val = sum(v['selected_by_cv_value'][2] for v in per.values())
    for i, (s, v) in enumerate(per.items()):
        y = 3 - i; u = v['selected_by_cv_auc'][2]; w_ = v['selected_by_cv_value'][2]
        ac.plot([u, w_], [y, y], color=pal.spec('sapphire'), lw=2.4, solid_capstyle='butt', zorder=2)
        sheet.squares(ac, [u], [y], 'coral', 1, d=5.4)
        bead(ac, w_, y, 'sapphire', 6, 2)
    ac.set_xticks([-4, -2, 0, 2, 4]); ac.set_xlabel('Held-out net repairs', labelpad=1)
    legend_row(f, fx, xb + .05, ctop - .08, [('square', 'coral', f'CV-AUC pick, sum {tot_auc:+.2f}', None),
                                            ('disc', 'sapphire', f'CV-value pick, sum {tot_val:+.2f}', None)], gap=.14)
    diff = dv['C2_auc_vs_value']['pooled']['repairs_forfeited_selecting_by_cv_auc_instead_of_cv_value_sum4']
    t = sheet.chip(ac, -4.0, 2.5, f'+{diff:.1f} held-out\nrepairs over AUC', 'amber', size=FS, ha='left', va='center', face=pal.wash('amber'), pad=.3, bold=True)
    t.set_multialignment('center')
    DATA['selectors'] = {'per_system': per, 'sum_auc': tot_auc, 'sum_value': tot_val, 'difference': diff, 'openhands_gate': oh}
    save(f, 'fig3_stopping_value_manifold')


# ================================================================ main
def main():
    global AUDIT
    OUT.mkdir(exist_ok=True, parents=True); PREVIEW.mkdir(exist_ok=True, parents=True)
    for p in (BASE / 'aimo_palette.py', BASE / 'aimo_cel.py', ARC / 'cel.py', ARC / 'figcheck.py'): raw(p)
    AUDIT = js('gemma4_sandbox_and_schema_audit')
    fig1(); fig2(); fig3()
    for old, new in [('fig1_schedule_and_persistence', 'fig1_persistence_and_queue_crossing'),
                     ('fig2_value_identity_and_gemma_audit', 'fig2_gemma_xray_and_sandbox_waterline')]:
        for ext in ('pdf', 'png'): shutil.copyfile(OUT / (new + '.' + ext), OUT / (old + '.' + ext))
    payload = dict(canvas_width_in=W, font_pt={'body': FS, 'tick': FSS, 'floor': figcheck.MIN}, sources_sha256=SOURCES,
                   checks=CHECKS, notes=NOTES, data=DATA, official_gain=0)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=float) + '\n'
    (OUT / 'publication_figures_provenance.json').write_text(text)
    (OUT / 'figure_provenance.json').write_text(text)
    html = ('<html><meta charset="utf-8"><title>Plates S4</title><style>body{margin:32px auto;max-width:1300px;font:16px Georgia;'
            'color:#1E2530;background:#fff}img{width:100%;border:1px solid #E9EDF7}article{margin:36px 0}a{color:#22356A}'
            '.p{width:6.85in}</style><h1>Plates S4, print size 6.85 in</h1>')
    for n, c in CHECKS.items():
        html += (f'<article><h2>{n}</h2><p>hits {c["hit_count"]}, green px {c["green_px"]}, pdf {c["pdf_bytes"]:,} B, '
                 f'<a href="../../../paper/figures/{n}.pdf">vector PDF</a></p><img src="../../../paper/figures/{n}.png">'
                 f'<p>At print size:</p><img class="p" src="../../../paper/figures/{n}.png"></article>')
    (PREVIEW / 'index.html').write_text(html + '</html>')
    print(json.dumps(CHECKS, indent=2, default=float))
    if any(v['hit_count'] or v['green_px'] or v['pdf_bytes'] >= 1_000_000 for v in CHECKS.values()): sys.exit(2)


if __name__ == '__main__':
    main()
