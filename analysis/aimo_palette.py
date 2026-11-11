"""The report's one role palette: eight color-vision-safe families on white paper, built on the Okabe-Ito set.

There is no green anywhere: every step of every family has its green channel below its red or its blue channel, so a
deuteranope or protanope reads each contrast as blue against orange, never red against green.
sapphire   navy: the full system C and the main track
cobalt     Okabe-Ito blue: the white-box readout M_wb, robust rows and the small track; also the final layer L
amethyst   streaming sibling anchors P_sib, Olmo and the middle layer L/2
amber      gold: exact verification V_sym and the single focus reading of a panel
coral      Okabe-Ito vermilion, a warm terracotta: right-censoring at the 6,144-token cap, non-robust rows and vetoes
burgundy   premise removal (no_solution) and the subtractive C minus M_wb ablation
glacier    Okabe-Ito sky blue: the shallow layer L/4 and DeepSeek
slate      every baseline: text classifier, answer agreement, pooled probe, and the numerical null q
Each family has five flat steps, deep to pale: rim (outline of the deepest marks), body (stroke), mid, spec, tint
(wash); `pale` sits between spec and tint as the fill of a pale mark.
"""
from matplotlib.colors import to_hex, to_rgb


def mix(a, b, t):
    a, b = to_rgb(a), to_rgb(b)
    return to_hex([x + (y - x) * t for x, y in zip(a, b)])


FAMILY = {  # name: (rim, body, mid, spec, tint)
    'sapphire': ('#121E3D', '#22356A', '#3E5A9A', '#97A9D6', '#E9EDF7'),
    'cobalt': ('#062B4A', '#0B5A96', '#2382C8', '#8CC0E8', '#E5F1FB'),
    'amethyst': ('#2D1E4A', '#563D82', '#7B5EA7', '#B59ED9', '#EFEBF7'),
    'amber': ('#5C3A08', '#B07219', '#D99B26', '#F2C977', '#FDF4E3'),
    'coral': ('#561E05', '#A9420C', '#D5601C', '#F0A679', '#FCEEE5'),
    'burgundy': ('#4A152B', '#8C2D52', '#B84A72', '#E08AA8', '#F9EBF0'),
    'glacier': ('#153F55', '#2B7EA6', '#56AAD6', '#A6D3EE', '#EBF5FB'),
    'slate': ('#262A30', '#4A525D', '#737D8C', '#B4BCC8', '#F1F3F6'),
}
RIM = {k: v[0] for k, v in FAMILY.items()}
SPEC = {k: v[3] for k, v in FAMILY.items()}
PALE = {k: mix(v[3], v[4], 0.55) for k, v in FAMILY.items()}
# (stroke, mid, fill, wash): the four steps every mark helper reads
HUES = {k: (v[1], v[2], PALE[k], v[4]) for k, v in FAMILY.items()}

ROLE = {
    'full': 'sapphire', 'main': 'sapphire',
    'wb': 'cobalt', 'rob': 'cobalt', 'small': 'cobalt', 'l1': 'cobalt',
    'pool': 'amethyst', 'sib': 'amethyst', 'l2': 'amethyst', 'olmo': 'amethyst',
    'sym': 'amber', 'focus': 'amber', 'sky': 'amber',
    'cut': 'coral', 'non': 'coral', 'veto': 'coral',
    'minus': 'burgundy', 'nosol': 'burgundy',
    'l4': 'glacier', 'ds': 'glacier',
    'base': 'slate', 'gray': 'slate', 'null': 'slate', 'qwen': 'slate',
}
MODEL_ROLE = {'Qwen': 'qwen', 'Skywork': 'sky', 'Olmo': 'olmo', 'DeepSeek': 'ds', 'gpt-oss': 'main'}
INK = '#1E2530'
MUTED = '#5A6270'
PAPER = '#FFFFFF'


def hue(role):
    return ROLE.get(role, role)


def line(role):
    return HUES[hue(role)][0]


def mid(role):
    return HUES[hue(role)][1]


def light(role):
    return HUES[hue(role)][2]


def wash(role):
    return HUES[hue(role)][3]


def rim(role):
    return RIM[hue(role)]


def spec(role):
    return SPEC[hue(role)]
