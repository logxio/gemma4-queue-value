"""Synchronize the locked Markdown into a screen-readable academic PDF."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
STEM = 'gemma4_queue_value_paper'


def command(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(result.stdout[-5000:] + '\n' + result.stderr[-1000:])
    return result.stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latex(blocks, api):
    result = command(['pandoc', '-f', 'json', '-t', 'latex', '--wrap=none', '--no-highlight'],
                     input=json.dumps({'pandoc-api-version': api, 'meta': {}, 'blocks': blocks}))
    # Range typography uses an ASCII hyphen; mathematical negative signs stay mathematical.
    return result.strip().replace('\N{EN DASH}', r'\mbox{-}').replace('\N{MINUS SIGN}', r'\ensuremath{-}').replace('²', r'\textsuperscript{2}').replace('×', r'\ensuremath{\times}')


def code_inline(node):
    if isinstance(node, list):
        return [code_inline(v) for v in node]
    if not isinstance(node, dict):
        return node
    if node.get('t') == 'Code':
        text = node['c'][1]
        escaped = ''
        for c in text:
            escaped += {'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '{': r'\{', '}': r'\}', '$': r'\$'}.get(c, c)
            if c in '_/.,': escaped += r'\allowbreak{}'
        return {'t': 'RawInline', 'c': ['latex', escaped]}
    return {k: code_inline(v) for k, v in node.items()}


def inlines(items, api):
    return latex([{'t': 'Plain', 'c': items}], api)


def table(block, caption, index, api):
    _, _, columns, head, bodies, foot = block['c']
    rows = list(head[1])
    for body in bodies:
        rows.extend(body[2]); rows.extend(body[3])
    rows.extend(foot[1])
    specs = {
        1: r'@{}>{\hsize=1.9\hsize}L S S S S S >{\hsize=1.1\hsize}R@{}',
        2: r'@{}>{\hsize=1.35\hsize}L S S S S >{\hsize=1.15\hsize}R >{\hsize=1.15\hsize}R >{\hsize=1.15\hsize}R@{}',
        3: r'@{}>{\hsize=1.9\hsize}L >{\hsize=.65\hsize}R >{\hsize=.65\hsize}R >{\hsize=.9\hsize}R >{\hsize=.7\hsize}R >{\hsize=1.2\hsize}L@{}',
        4: r'@{}>{\hsize=1.7\hsize}L S S S S S >{\hsize=1.3\hsize}R@{}',
    }
    lines = [r'\begin{table}[htbp]', r'\caption*{' + caption + '}', r'\centering',
             r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\renewcommand{\arraystretch}{1.18}', r'\setlength{\extrarowheight}{2.5pt}',
             r'\begin{tabularx}{\textwidth}{' + specs[index] + '}', r'\toprule']
    for j, row in enumerate(rows):
        cells = [latex(cell[4], api) for cell in row[1]]
        if index == 2:
            cells = [s.replace(' [', r'\newline[') for s in cells]
        if j == 0:
            cells = [r'\textbf{' + c + '}' for c in cells]
        lines.append(' & '.join(cells) + r' \\')
        if j == 0: lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabularx}', r'\end{table}']
    return '\n'.join(lines)


PREAMBLE = r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm,top=17mm,bottom=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,times,newtxmath}
\renewcommand{\ttdefault}{ptm}
\usepackage{booktabs,array,tabularx,graphicx,microtype}
\usepackage{caption,float,enumitem}
\makeatletter
\newcommand{\FloatBarrier}{\par\ifx\@deferlist\@empty\else\newpage\fi}
\makeatother
\usepackage{xcolor}
\definecolor{linkblue}{HTML}{1E3A8A}
\usepackage[colorlinks=true,urlcolor=linkblue,linkcolor=linkblue,citecolor=linkblue,unicode]{hyperref}
\urlstyle{same}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{S}{>{\hsize=.8\hsize\raggedleft\arraybackslash}X}
\captionsetup{font=small,labelfont=bf,skip=6pt,justification=raggedright,singlelinecheck=false}
\setlength{\aboverulesep}{3pt}
\setlength{\belowrulesep}{3pt}
\setlength{\textfloatsep}{12pt plus 1pt minus 1pt}
\setlength{\floatsep}{10pt plus 1pt minus 1pt}
\setlength{\intextsep}{10pt plus 1pt minus 1pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3.5pt plus 1pt minus 0.5pt}
\setlist{nosep,leftmargin=1.6em,topsep=4pt}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{.95}
\renewcommand{\bottomfraction}{.9}
\renewcommand{\textfraction}{.05}
\renewcommand{\floatpagefraction}{.8}
\makeatletter
\setlength{\@fptop}{0pt}\setlength{\@fpsep}{10pt}
\renewcommand{\section}{\@startsection{section}{1}{\z@}{-1.6ex plus -.2ex minus -.2ex}{.65ex}{\large\bfseries\raggedright}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}{-1.1ex plus -.2ex minus -.1ex}{.5ex}{\normalsize\bfseries\raggedright}}
\makeatother
\raggedbottom
\emergencystretch=1.5em
\widowpenalty=5000
\clubpenalty=5000
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\begin{document}
\small
'''


def synchronize():
    source = PAPER / 'writeup.md'
    before = sha(source)
    ast = json.loads(command(['pandoc', '-f', 'markdown-smart-implicit_figures', '-t', 'json', str(source)]))
    api = ast['pandoc-api-version']
    original = ast['blocks']
    blocks = code_inline(original)
    out = [PREAMBLE]
    index = 0
    table_index = figure_index = 0
    figures = []
    while index < len(blocks):
        block = blocks[index]
        kind, content = block['t'], block.get('c')
        if index == 0:
            title = inlines(content[2], api)
            subtitle = latex([blocks[index+1]], api)
            subtitle = subtitle.replace('endogenous retry crossing, and runtime', r'endogenous retry crossing,\\ and runtime')
            keywords = latex([blocks[index+2]], api)
            out += [r'\begin{center}', r'{\LARGE\bfseries ' + title + r'\par}', r'\vspace{5pt}',
                    r'{\normalsize ' + subtitle + r'\par}', r'\vspace{5pt}',
                    r'{\small ' + keywords + r'\par}', r'\end{center}', r'\vspace{2pt}']
            index += 3
            continue
        if kind == 'Header':
            title = inlines(content[2], api)
            if title == 'Abstract':
                out += [r'\begin{center}\textbf{Abstract}\end{center}', r'\begingroup\small', latex([blocks[index+1]], api), r'\endgroup\medskip']
                index += 2
                continue
            level = 'section' if content[0] == 2 else 'subsection'
            if title == 'References':
                out += [r'\clearpage', r'\section*{References}', r'\begingroup\small',
                        r'\setlength{\baselineskip}{12pt}', r'\setlength{\parskip}{2pt}']
            else:
                out.append('\\' + level + '*{' + title + '}')
        elif kind == 'Para' and index+1 < len(blocks) and blocks[index+1]['t'] == 'Table':
            table_index += 1
            out.append(table(blocks[index+1], inlines(content[0]['c'], api), table_index, api))
            index += 2
            continue
        elif kind == 'Para' and len(content) == 1 and content[0]['t'] == 'Image':
            figure_index += 1
            target = content[0]['c'][2][0]
            path = Path(target).with_suffix('.pdf')
            figures.append(path.as_posix())
            caption = blocks[index+1]['c'][0]['c']
            placement = 'H' if figure_index == 1 else 'htbp'
            out += [rf'\begin{{figure}}[{placement}]', r'\centering',
                    r'\includegraphics[width=\textwidth]{' + (Path('.build/outlined') / path.name).as_posix() + '}',
                    r'\caption*{' + inlines(caption, api) + '}', r'\end{figure}']
            if figure_index == 1:
                out += [r'\newpage', r'\normalsize']
            index += 2
            continue
        else:
            out.append(latex([block], api))
        index += 1
    out += [r'\endgroup', r'\end{document}']
    text = '\n\n'.join(out) + '\n'
    assert '\N{EM DASH}' not in text and '\N{EN DASH}' not in text
    (PAPER / (STEM + '.tex')).write_text(text)
    assert sha(source) == before
    receipt = {'source_sha256': before, 'tables': table_index, 'figures': figures,
               'source_blocks': len(original), 'text_sync': 'All blocks converted in source order using Pandoc AST; captions and table cells retained verbatim; range glyphs use ASCII hyphens',
               'template_basis': 'single-column A4 Times/NewTX template'}
    return receipt


def build():
    receipt = synchronize()
    receipt['figure_sha256'] = {p: sha(PAPER / p) for p in receipt['figures']}
    receipt['locked_figure_sha256'] = {
        f'figures/{Path(name).stem}{suffix}': sha(PAPER / f'figures/{Path(name).stem}{suffix}')
        for name in receipt['figures'] for suffix in ('.pdf', '.png')
    }
    outlined = PAPER / '.build/outlined'
    outlined.mkdir(parents=True, exist_ok=True)
    receipt['embedded_figures'] = {}
    for name in receipt['figures']:
        source = PAPER / name
        target = outlined / source.name
        command(['gs', '-q', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite', '-dNoOutputFonts',
                 '-dCompatibilityLevel=1.7', '-sOutputFile=' + str(target), str(source)])
        receipt['embedded_figures'][str(target.relative_to(PAPER))] = sha(target)
    receipt['figure_text_policy'] = 'Vector figure text is outlined without rasterization so pdftotext counts only title, abstract, prose, tables and captions; original editable vector PDFs retained'
    for _ in range(2):
        command(['pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', STEM + '.tex'], cwd=PAPER)
    pdf = PAPER / (STEM + '.pdf')
    cleaned = PAPER / (STEM + '.clean.pdf')
    command(['qpdf', '--remove-info', '--remove-metadata', str(pdf), str(cleaned)])
    from pypdf import PdfWriter
    writer = PdfWriter(clone_from=cleaned)
    writer.metadata = None
    writer.write(pdf)
    cleaned.unlink()
    receipt['pdf_sha256'] = sha(pdf)
    receipt['tex_sha256'] = sha(PAPER / (STEM + '.tex'))
    (ROOT / 'evidence/results/latex_sync_s2.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    build()
