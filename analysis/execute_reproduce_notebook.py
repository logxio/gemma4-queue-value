"""Execute every offline notebook cell and record numeric, memory and figure checks."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'paper/reproduce.ipynb'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
    os.environ[key] = '1'

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

notebook = nbformat.read(path, as_version=4)
shell = InteractiveShell.instance()
shell.user_ns['__name__'] = '__main__'
start = time.monotonic()
previous_cwd = Path.cwd()
os.chdir(ROOT)
try:
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise error
        cell.execution_count = count
        cell.outputs = []
        for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
                print(value, end='', file=sys.stdout if name == 'stdout' else sys.stderr)
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
finally:
    os.chdir(previous_cwd)
elapsed = time.monotonic() - start
code = [c for c in notebook.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type == 'error' for c in code for o in c.outputs)
tail = ''.join(o.text for o in code[-1].outputs if o.output_type == 'stream')
kernel = json.loads(tail)
assert kernel['execution_checks'] == 'passed' and kernel['kernel_peak_rss_bytes'] < 4 * 1024**3
assert kernel['publication_checks'] == 'passed'
# Publish the executed notebook only after every cell and all numeric checks pass.
pending = path.with_suffix('.pending.ipynb')
nbformat.write(notebook, pending)
pending.replace(path)
receipt = {
    'command': [sys.executable, 'analysis/execute_reproduce_notebook.py'],
    'execution_mode': 'in-process IPython; every code cell executed in notebook order',
    'returncode': 0, 'wall_seconds': elapsed, 'kernel': kernel,
    'code_cells_executed': len(code), 'error_outputs': 0,
    'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'versions': {m: version(m) for m in ('numpy', 'pandas', 'pyarrow', 'duckdb', 'nbformat', 'ipython', 'matplotlib', 'Pillow')},
    'passed': True,
    'scope': 'Four publication tables; per-cell comparison with the locked manuscript; compact attempt and saved-draw validation; three read-only vector figure checks. Full scheduling draw generation is a separate build.'
}
output = ROOT / 'evidence/results/notebook_execution.json'
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
