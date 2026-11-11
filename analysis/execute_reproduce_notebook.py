"""Execute the offline notebook and record measured time, memory and cell checks."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'paper/reproduce.ipynb'
env = os.environ.copy()
env['JUPYTER_PATH'] = str(Path(sys.prefix)/'share/jupyter')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_MAX_THREADS'):
    env[key] = '1'
command = [sys.executable,'-m','nbconvert','--to','notebook','--execute','--inplace',
           '--ExecutePreprocessor.timeout=60',str(path)]
start = time.monotonic()
completed = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
elapsed = time.monotonic()-start
if completed.returncode:
    raise RuntimeError(completed.stdout+'\n'+completed.stderr)
notebook = json.loads(path.read_text())
code = [c for c in notebook['cells'] if c['cell_type']=='code']
assert all(c['execution_count'] is not None for c in code)
assert not any(o.get('output_type')=='error' for c in code for o in c['outputs'])
tail = ''.join(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else o.get('text','')
               for o in code[-1]['outputs'] if o['output_type']=='stream')
kernel = json.loads(tail)
assert kernel['execution_checks']=='passed' and kernel['kernel_peak_rss_bytes'] < 4*1024**3
assert elapsed < 60
receipt = {'command':['jupyter','nbconvert','--to','notebook','--execute','--inplace','--ExecutePreprocessor.timeout=60','paper/reproduce.ipynb'],'returncode':completed.returncode,'wall_seconds':elapsed,
           'kernel':kernel,'code_cells_executed':len(code),'error_outputs':0,
           'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'versions':{m:version(m) for m in ('numpy','pandas','pyarrow','duckdb','nbconvert','ipykernel','matplotlib','Pillow')},
           'passed': kernel['publication_checks'] == 'passed',
           'scope':'Four publication tables; per-cell comparison with the locked manuscript; compact attempt and saved-draw validation; three vector figure builds and checks. Full scheduling draw generation is a separate build.'}
output = ROOT/'evidence/results/notebook_execution.json'
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
