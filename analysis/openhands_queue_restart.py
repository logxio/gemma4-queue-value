"""Issue-weighted repeated-attempt statistics and paired, causal queue replay.

Only completed failures expose their cost/empty-patch flag to the scheduler.
Assistant counts are recorded assistant-role messages, not billed API requests.
Run with the repository venv; no network or accelerator is used.
"""
import argparse
import csv
import hashlib
import heapq
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from queue_restart_decision import train_rates, PRIOR_CALLS

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
FRACTIONS = np.array([.2, .3, .4, .5])
MODES = ('plain_breadth', 'depth_first', 'ranked_breadth', 'deferred_rate', 'adaptive',
         'short_nonempty_first', 'long_empty_first', 'numeric27b_adaptive', 'numeric27b_deferred')


def boolean(value):
    value = str(value).strip().lower()
    if value in ('true', '1'): return True
    if value in ('false', '0'): return False
    raise ValueError(f'Unrecognized boolean: {value!r}')


def load(path):
    issues, repos, seen = defaultdict(list), {}, set()
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['trajectory_id'] in seen: raise ValueError('Duplicate trajectory ID')
            seen.add(row['trajectory_id'])
            issue = row['instance_id']
            repos[issue] = row['repo']
            costs = [int(row[k]) for k in ('assistant_calls', 'messages')]
            if min(costs) < 0: raise ValueError('Negative cost')
            issues[issue].append([boolean(row['resolved']), *costs, boolean(row['patch_empty']), boolean(row['model_EmptyPatch'])])
    return {i: np.asarray(v, dtype=float) for i, v in sorted(issues.items())}, repos


def interval(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(np.mean(values)), 'ci95': np.quantile(values, [.025, .975]).tolist()}


def estimates(n, k, weights=None):
    w = np.ones(len(n)) if weights is None else weights
    p = np.average(k / n, weights=w)
    fail = np.average(1 - k / n, weights=w)
    fs = np.average(k * (n-k) / (n*(n-1)), weights=w)
    joint = np.average(k * (k-1) / (n*(n-1)), weights=w)
    q = fs / fail if fail else float('nan')
    return np.array([p, q, q / p, (joint-p*p)/(p*(1-p))])


def statistics(issues, reps):
    ns = np.array([len(v) for v in issues.values()])
    ks = np.array([v[:, 0].sum() for v in issues.values()])
    mask = ns >= 2
    n, k = ns[mask], ks[mask]
    rng = np.random.default_rng(SEED+71)
    boots = []
    for _ in range(reps):
        ix = rng.integers(len(n), size=len(n))
        boots.append(estimates(n[ix], k[ix]))
    boots = np.asarray(boots)
    names = ('p', 'q', 'q_over_p', 'icc')
    point = estimates(n, k)
    out = {'eligible_issues': len(n), 'singletons_excluded': int((ns==1).sum()),
           'definitions': {'p': 'Uniform issue (n>=2), uniformly chosen first attempt success probability.',
             'q': 'Same issue sampling; uniform ordered distinct attempt pair, second success conditional on first failure; ratio of issue-averaged joint and failure probabilities.',
             'icc': 'Correlation of two uniformly chosen distinct binary attempts from a uniformly chosen issue: (E[Y1Y2]-p^2)/(p(1-p)). Equals 1-q/p for this estimand; not ANOVA ICC.',
             'ci': 'Percentile issue bootstrap; all observations from a sampled issue stay together.'},
           'equal_issue': {name: {'point': float(point[j]), **interval(boots[:,j])} for j,name in enumerate(names)},
           'attempt_weighted': dict(zip(names, estimates(n,k,n).tolist())),
           'ordered_pair_weighted': dict(zip(names, estimates(n,k,n*(n-1)).tolist())),
           'weighting_bias': {label: dict(zip(names, (estimates(n,k,w)-point).tolist())) for label,w in [('attempt_weighted_minus_equal_issue',n),('ordered_pair_weighted_minus_equal_issue',n*(n-1))]},
           'all_attempt_p': float(ks.sum()/ns.sum()),
           'attempt_count_selection': []}
    for count in sorted(set(ns)):
        m = ns == count
        out['attempt_count_selection'].append({'attempts_per_issue':int(count), 'issues':int(m.sum()),
            'p':float(np.mean(ks[m]/ns[m])), 'ever_solved':float(np.mean(ks[m]>0))})
    return out


def draw(issues, ids, rng):
    return np.asarray([issues[i][rng.choice(len(issues[i]), 3, replace=False)] for i in ids])


def schedule(s, c, empty, cats, rates, mode, order):
    """Budget-independent trace. Future costs/outcomes read only after selection."""
    fresh_rates, retry, edges = rates
    fresh, retries = [], []
    random_fresh = mode in ('plain_breadth', 'depth_first', 'short_nonempty_first', 'long_empty_first')
    for rank,i in enumerate(order):
        rate = fresh_rates.get(str(cats[i]), fresh_rates['__global__'])
        fresh.append((0. if random_fresh else -rate, rank, int(i)))
    heapq.heapify(fresh)
    slot = np.zeros(len(s), dtype=np.int8)
    serial = 0
    trace = []
    adaptive = mode in ('adaptive', 'numeric27b_adaptive')
    while fresh or retries:
        use_retry = bool(retries) and (not fresh or mode == 'depth_first' or
                                       (adaptive and retries[0][0][0] < fresh[0][0]))
        early = bool(fresh) if use_retry else False
        if use_retry: _,_,i = heapq.heappop(retries)
        else: _,_,i = heapq.heappop(fresh)
        j = int(slot[i])
        trace.append((float(c[i,j]), bool(s[i,j]), early))
        slot[i] += 1
        if s[i,j] or j == 2: continue
        count = j+1
        bucket = int(np.searchsorted(edges, float(c[i,:count].mean()), side='left'))
        key = (str(cats[i]), bucket if count == 1 else None, count)
        rate = retry.get(key, retry[(None,None,count)])
        if mode in ('adaptive','deferred_rate','numeric27b_adaptive','numeric27b_deferred'): pri=(-rate,count)
        elif mode == 'ranked_breadth': pri=(count,-rate)
        elif mode in ('short_nonempty_first','long_empty_first'):
            # Only already observed failures contribute. First failure tercile is train-only.
            observed_empty = bool(empty[i,:count].any())
            short_nonempty = bucket == 0 and not observed_empty
            long_or_empty = bucket == 2 or observed_empty
            group = 0 if short_nonempty else 2 if long_or_empty else 1
            pri=(group if mode == 'short_nonempty_first' else 2-group, count)
        else: pri=(count,0.)
        heapq.heappush(retries,(pri,serial,i)); serial += 1
    return np.asarray(trace)


def replay(issues, repos, reps, npz_path, checkpoint=None, batch_seconds=480):
    ids = [i for i,v in issues.items() if len(v)>=3]
    train = [i for i in ids if hashlib.sha256((i+':openhands-queue').encode()).digest()[0]<128]
    train_set = set(train); test = [i for i in ids if i not in train_set]
    counts=Counter(repos[i] for i in train)
    category=lambda i: repos[i] if counts[repos[i]]>=10 else 'other'
    rng=np.random.default_rng(SEED)
    pilot=draw(issues,train,rng); traincats=np.array([category(i) for i in train])
    calibrations={}
    for ci,name in enumerate(('assistant_calls','messages'),1):
        fresh,retry,edges,base=train_rates(pilot[:,:,0].astype(bool),pilot[:,:,ci],traincats)
        fresh['__global__']=base
        calibrations[name]=(fresh,retry,edges)
    frozen=json.loads((ROOT/'evidence/results/queue_restart_decision.json').read_text())
    # Serialized frozen 27B aggregate numerical rates, no OpenHands refit or unit rescaling.
    numeric=({'__global__':frozen['train_fresh_per_1000_calls']['__global__']/1000},
             {(None,None,int(k)):v/1000 for k,v in frozen['train_restart_global_per_1000_calls'].items()},
             frozen['failure_call_tercile_edges'])
    records={name:{mode:[] for mode in MODES} for name in calibrations}
    begin = 0
    signature = hashlib.sha256(Path(__file__).read_bytes() + (ROOT/'evidence/results/openhands_attempt_costs.csv').read_bytes()).hexdigest()
    if checkpoint and checkpoint.exists():
        with checkpoint.open('rb') as f: saved = pickle.load(f)
        if saved['signature'] != signature or saved['reps'] != reps: raise ValueError('Stale checkpoint')
        begin, records = saved['done'], saved['records']
        rng.bit_generator.state = saved['rng']
    batch_start = time.monotonic()
    for rep in range(begin,reps):
        sampled=[test[j] for j in rng.integers(len(test),size=len(test))]
        a=draw(issues,sampled,rng); cats=np.array([category(i) for i in sampled]); order=rng.permutation(len(test))
        for ci,name in enumerate(calibrations,1):
            budgets=FRACTIONS*a[:,:,ci].sum()
            for mode in MODES:
                rates=numeric if mode.startswith('numeric27b') else calibrations[name]
                tr=schedule(a[:,:,0].astype(bool),a[:,:,ci],a[:,:,3].astype(bool),cats,rates,mode,order)
                cum=np.cumsum(tr[:,0]); ran=np.searchsorted(cum,budgets,side='right')
                metrics=np.column_stack([np.r_[0,np.cumsum(tr[:,1])][ran],np.r_[0,cum][ran],np.r_[0,np.cumsum(tr[:,2])][ran]])
                records[name][mode].append(metrics)
        if (rep+1)%25==0:
            print(f'replicates {rep+1}/{reps}',flush=True)
            if checkpoint:
                with checkpoint.with_suffix('.tmp').open('wb') as f:
                    pickle.dump({'signature':signature,'reps':reps,'done':rep+1,'records':records,'rng':rng.bit_generator.state},f)
                checkpoint.with_suffix('.tmp').replace(checkpoint)
            if checkpoint and time.monotonic()-batch_start > batch_seconds and rep+1 < reps:
                print('Batch checkpoint saved; rerun identical command to resume.',flush=True)
                raise SystemExit(75)
    result={'replicates':reps,'train_issues':len(train),'heldout_issues':len(test),'eligible_issues':len(ids),
            'split':"SHA256(instance_id + ':openhands-queue').digest()[0] < 128 training",
            'train_ids_sha256':hashlib.sha256('\n'.join(train).encode()).hexdigest(),
            'test_ids_sha256':hashlib.sha256('\n'.join(test).encode()).hexdigest(),
            'calibration':'Frozen 27B train_rates structure, PRIOR_CALLS=2000, training-failure terciles, repository category with >=10 training issues; OpenHands training-half numerical calibration. No test threshold selection.',
            'numeric_transfer':'Separately replay serialized rounded 27B global fresh/retry rates and edges without refit. No language metadata alignment: global fallback only; message-cost replay is an unscaled unit-mismatch sensitivity, not calibrated transfer.',
            'interval_scope':'Percentile intervals over heldout issue resampling and random attempt selection/order; conditioning on the once-fitted training calibration, not including training-fit uncertainty.',
            'sampling':'Issue bootstrap of heldout set; draw 3 attempts uniformly without replacement in random order per sampled issue. Same issue draws, attempt orders, and fresh tie order for every policy and both costs.',
            'budget':'20/30/40/50% of total cost of all three sampled attempts; analytical ex-post normalization, never exposed to policy. Prefix ends at first over-budget attempt; no future-cost skipping. spent_cost means completed-run cost; interrupted remainder gets zero solves.',
            'empty_patch_definition':"patch_empty is model_patch is null OR model_patch.strip()=='' (nulls treated as no observed patch); model_EmptyPatch is separately retained as exit-status-derived metadata. Only already completed failed attempts expose patch_empty to the queue.",
            'short_nonempty':'Short/nonempty = train bottom failure-cost tercile AND no empty patch in observed failures; long/empty = upper tercile OR any empty patch in observed failures; middle group stays between them. Compare reversed group priority with same random fresh order; restarts only after all fresh issues. No future failure features.',
            'train_rates':{},'costs':{}}
    comparisons=[('plain_breadth','depth_first'),('adaptive','plain_breadth'),('adaptive','ranked_breadth'),('adaptive','deferred_rate'),('deferred_rate','ranked_breadth'),('short_nonempty_first','long_empty_first'),('numeric27b_adaptive','numeric27b_deferred'),('numeric27b_adaptive','plain_breadth'),('adaptive','numeric27b_adaptive')]
    for name,modes in records.items():
        f,r,e=calibrations[name]
        result['train_rates'][name]={'fresh':f,'retry':{repr(k):v for k,v in r.items()},'failure_terciles':e}
        arrays={m:np.asarray(v) for m,v in modes.items()}
        result['costs'][name]={}
        for j,frac in enumerate(FRACTIONS):
            row={'policies':{m:{metric:interval(v[:,j,k]) for k,metric in enumerate(('solved','spent_cost','early_restarts'))} for m,v in arrays.items()},
                 'paired_differences':{a+'_minus_'+b:interval(arrays[a][:,j,0]-arrays[b][:,j,0]) for a,b in comparisons}}
            result['costs'][name][str(frac)]=row
    packed = {name + '__' + mode: np.asarray(rows) for name, modes in records.items() for mode, rows in modes.items()}
    np.savez_compressed(npz_path, **packed)
    result['replicate_artifact'] = {'path': str(npz_path.relative_to(ROOT)) if npz_path.is_relative_to(ROOT) else str(npz_path), 'sha256': hashlib.sha256(npz_path.read_bytes()).hexdigest(), 'keys': sorted(packed), 'shape': '[replicate, budget index, metric index]', 'budgets': FRACTIONS.tolist(), 'metrics': ['solved', 'spent_cost', 'early_restarts']}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--reps',type=int,default=1000);parser.add_argument('--queue-reps',type=int,default=250);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--batch-seconds',type=int,default=480);parser.add_argument('--out',type=Path,default=ROOT/'evidence/results/openhands_queue_restart.json');args=parser.parse_args()
    start=time.monotonic(); path=ROOT/'evidence/results/openhands_attempt_costs.csv'
    issues,repos=load(path)
    result={'source_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seed':SEED,'replicates':args.reps,
            'attempts':sum(map(len,issues.values())),'issues':len(issues),
            'model_EmptyPatch_status_count':int(sum(v[:,4].sum() for v in issues.values())),
            'patch_empty_count':int(sum(v[:,3].sum() for v in issues.values())),
            'empty_status_signal_present':bool(any(v[:,4].any() for v in issues.values())),
            'cost_definition':'assistant_calls counts recorded assistant-role messages; excludes unrecorded API retries and is not an API bill. messages counts all recorded trajectory messages.',
            'method_provenance': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), ROOT/'analysis/queue_restart_decision.py', ROOT/'analysis/openhands_attempt_costs.py', ROOT/'evidence/results/queue_restart_decision.json')},
            'boolean_audit':'Strict accepts True/False/1/0; old resolved == 1 string check rejects True. No legacy estimates copied.',
            'statistics':statistics(issues,args.reps),'queue':replay(issues,repos,args.queue_reps,args.out.with_name(args.out.stem+'_replicates.npz'),args.checkpoint,args.batch_seconds)}
    result['elapsed_seconds']=time.monotonic()-start
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(args.out),'elapsed_seconds':result['elapsed_seconds'],'attempts':result['attempts'],'issues':result['issues']}))

if __name__=='__main__': main()
