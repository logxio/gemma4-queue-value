"""Fast, offline statistical reproduction from redacted attempt tables and draws.

Attempt probabilities are recomputed from observations. Queue confidence tables
are recomputed from the saved paired replay draws; constructing those draws is
the separate, longer openhands_queue_restart.py step. No network or figure writes.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'evidence/results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((RESULTS / name).read_text())


def boolean(value):
    if str(value).lower() in ('true', '1'): return True
    if str(value).lower() in ('false', '0'): return False
    raise ValueError(value)


def point(n, k):
    p = np.mean(k / n)
    q = np.mean((n-k)*k/(n*(n-1))) / np.mean((n-k)/n)
    return np.array([p, q, q/p, 1-q/p])


def model_summary(name, groups, reps=1000):
    counts = np.array([(len(groups[i]), sum(groups[i])) for i in sorted(groups)], dtype=float)
    paired = counts[counts[:, 0] >= 2]
    n, k = paired.T
    value = point(n, k)
    rng = np.random.default_rng(20260930 + 71)
    draws = np.array([point(*(paired[rng.integers(len(n), size=len(n))].T))
                      for _ in range(reps)])
    row = {'model': name, 'attempts': int(counts[:,0].sum()), 'issues': len(counts),
           'pairable_issues': len(paired), 'singletons_excluded': len(counts)-len(paired)}
    for j, key in enumerate(('p', 'q', 'q_over_p', 'icc')):
        row[key] = float(value[j])
        row[key+'_ci95'] = np.quantile(draws[:,j], [.025, .975]).tolist()
    return row


def cross_model_table():
    groups = defaultdict(lambda: defaultdict(list))
    from schedule_frontier import load
    ids, success, *_ = load()
    for i, y in zip(ids, success): groups['Qwen3.8-27B'][i] = y.tolist()
    with (RESULTS/'nebius_attempt_costs.csv').open() as handle:
        for r in csv.DictReader(handle):
            groups[r['model_name']][r['instance_id']].append(boolean(r['target']))
    with (RESULTS/'openhands_attempt_costs.csv').open() as handle:
        for r in csv.DictReader(handle):
            groups['Qwen3-Coder-480B-A35B'][r['instance_id']].append(boolean(r['resolved']))
    path = ROOT/'evidence/gemma_a100/full76_official.jsonl'
    for line in path.read_text().splitlines():
        r = json.loads(line)
        groups['Gemma-4-31B'][r['task']].append(bool(r['resolved']))
    return [model_summary(name, g) for name, g in groups.items()]


def queue_tables():
    result = read_json('openhands_queue_restart.json')
    assert result['source_csv_sha256'] == digest(RESULTS/'openhands_attempt_costs.csv')
    for name, expected in result['method_provenance'].items():
        assert digest(ROOT/name) == expected, f'Method provenance changed: {name}'
    artifact = result['queue']['replicate_artifact']
    path = ROOT/artifact['path']
    assert digest(path) == artifact['sha256']
    draws = np.load(path, allow_pickle=False)
    out = []
    for cost, budgets in result['queue']['costs'].items():
        for j, (budget, row) in enumerate(budgets.items()):
            for comparison, summary in row['paired_differences'].items():
                a,b = comparison.split('_minus_')
                d = draws[cost+'__'+a][:,j,0] - draws[cost+'__'+b][:,j,0]
                ci = np.quantile(d, [.025,.975])
                assert np.isclose(d.mean(), summary['mean'])
                assert np.allclose(ci, summary['ci95'])
                out.append({'cost':cost, 'budget':float(budget), 'comparison':comparison,
                            'mean':float(d.mean()), 'ci95':ci.tolist(), 'replicates':len(d)})
            for policy, metrics in row['policies'].items():
                for k, metric in enumerate(artifact['metrics']):
                    v = draws[cost+'__'+policy][:,j,k]
                    assert np.isclose(v.mean(), metrics[metric]['mean'])
                    assert np.allclose(np.quantile(v,[.025,.975]),metrics[metric]['ci95'])
    return out


def gemma_tables():
    result = read_json('gemma4_sandbox_and_schema_audit.json')
    prefix = RESULTS/'gemma4_sandbox_and_schema_audit'
    attempts = pd.read_csv(str(prefix)+'_attempts.csv')
    refs = pd.read_csv(str(prefix)+'_reference.csv')
    events = pd.read_csv(str(prefix)+'_events.csv')
    refs['repo'] = refs.task.str.rsplit('_',n=1).str[0]
    reference = []
    for repo, rows in refs.groupby('repo'):
        passed = int(rows.reference_patch_resolved.sum())
        expected = result['reference_audit'][repo]
        assert (len(rows),passed) == (expected['total'],expected['reference_pass'])
        reference.append({'repo':repo,'total':len(rows),'reference_pass':passed,
                          'reference_fail':len(rows)-passed})
    assert len(attempts) == 228 and int(attempts.resolved.sum()) == 68
    assert int(attempts.schema_errors.sum()) == result['ghost_costs']['schema_rejected_calls']
    assert int(events.schema_error.sum()) == int(attempts.schema_errors.sum())
    paired_rows = pd.read_csv(str(prefix)+'_paired_attempts.csv')
    eligible = paired_rows[paired_rows.paired_eligible_attempt]
    paired = {'tasks':int(paired_rows.task.nunique()),
              'exact_failure_flagged_attempts':int((~eligible.resolved & eligible.exact_loop_flag).sum()),
              'exact_success_flagged_attempts':int((eligible.resolved & eligible.exact_loop_flag).sum())}
    pairing = result['same_task_success_normal_failure_pairing']
    assert paired['tasks'] == pairing['tasks']
    for k in ('exact_failure_flagged_attempts','exact_success_flagged_attempts'):
        assert paired[k] == pairing['recomputed'][k]
    error_table = []
    for field in ('edit_errors','schema_errors','other_edit_errors'):
        exposed = attempts[field] > 0
        for resolved in (False,True):
            n = int((attempts.resolved == resolved).sum())
            k = int((exposed & (attempts.resolved == resolved)).sum())
            expected = result['observed']['error_modalities'][field]
            assert k == expected['exposed_successes' if resolved else 'exposed_failures']
            error_table.append({'error':field,'resolved':resolved,'exposed':k,'attempts':n,'fraction':k/n})
    # Recompute conditional restarts from the compact clustered attempt table.
    conditional = {}
    for field, stored in result['observed']['error_modalities'].items():
        pairs = wins = chronological_pairs = chronological_wins = 0
        for _, rows in attempts.groupby('task'):
            rows = rows.sort_values('round')
            y = rows.resolved.to_numpy(bool)
            flag = rows[field].to_numpy() > 0
            for j in range(len(rows)):
                if y[j] or not flag[j]: continue
                pairs += len(rows)-1
                wins += int(y.sum())
                if j+1 < len(rows):
                    chronological_pairs += 1
                    chronological_wins += int(y[j+1])
        assert (pairs,wins) == (stored['restart_pairs'],stored['restart_successes'])
        assert (chronological_pairs,chronological_wins) == (stored['chronological_next_pairs'],stored['chronological_next_successes'])
        conditional[field] = {'restart_pairs':pairs,'restart_successes':wins,
                              'restart_rate':wins/pairs if pairs else None,
                              'chronological_next_pairs':chronological_pairs,
                              'chronological_next_rate':chronological_wins/chronological_pairs if chronological_pairs else None}
    draws = pd.read_csv(str(prefix)+'_queue_draws.csv')
    observed = result['queue']['replay_600_and_breakers']
    sim = []
    for policy, rows in draws.groupby('policy'):
        if policy not in observed: continue
        for metric, stored in observed[policy].items():
            value = rows[metric].mean()
            assert np.isclose(value, stored['mean'])
            assert np.allclose(np.quantile(rows[metric],[.025,.975]),stored['task_bootstrap_95ci'])
        if policy == 'cap_600': continue
        paired = rows.set_index('bootstrap_draw').solved - draws[draws.policy=='cap_600'].set_index('bootstrap_draw').solved
        stored = result['queue']['paired_breaker_minus_600_solved'][policy]
        assert np.isclose(paired.mean(),stored['mean'])
        assert np.allclose(np.quantile(paired,[.025,.975]),stored['task_bootstrap_95ci'])
        sim.append({'policy':policy,'solved_delta':float(paired.mean()),
                    'ci95':np.quantile(paired,[.025,.975]).tolist()})
    return {'reference_audit':reference,'errors':error_table,'queue_replay':sim,
            'schema_rejected_calls':int(attempts.schema_errors.sum()),
            'observed_interval_seconds':float(events.loc[events.schema_error,'preceding_event_interval_seconds'].sum()),
            'denominator_distortion':result['denominator_distortion'],
            'paired_error_tests':pairing,
            'conditional_restarts':conditional,
            'cases':result['cases'],'queue_assumptions':result['queue']['measured_support']}


def reproduce():
    started = time.monotonic()
    models = cross_model_table()
    openhands = read_json('openhands_queue_restart.json')
    extracted = read_json('openhands_attempt_costs_summary.json')
    assert extracted['csv_sha256'] == digest(RESULTS/'openhands_attempt_costs.csv')
    row = next(r for r in models if r['model']=='Qwen3-Coder-480B-A35B')
    assert row['attempts'] == extracted['attempts'] == openhands['attempts']
    assert row['issues'] == extracted['issues'] == openhands['issues']
    for metric in ('p','q','q_over_p','icc'):
        stored = openhands['statistics']['equal_issue'][metric]
        assert np.isclose(row[metric],stored['point'])
        assert np.allclose(row[metric+'_ci95'],stored['ci95'])
    result = {'model_configurations':len(models), 'attempts_total':sum(r['attempts'] for r in models),
              'model_table':models, 'openhands_queue':queue_tables(), 'gemma':gemma_tables()}
    assert result['model_configurations'] == 6
    result['elapsed_seconds'] = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_bytes'] = rss if sys.platform=='darwin' else rss*1024
    assert result['peak_rss_bytes'] < 4*1024**3
    return result


if __name__ == '__main__':
    result = reproduce()
    (RESULTS/'confirmation_reproduction.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('model_configurations','attempts_total','elapsed_seconds','peak_rss_bytes')}))
