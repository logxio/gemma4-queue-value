"""Recompute publication tables and compare every numeric cell with the source."""
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'evidence/results'


def load(name):
    return json.loads((RESULTS / name).read_text())


def source_tables():
    text = (ROOT / 'paper/writeup.md').read_text()
    tables = []
    for part in text.split('**Table ')[1:]:
        lines = [line for line in part.split('\n\n', 2)[1].splitlines() if line.startswith('|')]
        tables.append([[cell.strip() for cell in line.strip('|').split('|')] for line in lines])
    assert len(tables) == 4
    return tables


def f(value, places=3, signed=False):
    rounded = Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return format(rounded, ('+' if signed else '') + f'.{places}f').replace('-', '\N{MINUS SIGN}')


def interval(values, places=3):
    return f(values[0], places) + '-' + f(values[1], places)


def contrast(row):
    return f(row['mean'], 1, row['mean'] != 0) + ' [' + ', '.join(f(v, 1) for v in row['ci95']) + ']'


def numbers(cell):
    return re.findall(r'[+\-\N{MINUS SIGN}]?\d+(?:,\d{3})*(?:\.\d+)?',
                      cell.replace('\N{EN DASH}', ' ').replace('**', '').replace('\\ge', ''))


def numeric_key(cell):
    cell = re.sub(r'(?<=\d)-(?=\d)', ' ', cell)
    return [v.replace(',', '').replace('\N{MINUS SIGN}', '-').lstrip('+') for v in numbers(cell)]


def reproduce_tables():
    started = time.monotonic()
    source = source_tables()
    tables = [[list(row) for row in table] for table in source]
    provenance = {}
    required = ['openhands_queue_restart.json', 'gemma4_sandbox_and_schema_audit.json',
                'decision_value.json', 'nebius_attempt_costs.csv', 'restart_value.json',
                'nebius_restart.json', 'queue_restart_decision.json']
    for name in required:
        provenance[name] = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
    oh = load('openhands_queue_restart.json')
    gemma = load('gemma4_sandbox_and_schema_audit.json')
    decision = load('decision_value.json')
    qwen = load('restart_value.json')['primary']
    models = [
        [qwen['attempts'], qwen['tasks'], qwen['p'], qwen['q'], qwen['q_over_p'], qwen['bootstrap_95']['q_over_p']],
        [oh['attempts'], oh['statistics']['eligible_issues'], *[oh['statistics']['equal_issue'][k]['point'] for k in ['p', 'q', 'q_over_p']], oh['statistics']['equal_issue']['q_over_p']['ci95']],
        [gemma['observed']['attempts'], gemma['observed']['tasks'], *[gemma['observed']['rates'][k] for k in ['p', 'q', 'q_over_p']], gemma['observed']['rate_bootstrap_95ci']['q_over_p']],
    ]
    groups = defaultdict(lambda: defaultdict(list))
    with (RESULTS / 'nebius_attempt_costs.csv').open() as handle:
        for row in csv.DictReader(handle):
            value = row['target'].lower()
            assert value in ('true', 'false', '1', '0')
            groups[row['model_name']][row['instance_id']].append(value in ('true', '1'))
    for model in ['swe-agent-llama-8b', 'swe-agent-llama-70b', 'swe-agent-llama-405b']:
        rows = groups[model]
        counts = np.array([(len(v), sum(v)) for v in rows.values() if len(v) >= 3], dtype=float)
        n, k = counts.T
        p = np.mean(k / n)
        q = np.mean((n-k)*k/(n*(n-1))) / np.mean((n-k)/n)
        rng = np.random.default_rng(20260930 + 71)
        draws = []
        for _ in range(1000):
            nb, kb = counts[rng.integers(len(n), size=len(n))].T
            pb = np.mean(kb / nb)
            qb = np.mean((nb-kb)*kb/(nb*(nb-1))) / np.mean((nb-kb)/nb)
            draws.append(qb/pb)
        models.append([sum(len(v) for v in rows.values()), len(counts), p, q, q/p, np.quantile(draws, [.025, .975]).tolist()])
    for row, values in zip(tables[0][2:], models):
        a, n, p, q, ratio, ci = values
        row[1:] = [f'{a:,}', f'{n:,}', f(p), f(q), f(ratio), interval(ci)]

    qqueue = load('queue_restart_decision.json')
    missing = []
    for j, budget in enumerate(['0.3', '0.4', '0.5']):
        rows = qqueue['budgets'][budget]
        target = tables[1][j+2]
        target[1:5] = [f(rows[p]['solved_mean'], 1) for p in ['plain_breadth', 'ranked_breadth', 'deferred_rate', 'adaptive']]
        target[5] = contrast({'mean': rows['deferred_rate']['minus_ranked_breadth']['mean'], 'ci95': rows['deferred_rate']['minus_ranked_breadth']['paired_95']})
        target[6:] = [contrast({'mean': rows['adaptive'][key]['mean'], 'ci95': rows['adaptive'][key]['paired_95']}) for key in ['minus_deferred_rate', 'minus_plain']]
    for j, budget in enumerate(['0.2', '0.3', '0.5']):
        rows = oh['queue']['costs']['assistant_calls'][budget]
        target = tables[1][j+5]
        target[1:5] = [f(rows['policies'][p]['solved']['mean'], 1) for p in ['plain_breadth', 'ranked_breadth', 'deferred_rate', 'adaptive']]
        target[5:] = [contrast(rows['paired_differences'][key]) for key in ['deferred_rate_minus_ranked_breadth', 'adaptive_minus_deferred_rate', 'adaptive_minus_plain_breadth']]

    refs = gemma['reference_audit']
    total = sum(v['total'] for v in refs.values())
    passed = sum(v['reference_pass'] for v in refs.values())
    missing_modules = sum(v['failed_with_missing_module'] for v in refs.values())
    tables[2][2][3:] = [f'{passed} / {total} tasks', f(gemma['observed']['rates']['q']), interval(gemma['observed']['rate_bootstrap_95ci']['q'])]
    tables[2][3][3] = f'{missing_modules} / {total} tasks'
    tables[2][4][3] = f'{total-passed-missing_modules} / {total} tasks'
    for j, field in enumerate(['edit_errors', 'schema_errors', 'other_edit_errors', 'timeout', 'context_window', 'early_loop']):
        values = gemma['observed']['error_modalities'][field]
        events = gemma['ghost_costs']['schema_rejected_calls'] if field == 'schema_errors' else values['exposed_successes'] + values['exposed_failures']
        tables[2][5+j][1:] = [str(values['exposed_successes']), str(values['exposed_failures']),
                             f'{events:,} ' + ('calls' if field == 'schema_errors' else 'attempts'),
                             f(values['restart_rate']), interval(values['task_bootstrap_95ci']['restart_rate'])]
    paired = gemma['same_task_success_normal_failure_pairing']
    tables[2][-1][1:] = [str(paired['recomputed']['exact_success_flagged_attempts']), str(paired['recomputed']['exact_failure_flagged_attempts']), f"{paired['tasks']} paired tasks", 'n/a', 'Sign test $p = ' + f(paired['exact_task_matched_sign_test']['two_sided_sign_p'], 5) + '$']

    auc_sum = value_sum = auc_mean = stops = lost = saved = 0
    for j, system in enumerate(['sweagent', 'agentless', 'openhands', 'mini']):
        held = decision['C3_breakeven_screen']['selected'][system]['heldout']
        auc_gain = decision['C2_auc_vs_value']['per_system'][system]['selected_by_cv_auc'][2]
        value_gain = held['replay_gain']
        tables[3][2+j][1:] = [f(auc_gain, 2, True), f(value_gain, 2, True), f(held['auc']),
                             f"{held['stops']} / {held['sacrificed']:.1f}", f"{held['saved_calls']:,.0f}",
                             f(held['sacrificed_per_saved_call'], 4) + ' < ' + f(held['lambda'], 4)]
        auc_sum += auc_gain
        value_sum += value_gain
        auc_mean += held['auc']/4
        stops += held['stops']; lost += held['sacrificed']; saved += held['saved_calls']
    tables[3][-1][1:] = [f(auc_sum, 2, True), f(value_sum, 2, True), f(auc_mean) + ' mean',
                         f'{stops} / {lost:.1f}', f'{saved:,.0f}', f(value_sum-auc_sum, 1, True) + ' over AUC']

    comparisons = []
    for t, (expected, actual) in enumerate(zip(source, tables), 1):
        for r, (erow, arow) in enumerate(zip(expected[2:], actual[2:]), 1):
            for c, (e, a) in enumerate(zip(erow[1:], arow[1:]), 2):
                comparisons.append({'table': t, 'row': r, 'column': c, 'manuscript': e,
                                    'recomputed': a, 'match': numeric_key(e) == numeric_key(a) and 'NaN' not in a})
    mismatches = [v for v in comparisons if not v['match']]
    return {'tables': tables, 'comparisons': comparisons, 'mismatches': mismatches,
            'missing_measurements': missing, 'all_numeric_cells_match': not mismatches,
            'numeric_cells_checked': len(comparisons), 'source_sha256': provenance,
            'elapsed_seconds': time.monotonic()-started,
            'scope': 'Frozen aggregate evidence and compact attempt data; bootstrap Llama equal-issue rows from CSV; no model calls or new scheduling draws'}


if __name__ == '__main__':
    report = reproduce_tables()
    (RESULTS / 'publication_table_verification_s2.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ['all_numeric_cells_match', 'numeric_cells_checked', 'elapsed_seconds']}))
    print(json.dumps(report['mismatches'], ensure_ascii=False, indent=2))
