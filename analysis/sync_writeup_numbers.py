"""Inject completed OpenHands replay evidence into the candidate, then run halo.

Contract: analysis/nebius_queue_restart.py, SOURCES['coder480b'].
No missing-result fallback to another model. Unknown contracts fail closed.
Default target is the candidate; the main manuscript is never selected implicitly.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- OPENHANDS_CONFIRMATION_START -->'
END = '<!-- OPENHANDS_CONFIRMATION_END -->'
HALO = ROOT / 'analysis/fast_blind_review.py'
SOURCE = 'nebius/SWE-rebench-openhands-trajectories, Qwen3-Coder-480B-A35B-Instruct, OpenHands 0.54.0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(x):
    require(type(x) in (int, float) and math.isfinite(x), 'Expected a finite numeric measurement')
    return x


def positive_integer(x):
    require(type(x) is int and x > 0, 'Expected a positive integer population')
    return x


def contrast(x):
    require(set(x) == {'mean', 'paired_95'}, 'Unknown contrast schema')
    mean = number(x['mean'])
    interval = x['paired_95']
    require(isinstance(interval, list) and len(interval) == 2, 'Expected paired 95% interval')
    lo, hi = map(number, interval)
    require(lo <= hi, 'Reversed interval')
    return mean, lo, hi


def render(data):
    expected = {'source', 'cost', 'split', 'attempts_per_issue', 'train_issues', 'heldout_issues',
                'category_min_train_issues', 'categories', 'replicates', 'train_fresh_per_1000_cost',
                'train_retry_global_per_1000_cost', 'budgets', 'curve', 'decomposition', 'train_curve', 'primary'}
    require(set(data) == expected, 'Unknown OpenHands schema: top-level fields differ from coder480b contract')
    require(data['source'] == SOURCE, 'Wrong source or model')
    require(data['cost'] == 'assistant turns (main-model calls)', 'Wrong cost unit')
    require(data['split'] == "SHA-256(instance_id + ':openhands-queue') < 128 for training", 'Wrong split')
    require(data['attempts_per_issue'] == 3 and data['category_min_train_issues'] == 10, 'Wrong protocol')
    train, held, reps = (positive_integer(data[k]) for k in ('train_issues', 'heldout_issues', 'replicates'))
    positive_integer(data['categories'])
    require(set(data['budgets']) == {'0.3', '0.4', '0.5'}, 'Unexpected budgets')
    for budget in data['budgets'].values():
        require(set(budget) == {'plain_breadth', 'ranked_breadth', 'deferred_rate', 'adaptive'}, 'Unknown policy set')
        for policy in budget.values():
            for key in ('solved_mean', 'spent_cost_mean', 'early_restarts_mean'):
                require(number(policy[key]) >= 0, 'Negative policy measurement')
        contrast(budget['adaptive']['minus_deferred_rate'])
        contrast(budget['adaptive']['minus_ranked_breadth'])
    primary = data['primary']
    require(set(primary) == {'fraction', 'chosen_by', 'heldout_crossing'}, 'Unknown primary endpoint')
    require(primary['chosen_by'] == "argmax of the training half's crossing gain", 'Endpoint was not training-selected')
    fractions = data['train_curve']['fractions']
    require(fractions == data['curve']['fractions'] and len(fractions) > 0, 'Inconsistent budget grids')
    gains = data['train_curve']['adaptive_minus_deferred_rate']['mean']
    require(len(gains) == len(fractions), 'Incomplete training curve')
    k = max(range(len(gains)), key=lambda i: number(gains[i]))
    require(primary['fraction'] == fractions[k], 'Primary budget does not match training argmax')
    m, lo, hi = contrast(primary['heldout_crossing'])
    curve = data['curve']['adaptive_minus_deferred_rate']
    require(all(len(curve[x]) == len(fractions) for x in ('mean', 'lo', 'hi')), 'Incomplete held-out curve')
    require([m, lo, hi] == [curve[x][k] for x in ('mean', 'lo', 'hi')], 'Primary estimate does not match held-out curve')
    total = contrast(data['budgets']['0.5']['adaptive']['minus_ranked_breadth'])
    return (f'An independent OpenHands collection evaluates Qwen3-Coder-480B-A35B with recorded assistant turns as the call unit. '
            f'The analysis fits on {train:,} issues and replays {held:,} held-out issues over {reps:,} paired resamples. '
            f'The training-selected crossing budget is {100 * number(primary["fraction"]):g}% of observed three-attempt calls. '
            f'At that budget, held-out crossing adds {m:+.1f} repairs, with a 95% interval of {lo:.1f} to {hi:.1f}. '
            f'At 50% budget, adaptive scheduling changes repairs by {total[0]:+.1f} against ranked breadth, with an interval of {total[1]:.1f} to {total[2]:.1f}. '
            'These are paired replay estimates on the independently collected trajectories.')


def synchronize(text, result):
    require(text.count(START) == 1 and text.count(END) == 1, 'Expected one confirmation anchor pair')
    start, end = text.index(START) + len(START), text.index(END)
    require(start < end, 'Reversed anchors')
    if result.exists():
        paragraph = render(json.loads(result.read_text()))
        status = 'injected'
    else:
        paragraph = 'Independent OpenHands queue confirmation: NaN（未重算）. The completed result artifact is pending; no confirmation estimate is included here.'
        status = 'pending'
    return text[:start] + '\n' + paragraph + '\n' + text[end:], status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, default=ROOT / 'paper/writeup_v6_candidate.md')
    parser.add_argument('--result', type=Path, default=ROOT / 'evidence/results/openhands_queue_restart.json')
    parser.add_argument('--check', action='store_true', help='Validate and halo-check without writing')
    args = parser.parse_args()
    try:
        updated, status = synchronize(args.manuscript.read_text(), args.result)
        with tempfile.TemporaryDirectory(prefix='writeup-sync-') as tmp:
            draft = Path(tmp) / 'candidate.md'
            draft.write_text(updated)
            run = subprocess.run([sys.executable, str(HALO), 'halo-check', '--venue', 'GEMMA',
                                  '--manuscript', str(draft)], capture_output=True, text=True)
            require(run.returncode == 0, f'Halo execution failed: {run.stderr}')
            halo = json.loads(run.stdout)
            require(halo.get('passed') is True, f'Halo rejected injection: {run.stdout}')
            require(2750 <= halo['word_count'] <= 2950, 'Candidate outside requested 2750–2950 word range')
        if not args.check:
            args.manuscript.write_text(updated)
        print(json.dumps({'status': status, 'written': not args.check, 'halo': halo}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'Refusing number synchronization: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
