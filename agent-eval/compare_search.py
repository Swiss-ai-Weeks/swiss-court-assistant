"""Compare frozen-parameter replays with a live-search control on identical queries and gold coverage.

Combining development and validation is descriptive, NOT a second independent validation result.
Use --subset to compare a validation replay against just its matching control queries.
"""
import argparse
import hashlib
import json
from pathlib import Path

from opencaselaw import aggregate


def compare(control, reports, subset=False):
    if control['summary']['errors'] or control['summary']['completed'] != control['summary']['total']:
        raise ValueError('Control must be complete and error-free')
    old = {r['id']: r for r in control['per_query'] if r['status'] == 'ok'}
    new = {}
    for report in reports:
        if report['parameters'] != reports[0]['parameters']:
            raise ValueError('Do not combine replays with different parameters')
        for key in ('golden_sha256', 'decisions'):
            if report['config'][key] != control['config'][key]:
                raise ValueError(f'Control/replay {key} differs')
        for row in report['per_query']:
            if row['id'] in new:
                raise ValueError('Duplicate query across replay files')
            if row['id'] not in old or row['available_gold'] != old[row['id']]['available_gold']:
                raise ValueError('Queries and available gold labels must match the control')
            new[row['id']] = row
    if not new or (not subset and new.keys() != old.keys()):
        raise ValueError('Replay must cover all scored control queries (or explicitly use --subset)')
    pairs = [(old[i], new[i]) for i in sorted(new)]
    before, after = aggregate([a for a, b in pairs]), aggregate([b for a, b in pairs])
    return {'mode': 'frozen-parameter replay of live candidate caches; not answer citations',
            'n': len(pairs), 'control_population': control['summary']['total'], 'subset': subset,
            'parameters': reports[0]['parameters'], 'control': before, 'candidate': after,
            'delta': {k: after[k] - before[k] for k in before},
            'paired_hit1': {'wins': [b['id'] for a, b in pairs if b['hit@1'] > a['hit@1']],
                            'losses': [b['id'] for a, b in pairs if b['hit@1'] < a['hit@1']]},
            'by_language': {lang: {'n': len(group), 'control': aggregate([a for a, b in group]),
                                  'candidate': aggregate([b for a, b in group])}
                            for lang in ('de', 'fr', 'it')
                            if (group := [(a, b) for a, b in pairs if b['language'] == lang])},
            'per_query': [{'id': b['id'], 'query': a['query'], 'old': a['ranked_ids'], 'new': b['ranked_ids']}
                          for a, b in pairs]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--subset', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(json.loads(args.control.read_text()), [json.loads(p.read_text()) for p in args.reports], args.subset)
    result['inputs_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.control, *args.reports]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_query', 'inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
