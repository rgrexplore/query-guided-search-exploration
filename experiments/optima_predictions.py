"""Freeze tuning-only cost predictions, then check them without refitting.

The model selects among the declared study settings. Recall eligibility and
routing time come from tuning measurements. Only the explicitly identified
random-sign paths have work counts predicted from distribution parameters.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from experiments.analysis import load_cases, write_csv
from experiments.optima_model import branch_depth_optimum, ideal_group_stats


METHOD_NAMES = {'scan': 'A', 'branch': 'B', 'keys': 'C'}
WORK_FIELDS = ('documents_scored', 'bitplane_words', 'leaf_words', 'nodes',
               'key_attempts', 'keys_generated')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def features(method, work, top_k):
    """Separate work terms; all coefficients are combined empirical prices."""
    count = work['documents_scored']
    values = dict(fixed=1.0, scored_rows=count,
                  log_scored_ratio=math.log(max(count / top_k, 1)))
    if method == 'branch':
        values.update(split_words=work['bitplane_words'], leaf_words=work['leaf_words'],
                      nodes=work['nodes'])
    elif method == 'keys':
        generated = work['keys_generated']
        values.update(key_lookups=work['key_attempts'],
                      key_queue_work=generated * math.log2(generated + 2))
    return values


def summarize_configuration(items):
    """One observation per configuration, regardless of its repeat count.

    Each feature is summarized from whole-query work. Stage and total times
    are recorded separately; a sum of medians is only a model prediction.
    """
    case = items[0]['case']
    rows = [row for item in items for row in item['queries']]
    per_query_features = [features(case['method'], row, case['top_k']) for row in rows]
    return {
        'work': {name: float(np.median([row[name] for row in rows])) for name in WORK_FIELDS},
        'features': {name: float(np.median([row[name] for row in per_query_features]))
                     for name in per_query_features[0]},
        'search_ms': float(np.median([row['search_ms'] for row in rows])),
        'routing_ms': float(np.median([row['routing_ms'] for row in rows])),
        'query_ms': float(np.median([row['query_ms'] for row in rows])),
        'composition_gap_ms': float(np.median([
            row['query_ms'] - row['routing_ms'] - row['search_ms'] for row in rows])),
    }


def fit_model(observations, threshold_percent=20.0):
    """Relative-error NNLS, with one median row per unique configuration.

    Redundant or all-zero columns are named explicitly and omitted from the
    numerical fit. Their individual prices cannot be learned from this data.
    """
    names = list(observations[0]['features'])
    matrix = np.array([[row['features'][name] for name in names] for row in observations])
    times = np.array([row['search_ms'] for row in observations])
    scales = np.maximum(np.linalg.norm(matrix / times[:, None], axis=0), 1e-12)
    normalized = matrix / scales
    kept = []
    for column in range(len(names)):
        if np.linalg.matrix_rank(normalized[:, kept + [column]]) > len(kept):
            kept.append(column)
    coefficients, _ = nnls(normalized[:, kept] / times[:, None], times / times)
    prices = {name: 0.0 for name in names}
    prices.update({names[column]: float(value / scales[column])
                   for column, value in zip(kept, coefficients, strict=True)})
    predicted = matrix @ np.array([prices[name] for name in names])
    errors = 100 * (predicted / times - 1)
    rank = int(np.linalg.matrix_rank(normalized))
    return {
        'coefficients': prices, 'training_configurations': len(observations),
        'matrix_rank': rank, 'requested_columns': len(names),
        'identifiable': rank == len(names),
        'omitted_columns': [name for column, name in enumerate(names) if column not in kept],
        'composition_gap_ms': float(np.median([row['composition_gap_ms'] for row in observations])),
        'median_absolute_fit_error_percent': float(np.median(np.abs(errors))),
        'max_absolute_fit_error_percent': float(np.max(np.abs(errors))),
        'fit_within_threshold': bool(np.all(np.abs(errors) <= threshold_percent)),
        'fit_residual_percent': errors.tolist(),
        'scope': 'One median observation per tuning configuration; combined API costs, not instruction prices.',
    }


def predict_search(model, method, work, top_k):
    return sum(model['coefficients'][name] * value
               for name, value in features(method, work, top_k).items())


def analytic_work(case, data):
    """Expected counts for a small, explicitly bounded subset of the family.

    Prefix cases require fixed strong positions and disjoint local keys.
    B includes repeated constant-prefix splits in the current implementation.
    Mean group sizes can cross a leaf boundary on actual queries; the later
    check reports that discrepancy instead of changing the prediction.
    """
    dimensions, strong = data['dimensions'], data['strong_bits']
    if (dimensions - strong) * data['weak_weight'] >= 1:
        return None
    global_search = case['router_kind'] == 'none'
    fixed_prefix = case['router_kind'] == 'direct' and case['support'] == 'fixed'
    if not global_search and not fixed_prefix:
        return None
    routing_bits = 0 if global_search else int(math.log2(case['clusters']))
    selected_rows = case['documents'] * case['probes'] / case['clusters']
    result = dict.fromkeys(WORK_FIELDS, 0.0)
    if case['method'] == 'scan':
        result['documents_scored'] = selected_rows
        return result
    # These formulas describe one preferred prefix, not the interleaving of
    # many cluster roots or branches whose fixed signs contradict the query.
    if case['probes'] != 1 or routing_bits > strong:
        return None
    if case['method'] == 'keys':
        end = case['key_offset'] + case['key_bits']
        if (case['support'] != 'fixed' or case['key_offset'] != routing_bits
                or end > strong or case['candidate_target'] or case['key_limit']):
            return None
        result.update(documents_scored=case['documents'] * 2.0**-end,
                      key_attempts=1.0, keys_generated=1.0)
        return result
    if case['node_budget'] or case.get('exploration', 0):
        return None
    words = math.ceil(selected_rows / 64)
    for depth in range(strong + 1):
        count = selected_rows * 2.0**-max(depth - routing_bits, 0)
        if count <= case['leaf_size']:
            result.update(documents_scored=count, nodes=depth + 1,
                          bitplane_words=depth * words, leaf_words=words)
            return result
    return None


def global_branch_derivative(model, data, top_k):
    """Derivative of the fitted one-path model while its leaf has at least K rows."""
    prices = model['coefficients']
    documents, depth = data['documents'], data['strong_bits']
    if not model['identifiable'] or prices['scored_rows'] <= 0:
        return {'available': False, 'reason': 'Individual work prices are not identifiable or row price is zero.'}
    if documents * 2.0**-depth < top_k:
        return {'available': False, 'reason': 'The one-leaf path cannot supply K expected rows at maximum depth.'}
    words = math.ceil(documents / 64)
    # For F=N*2^-j > K, d(log(F/K))/dj = -ln(2). One more
    # depth also adds a node. Incorporate both in the stationary equation.
    marginal_fixed = (words * prices['split_words'] + prices['nodes']
                      - math.log(2) * prices['log_scored_ratio'])
    if marginal_fixed <= 0:
        optimum = {'unconstrained_depth': None, 'candidate_depths': [0, depth]}
    else:
        optimum = branch_depth_optimum(documents, marginal_fixed / words,
                                       prices['scored_rows'], depth)
    return dict(available=True, **optimum,
                scope='Global balanced single path; fixed leaf width; F>=K. Check recall and integer choices.',
                equation='W*c_split+c_node-ln(2)*c_log = ln(2)*N*2^(-j)*c_row')


def fit_support(folder, threshold_percent):
    cases, failures = load_cases(folder / 'tuning')
    details = json.loads((folder / 'frozen' / 'all-settings.json').read_text())
    settings = read_csv(folder / 'frozen' / 'settings.csv')
    data = json.loads((folder / 'configuration.json').read_text())['data']
    assert all(set(item['case']['query_rows']) <= set(range(data['tuning_queries']))
               for item in cases), 'prediction fitting must use tuning query rows only'
    by_case = {item['case_id']: item for item in cases}
    observations, router_samples = {}, {}
    for row in settings:
        identifier = row['setting_id']
        detail = details[identifier]
        items = [by_case[source['case_id']] for source in detail['sources']]
        observations[identifier] = summarize_configuration(items)
        key = (detail['case']['router'], detail['case']['probes'])
        router_samples.setdefault(key, []).append(observations[identifier]['routing_ms'])
    models = {
        method: fit_model([observations[row['setting_id']] for row in settings
                           if row['method'] == method], threshold_percent)
        for method in METHOD_NAMES
    }
    for model in models.values():
        model['native_sha256'] = cases[0]['result']['native_sha256']
        model['worker_sha256'] = cases[0]['result']['worker_sha256']
    predictions = []
    for row in settings:
        identifier = row['setting_id']
        case = details[identifier]['case']
        observation = observations[identifier]
        calculated = analytic_work(case, data)
        work = observation['work'] if calculated is None else calculated
        routing = float(np.median(router_samples[(case['router'], case['probes'])]))
        model = models[case['method']]
        predicted = predict_search(model, case['method'], work, case['top_k']) + routing + model['composition_gap_ms']
        predictions.append(dict(
            support=folder.name, setting_id=identifier, method=case['method'],
            router=case['router_kind'], clusters=case['clusters'], probes=case['probes'],
            leaf_size=case.get('leaf_size', 0), key_bits=case.get('key_bits', 0),
            key_offset=case.get('key_offset', 0),
            predicted_p50_ms=predicted, measured_tuning_p50_ms=float(row['p50_ms']),
            recall=float(row['recall']), recall_low=None if not row['recall_low'] else float(row['recall_low']),
            lifetime_peak_bytes=int(row['lifetime_peak_bytes']),
            eligible_99=bool(row['recall_low'] and float(row['recall_low']) >= .99
                             and int(row['lifetime_peak_bytes']) <= case['ram_budget_bytes']),
            work_source='analytic_uniform_sign_expectation' if calculated is not None else 'empirical_tuning_medians',
            routing_source='empirical_tuning_layout_probe_median', predicted_routing_ms=routing,
            recall_source='empirical_tuning_bootstrap_lower_bound',
            **{'predicted_' + name: value for name, value in work.items()},
        ))
    choices = []
    measured = json.loads((folder / 'frozen' / 'shortlist.json').read_text())
    for method, letter in METHOD_NAMES.items():
        eligible = [row for row in predictions if row['method'] == method and row['eligible_99']]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda row: (row['predicted_p50_ms'], row['setting_id']))
        detail = details[chosen['setting_id']]
        measured_best = next((choice for choice in measured if choice['case']['method'] == method
                              and dict(selection='conservative', target=.99) in choice['uses']), None)
        choices.append(dict(
            setting_id='formula-' + letter, support=folder.name, source_setting_id=chosen['setting_id'],
            case=detail['case'], evaluation_seeds=detail['evaluation_seeds'],
            uses=[dict(selection='formula', target=.99)], prediction=chosen,
            model_status='within_tuning_tolerance' if models[method]['fit_within_threshold'] else 'exploratory_fit_exceeds_tolerance',
            measured_best_setting=None if measured_best is None else measured_best['setting_id'],
            same_as_measured_best=measured_best is not None and measured_best['setting_id'] == chosen['setting_id'],
        ))
    formulas = {'ideal_group': ideal_group_stats(data['documents'], data['strong_bits'],
                                               details[settings[0]['setting_id']]['case']['top_k']),
                'global_branch_derivative': global_branch_derivative(models['branch'], data,
                    details[settings[0]['setting_id']]['case']['top_k'])}
    return models, predictions, choices, formulas, failures


def fit(study, supports, threshold_percent=20.0):
    output = study / 'predictions'
    output.mkdir(exist_ok=False)
    models, forecasts, choices, formulas, hashes, failures = {}, [], [], {}, {}, []
    for support in supports:
        folder = study / support
        models[support], rows, selected, formulas[support], failed = fit_support(folder, threshold_percent)
        forecasts.extend(rows)
        choices.extend(selected)
        failures.extend(failed)
        for relative in ('configuration.json', 'tuning/schedule.json', 'frozen/settings.csv',
                         'frozen/all-settings.json', 'frozen/shortlist.json'):
            hashes[f'{support}/{relative}'] = digest(folder / relative)
    save_json(output / 'models.json', models)
    save_json(output / 'settings.json', forecasts)
    write_csv(output / 'settings.csv', forecasts)
    save_json(output / 'shortlist.json', choices)
    save_json(output / 'formulas.json', formulas)
    save_json(output / 'provenance.json', dict(
        supports=supports, p50_error_threshold_percent=threshold_percent, source_hashes=hashes,
        script_sha256=digest(Path(__file__)),
        math_helpers_sha256=digest(Path(__file__).with_name('optima_model.py')), failures=failures,
        scope='Frozen before evaluation. One median observation per qualified tuning configuration. '
              'Recall and routing are empirical. Expected work is analytic only where explicitly labeled. '
              'Fit residuals are training diagnostics, not evidence of future accuracy.'))
    print(f'Frozen {len(forecasts)} predictions and {len(choices)} formula-selected settings in {output}.')


def check(study):
    output = study / 'predictions'
    provenance = json.loads((output / 'provenance.json').read_text())
    choices = json.loads((output / 'shortlist.json').read_text())
    threshold = provenance['p50_error_threshold_percent']
    rows = []
    for support in provenance['supports']:
        evaluation = study / support / 'evaluation'
        cases, failures = load_cases(evaluation)
        summaries = {row['setting_id']: row for row in read_csv(evaluation / 'report' / 'settings.csv')}
        for choice in [entry for entry in choices if entry['support'] == support]:
            identifier = choice['setting_id']
            if identifier not in summaries:
                rows.append(dict(support=support, setting_id=identifier, status='missing_evaluation', failures=failures))
                continue
            observed = summaries[identifier]
            prediction = choice['prediction']
            actual = float(observed['p50_ms'])
            error = 100 * (prediction['predicted_p50_ms'] / actual - 1)
            items = [item for item in cases if item['case']['setting_id'] == identifier]
            actual_work = summarize_configuration(items)['work']
            best_id = choice['measured_best_setting']
            best = summaries.get(best_id)
            work_errors = {}
            for name in WORK_FIELDS:
                forecast = prediction['predicted_' + name]
                measured = actual_work[name]
                work_errors[name] = dict(predicted=forecast, measured=measured,
                                        error_percent=None if measured == 0 else 100 * (forecast / measured - 1))
            rows.append(dict(
                support=support, setting_id=identifier, method=choice['case']['method'], status='checked',
                predicted_p50_ms=prediction['predicted_p50_ms'], measured_p50_ms=actual,
                p50_error_percent=error, within_frozen_tolerance=abs(error) <= threshold,
                recall=float(observed['recall']), met_target=float(observed['recall']) >= .99,
                qualified_environment=observed['qualified_environment'] == 'True',
                work_source=prediction['work_source'], work=work_errors, model_status=choice['model_status'],
                same_as_measured_best=choice['same_as_measured_best'],
                measured_best_setting=best_id,
                formula_to_measured_best_ratio=None if best is None else actual / float(best['p50_ms']),
                within_five_percent_of_measured_best=None if best is None else abs(actual / float(best['p50_ms']) - 1) <= .05,
            ))
    save_json(output / 'check.json', dict(
        frozen_models_sha256=digest(output / 'models.json'), frozen_choices_sha256=digest(output / 'shortlist.json'),
        p50_error_threshold_percent=threshold, rows=rows,
        scope='Frozen predictions compared with final queries; no refit or replacement of choices. '
              'The 5% comparison is descriptive, not a significance test.'))
    print(f'Checked {len(rows)} frozen formula choices; see {output / "check.json"}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'check'))
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--support', choices=('fixed', 'adaptive'))
    args = parser.parse_args()
    if args.stage == 'fit':
        fit(args.study.resolve(), [args.support] if args.support else ['fixed', 'adaptive'])
    else:
        check(args.study.resolve())


if __name__ == '__main__':
    main()
