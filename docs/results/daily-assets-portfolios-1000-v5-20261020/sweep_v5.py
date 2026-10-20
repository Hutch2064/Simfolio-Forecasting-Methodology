"""Ragged daily-panel sweep: immutable native forecasts, shared bounded CPU slots."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import signal
import statistics
import time
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from forecast import forecast as forecast_1000
from simfolio_forecasting_methodology.french_panel import runner as original
from simfolio_forecasting_methodology.french_panel import selected
from simfolio_forecasting_methodology.french_panel.sweep import (
    _identity,
    exclusive_lock,
    model_directory,
    read_control,
)

PROTOCOL = 'daily-cross-asset-ragged-native-crps-1000-v5'
PUBLIC = 'model_testing_joint_stationary_bootstrap_v1'
CONTEXT_CALENDAR = 'original synthetic business-day accounting labels;1,000-path streamed forecast v5'
TARGET_CALENDAR = 'actual per-series dates[position+1:position+1+max_horizon]; native observation index'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError('input path escapes frozen panel root')
    return path


def schedule(root, row):
    return json.loads(input_path(root, row['origins_file']).read_text())


def load_series(root, row, *, verify_files=True):
    if verify_files:
        for name, expected in row['files'].items():
            if sha256(input_path(root, name)) != expected:
                raise ValueError(f'input checksum differs: {name}')
    dates = np.load(input_path(root, row['dates_file']), mmap_mode='r', allow_pickle=False)
    values = np.load(input_path(root, row['returns_file']), mmap_mode='r', allow_pickle=False)
    origins = schedule(root, row)
    n = row['observations']
    if (dates.dtype != np.dtype('datetime64[D]') or values.dtype != np.dtype('<f8')
            or dates.shape != (n,) or values.shape != (n,) or n < 2
            or np.any(np.isnat(dates)) or np.any(dates[1:] <= dates[:-1])
            or not np.isfinite(values).all()
            or str(dates[0]) != row['first_date'] or str(dates[-1]) != row['last_date']):
        raise ValueError('invalid per-series dates/returns or metadata alignment')
    if not isinstance(origins, list) or len(origins) != row['origins_count'] or not origins:
        raise ValueError('invalid per-series origin count')
    labels = set()
    required = {'origin_label', 'evaluation_split', 'position', 'origin_date', 'max_horizon', 'train_fraction'}
    for origin in origins:
        if not isinstance(origin, dict) or set(origin) != required:
            raise ValueError('invalid origin schema')
        position, horizon = origin['position'], origin['max_horizon']
        if (type(position) is not int or type(horizon) is not int or position < 1 or horizon < 1
                or position + 1 + horizon > n or str(dates[position]) != origin['origin_date']
                or not isinstance(origin['origin_label'], str) or not origin['origin_label']
                or origin['origin_label'] in labels):
            raise ValueError('origin does not align with actual series dates')
        labels.add(origin['origin_label'])
    if max(o['max_horizon'] for o in origins) != row['max_horizon']:
        raise ValueError('series maximum horizon differs')
    return dates, values, origins


def load_panel(root, *, verify_files=True):
    panel = json.loads((Path(root) / 'panel.json').read_text())
    identity = {k: v for k, v in panel.items() if k != 'panel_fingerprint'}
    # Input builder uses canonical stdlib JSON with ensure_ascii=True.
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if panel.get('panel_fingerprint') != fingerprint or panel.get('simulations_per_origin') != 1000:
        raise ValueError('panel fingerprint/path-count differs')
    rows = panel['series']
    if not isinstance(rows, list) or not rows or len({r['series_id'] for r in rows}) != len(rows):
        raise ValueError('empty panel or duplicate series IDs')
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ('series_id', 'asset_class', 'family')):
            raise ValueError('missing series metadata')
        if not isinstance(row.get('files'), dict) or not row['files']:
            raise ValueError('series checksums must be explicit')
        if any(row[k] not in row['files'] for k in ('dates_file', 'returns_file', 'origins_file')):
            raise ValueError('required input file is outside declared checksum closure')
        if verify_files:
            load_series(root, row)
    if panel['expected_tasks'] != sum(row['origins_count'] for row in rows):
        raise ValueError('panel expected task count differs')
    return panel


def admitted_contract(mid, spec, registry_path):
    """Validate a declaration before importing its arbitrary factory module."""
    registry = json.loads(Path(registry_path).read_text())
    if registry.get('schema') != 'french-external-model-registry-v1':
        raise ValueError('invalid external registry schema')
    record = registry['models'][mid]
    identity = record['identity']
    if (identity.get('model_id') != mid or identity.get('factory') != spec['factory']
            or record['factory'] != spec['factory'] or record['implementation_digest'] != spec['implementation_digest']
            or original.stable_digest(identity) != spec['implementation_digest']):
        raise ValueError('external factory/canonical implementation identity differs')
    files, contract = identity.get('source_files'), identity.get('numerical_contract')
    if not isinstance(files, dict) or not files or not isinstance(contract, dict) or not contract:
        raise ValueError('external closure and runtime contract must be explicit')
    for relative, expected in files.items():
        if sha256(input_path(registry['source_root'], relative)) != expected:
            raise ValueError(f'external source differs: {relative}')
    return contract


def interleaved_series(panel):
    groups = defaultdict(lambda: defaultdict(deque))
    for i, row in enumerate(panel['series']):
        groups[row['asset_class']][row['family']].append(i)
    families = {key: deque(sorted(value)) for key, value in groups.items()}
    classes = deque(sorted(groups))
    while classes:
        asset_class = classes.popleft()
        family = families[asset_class].popleft()
        yield groups[asset_class][family].popleft()
        if groups[asset_class][family]:
            families[asset_class].append(family)
        if families[asset_class]:
            classes.append(asset_class)


def task_identity(fingerprint, model_id, series_id, origin):
    return {'execution_fingerprint': fingerprint, 'model_id': model_id, 'series_id': series_id,
            'origin': dict(origin), 'training_observations': origin['position'] + 1,
            'seed': original.coherent_daily_seed(origin['origin_date'], origin['max_horizon'], 1000)}


def task_paths(root, task):
    stem = f'{task[0]:06d}/{task[1]:03d}'
    return Path(root) / 'tasks' / f'{stem}.npz', Path(root) / 'errors' / f'{stem}.json'


def prepare_manifest(model, spec, panel, origins, root, contract=None):
    runtime = original.dependency_identity(model)
    if contract is not None and any(k not in runtime or runtime[k] != v for k, v in contract.items()):
        raise ValueError('saved runtime differs from approved external numerical contract')
    identity = {'protocol_id': PROTOCOL, 'panel_fingerprint': panel['panel_fingerprint'],
        'input_manifest': panel, 'series_ids': [r['series_id'] for r in panel['series']],
        'series_origins': [{'series_id': r['series_id'], 'origins': list(o)}
                           for r, o in zip(panel['series'], origins, strict=True)],
        'model_id': model.model_id, 'factory': spec['factory'],
        'implementation_digest': spec['implementation_digest'], 'runtime': runtime,
        'simulations_per_origin': 1000, 'source_closure': original._source_closure(),
        'sweep_sha256': sha256(__file__), 'seed_contract': 'coherent_daily_seed(origin_date,max_horizon,1000)',
        'forecast_context_calendar': CONTEXT_CALENDAR, 'training_target_calendar': TARGET_CALENDAR,
        'raw_aggregation': 'canonical origin mean per native series/horizon; equal native cells; diagnostic only',
        'canonical_scoring': 'separate versioned reporter; never inferred from raw scheduler diagnostics'}
    return original._manifest(Path(root), identity)


def initialize_model(mid, spec, panel_root, model_root, registry_path):
    contract = admitted_contract(mid, spec, registry_path)
    module, name = spec['factory'].split(':')
    model = getattr(importlib.import_module(module), name)()
    if getattr(model, 'model_id', None) != mid:
        raise ValueError('factory returned a different model ID')
    panel = load_panel(panel_root, verify_files=False)
    origins = tuple(tuple(schedule(panel_root, row)) for row in panel['series'])
    fingerprint = prepare_manifest(model, spec, panel, origins, model_root, contract)
    return model, panel, origins, fingerprint


def execute_task(model, panel, series, root, fingerprint, task, *, first_origin=False):
    """Versioned1,000-path streamed native laws; unchanged train/targets and score."""
    i, j = task
    dates, values, origins = series
    origin = origins[j]
    identity = task_identity(fingerprint, model.model_id, panel['series'][i]['series_id'], origin)
    checkpoint, error = task_paths(root, task)
    if checkpoint.exists():
        original._read_checkpoint(checkpoint, identity)
        return 'done'
    try:
        losses, diagnostics = forecast_1000(model, identity['series_id'], values, dates,
                                               SimpleNamespace(**origin), identity['seed'])
        diagnostics = {**diagnostics, 'worker_first_origin_warmup': first_origin,
                       'actual_target_first_date': str(dates[origin['position'] + 1]),
                       'actual_target_last_date': str(dates[origin['position'] + origin['max_horizon']]),
                       'calendar_scope': 'native observed training/targets; original synthetic weekday forecast context'}
        original._write_checkpoint(checkpoint, identity, losses, diagnostics)
    except Exception as exc:  # noqa: BLE001 -- preserve a failed origin without dropping the panel
        original._atomic_json(error, {'identity': identity, 'exception_type': type(exc).__name__, 'message': str(exc)})
        return 'failed'
    error.unlink(missing_ok=True)
    return 'done'


def continuing(path, mid, spec):
    try:
        control = read_control(path)
        current = control['models'].get(mid)
        return bool(not control['stop'] and current and current['enabled'] and _identity(current) == _identity(spec))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def worker(connection, mid, spec, panel_root, root, control_path, registry_path, workers_lock):
    try:
        with ExitStack() as stack:
            for path in (workers_lock, Path(root) / '.run.lock'):
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open('a+'))
                fcntl.flock(handle, fcntl.LOCK_SH)
            if not continuing(control_path, mid, spec):
                connection.send(('paused', None))
                return
            model, panel, _, fingerprint = initialize_model(mid, spec, panel_root, root, registry_path)
            connection.send(('ready', fingerprint))
            cached_i, series, first = None, None, True
            while True:
                task = connection.recv()
                if task is None:
                    return
                if not continuing(control_path, mid, spec):
                    connection.send(('paused', task))
                    return
                if task[0] != cached_i:
                    series = load_series(panel_root, panel['series'][task[0]])
                    cached_i = task[0]
                result = execute_task(model, panel, series, root, fingerprint, tuple(task), first_origin=first)
                first = False
                connection.send((result, task))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:  # noqa: BLE001 -- isolate a failed worker/model
        try:
            connection.send(('fatal', f'{type(exc).__name__}: {exc}'))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


def start_worker(*args):
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, *args))
    process.start()
    child.close()
    return process, parent


def workload(origin):
    n, h = origin['position'] + 1, origin['max_horizon']
    return (int(math.log2(n)), int(math.log2(h))), n + 1000 * h


class ModelProgress:
    def __init__(self, mid, spec, panel, origins, root, allowed):
        self.mid, self.spec, self.panel, self.origins, self.root = mid, spec, panel, origins, root
        self.allowed_order = tuple(allowed)
        self.allowed = set(allowed)
        self.expected = panel['expected_tasks']
        self.fingerprint = self.error = None
        self.queue = deque()
        self.completed, self.failures = set(), set()
        self.series_sums = {}
        self.resumed = self.attempted = self.warmup_count = self.sample_count = 0
        self.compute_seconds = 0.0
        self.samples = defaultdict(lambda: deque(maxlen=128))
        self.remaining = Counter()
        self.published_summary = None

    def identity(self, task):
        return task_identity(self.fingerprint, self.mid, self.panel['series'][task[0]]['series_id'], self.origins[task[0]][task[1]])

    def scan(self, fingerprint):
        self.fingerprint = fingerprint
        for i, origins in enumerate(self.origins):
            for j, origin in enumerate(origins):
                task = i, j
                checkpoint, error = task_paths(self.root, task)
                if checkpoint.exists():
                    self.accept(task, aggregate=False, initial=True)
                    self.resumed += 1
                elif error.exists():
                    if json.loads(error.read_text()).get('identity') != self.identity(task):
                        raise ValueError('saved failed-origin identity differs')
                    self.failures.add(task)
                elif task in self.allowed:
                    self.remaining[workload(origin)[0]] += 1
            self.aggregate_series(i)
        self.queue.extend(task for task in self.allowed_order if task not in self.completed and task not in self.failures)

    def accept(self, task, *, aggregate=True, initial=False):
        if task in self.completed:
            return
        _, diagnostics = original._read_checkpoint(task_paths(self.root, task)[0], self.identity(task))
        self.completed.add(task)
        self.failures.discard(task)
        elapsed = diagnostics['forecast_and_score_seconds']
        self.compute_seconds += elapsed
        origin = self.origins[task[0]][task[1]]
        key, proxy = workload(origin)
        if diagnostics.get('worker_first_origin_warmup', False):
            self.warmup_count += 1
        else:
            self.samples[key].append((elapsed, proxy))
            self.sample_count += 1
        if not initial and task in self.allowed:
            self.remaining[key] -= 1
        if aggregate:
            self.aggregate_series(task[0])

    def fail(self, task):
        if task not in self.failures:
            self.failures.add(task)
            self.remaining[workload(self.origins[task[0]][task[1]])[0]] -= 1

    def aggregate_series(self, i):
        origins = self.origins[i]
        if i in self.series_sums or not all((i, j) in self.completed for j in range(len(origins))):
            return
        capacity = max(o['max_horizon'] for o in origins)
        sums, counts = np.zeros(capacity), np.zeros(capacity, dtype=np.int64)
        for j, origin in enumerate(origins):
            losses, _ = original._read_checkpoint(task_paths(self.root, (i, j))[0], self.identity((i, j)))
            sums[:origin['max_horizon']] += losses
            counts[:origin['max_horizon']] += 1
        self.series_sums[i] = float(np.sum(sums / counts))

    def projection(self):
        result = {'sample_count': self.sample_count, 'retained_sample_count': sum(map(len, self.samples.values())),
                  'warmup_origins_excluded': self.warmup_count, 'observed_workload_strata': len(self.samples),
                  'fallback_projected_tasks': 0, 'mean_worker_seconds': None, 'median_worker_seconds': None}
        remaining = sum(self.remaining.values())
        if not remaining:
            result.update(mean_worker_seconds=0.0, median_worker_seconds=0.0)
            return result
        if not self.sample_count:
            return result
        totals = {'mean': 0.0, 'median': 0.0}
        for key, count in self.remaining.items():
            if count <= 0:
                continue
            nearest = key if key in self.samples else min(self.samples, key=lambda k: sum(abs(a-b) for a, b in zip(k, key, strict=True)))
            samples = self.samples[nearest]
            if nearest != key:
                result['fallback_projected_tasks'] += count
            target_proxy = 2.0**key[0] + 1000 * 2.0**key[1]
            for label, reduce in (('mean', statistics.mean), ('median', statistics.median)):
                costs = [s[0] for s in samples] if nearest == key else [s[0] * target_proxy / s[1] for s in samples]
                totals[label] += count * reduce(costs)
        result.update({f'{label}_worker_seconds': value for label, value in totals.items()})
        return result

    def summary(self):
        indices = sorted(self.series_sums)
        cells = sum(self.panel['series'][i]['max_horizon'] for i in indices)
        total = 0.0
        for i in indices:
            total += self.series_sums[i]
        raw = total / cells if cells else None
        complete = len(self.completed) == self.expected and not self.failures and not self.error
        return {'model_id': self.mid, 'protocol_id': PROTOCOL, 'panel_fingerprint': self.panel['panel_fingerprint'],
            'execution_fingerprint': self.fingerprint, 'status': 'complete' if complete else 'incomplete',
            'expected_tasks': self.expected, 'completed_tasks': len(self.completed), 'failed_tasks': len(self.failures),
            'pending_tasks': self.expected - len(self.completed) - len(self.failures),
            'target_tasks': len(self.allowed), 'completed_target_tasks': len(self.allowed & self.completed),
            'failed_target_tasks': len(self.allowed & self.failures), 'resumed_tasks': self.resumed,
            'attempted_tasks_this_call': self.attempted, 'completed_forecast_and_score_seconds': self.compute_seconds,
            'complete_series': [self.panel['series'][i]['series_id'] for i in indices], 'complete_series_cells': cells,
            'final_raw_crps': raw if complete else None, 'partial_complete_series_raw_crps': raw if not complete and not self.error else None,
            'raw_score_scope': 'diagnostic equal native cells; canonical v3 scores are in separate reporter output',
            'eta_projection': self.projection()}


def choose_model(candidates, counts, progress, admissions):
    waiting = [mid for mid in candidates if not counts.get(mid)]
    if waiting:
        return min(waiting, key=lambda mid: admissions.get(mid, -1))
    def priority(mid):
        item = progress[mid]
        estimate = item.projection()['mean_worker_seconds']
        return (estimate if estimate is not None else len(item.queue)) / counts[mid]
    return max(candidates, key=priority)


class ParallelController:
    def __init__(self, panel_root, output_root, control_path, *, registry_path=None, starter=start_worker,
                 selected_tasks=None, max_tasks=None, progress_callback=None):
        self.panel_root, self.output_root, self.control_path = Path(panel_root), Path(output_root), Path(control_path)
        self.registry_path = Path(registry_path or self.control_path.with_name('external_model_registry.json'))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.panel = load_panel(self.panel_root)
        self.origins = tuple(tuple(schedule(self.panel_root, r)) for r in self.panel['series'])
        all_tasks = [(i, j) for i in interleaved_series(self.panel) for j in range(len(self.origins[i]))]
        if selected_tasks is not None:
            lookup = {(r['series_id'], o['origin_label']): (i, j) for i, (r, origins) in enumerate(zip(self.panel['series'], self.origins, strict=True)) for j, o in enumerate(origins)}
            if len({(r['series_id'], r['origin_label']) for r in selected_tasks}) != len(selected_tasks):
                raise ValueError('duplicate selected smoke task')
            chosen = {lookup[(r['series_id'], r['origin_label'])] for r in selected_tasks}
            all_tasks = [task for task in all_tasks if task in chosen]
        if max_tasks is not None:
            if type(max_tasks) is not int or max_tasks < 0:
                raise ValueError('max tasks must be a nonnegative integer')
            all_tasks = all_tasks[:max_tasks]
        self.allowed = all_tasks
        self.bounded = selected_tasks is not None or max_tasks is not None
        self.progress, self.active, self.admissions = {}, {}, {}
        self.sequence = 0
        self.starter, self.progress_callback = starter, progress_callback
        self.workers_lock = self.output_root / '.parallel-workers.lock'
        self.last_control = {'models': {}, 'workers': 12, 'stop': True}
        self.control_error = None
        self.latest_status = None
        self.last_publish = 0.0
        self.draining = False

    def enabled(self, mid, control):
        spec = control['models'].get(mid)
        return bool(not control['stop'] and spec and spec['enabled'] and not self.progress[mid].error
                    and _identity(spec) == _identity(self.progress[mid].spec))

    def counts(self):
        return Counter(slot['mid'] for slot in self.active.values())

    def return_task(self, item, task, *, crash=False):
        if task is None:
            return
        path, error = task_paths(item.root, task)
        if path.exists():
            item.accept(task)
        elif crash:
            original._atomic_json(error, {'identity': item.identity(task), 'exception_type': 'WorkerProcessExit',
                                         'message': 'worker exited before reporting origin'})
            item.fail(task)
        else:
            item.queue.appendleft(task)

    def poll(self):
        self.control_error = None
        try:
            control = read_control(self.control_path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.control_error = f'{type(exc).__name__}: {exc}'
            control = {**self.last_control, 'stop': True}
        changed = control != self.last_control
        self.last_control = control
        if self.draining:
            control = {**control, 'stop': True}
        for mid, spec in control['models'].items():
            if mid not in self.progress:
                item = self.progress[mid] = ModelProgress(mid, spec, self.panel, self.origins,
                                                          model_directory(self.output_root, mid), self.allowed)
                try:
                    admitted_contract(mid, spec, self.registry_path)
                except Exception as exc:  # noqa: BLE001 -- reject unapproved modules before import
                    item.error = f'{type(exc).__name__}: {exc}'
                changed = True
            elif _identity(spec) != _identity(self.progress[mid].spec):
                self.progress[mid].error = 'factory or digest changed; use a new model ID'
        for key, slot in list(self.active.items()):
            item, connection, process = self.progress[slot['mid']], slot['connection'], slot['process']
            try:
                while not slot.get('connection_eof', False) and connection.poll():
                    kind, payload = connection.recv()
                    changed = True
                    if kind == 'ready':
                        if item.fingerprint is None:
                            item.scan(payload)
                        elif payload != item.fingerprint:
                            raise ValueError('worker execution identity mismatch')
                        slot['ready'] = True
                    elif kind == 'done':
                        item.accept(tuple(payload))
                        slot.update(task=None, ready=True)
                    elif kind == 'failed':
                        item.fail(tuple(payload))
                        slot.update(task=None, ready=True)
                    elif kind == 'paused':
                        self.return_task(item, slot['task'])
                        slot.update(task=None, ready=False, stopping=True)
                    elif kind == 'fatal':
                        item.error = payload
                        self.return_task(item, slot['task'], crash=True)
                        slot.update(task=None, ready=False, stopping=True)
                    else:
                        raise ValueError('unknown worker event')
            except EOFError:
                slot['connection_eof'] = True
            except Exception as exc:  # noqa: BLE001 -- stop corrupt/inconsistent model only
                item.error = f'{type(exc).__name__}: {exc}'
                try:
                    connection.send(None)
                except (OSError, EOFError):
                    pass
                slot.update(ready=False, stopping=True)
            if not process.is_alive():
                process.join()
                if slot['task'] is None and not slot['stopping']:
                    item.error = f'worker exited before ready (exit code {process.exitcode})'
                try:
                    self.return_task(item, slot['task'], crash=True)
                except Exception as exc:  # noqa: BLE001 -- checkpoint corruption must remain visible
                    item.error = f'{type(exc).__name__}: {exc}'
                connection.close()
                del self.active[key]
                changed = True
        counts = self.counts()
        waiting = [mid for mid, item in self.progress.items() if self.enabled(mid, control)
                   and not counts[mid] and (item.fingerprint is None or item.queue)]
        # A dynamic model gets the next boundary, rather than waiting for an
        # entire existing panel queue to exhaust. Retire only the slots needed.
        retire = max(0, len(waiting) - (control['workers'] - len(self.active) + sum(s['stopping'] for s in self.active.values())))
        allowed_keys = list(self.active)[:control['workers']]
        for key, slot in self.active.items():
            if not slot['ready']:
                continue
            item = self.progress[slot['mid']]
            lend = retire > 0 and item.mid not in waiting
            if key not in allowed_keys or not self.enabled(item.mid, control) or not item.queue or lend:
                slot['connection'].send(None)
                slot.update(ready=False, stopping=True)
                retire -= int(lend)
            else:
                task = item.queue.popleft()
                item.attempted += 1
                slot['connection'].send(task)
                slot.update(task=task, ready=False)
            changed = True
        while len(self.active) < control['workers'] and not control['stop']:
            counts = self.counts()
            candidates = [mid for mid, item in self.progress.items() if self.enabled(mid, control)
                and (item.fingerprint is None or len(item.queue) > sum(s['mid'] == mid and s['task'] is None and not s['stopping'] for s in self.active.values()))]
            if not candidates:
                break
            mid = choose_model(candidates, counts, self.progress, self.admissions)
            item = self.progress[mid]
            if not counts[mid]:
                try:
                    with exclusive_lock(item.root / '.run.lock'):
                        pass
                except RuntimeError as exc:
                    item.error = str(exc)
                    continue
            process, connection = self.starter(mid, item.spec, self.panel_root, item.root,
                                               self.control_path, self.registry_path, self.workers_lock)
            self.sequence += 1
            self.admissions[mid] = self.sequence
            self.active[self.sequence] = {'mid': mid, 'process': process, 'connection': connection,
                                          'ready': False, 'stopping': False, 'task': None}
            changed = True
        if changed or self.latest_status is None or time.monotonic() - self.last_publish >= 5:
            return self.publish(control)
        return self.latest_status

    def publish(self, control):
        counts, models = self.counts(), {}
        catalogue = self.control_path.with_name('model_catalogue.json')
        names = {}
        if catalogue.exists():
            try:
                names = {r['model_id']: r['display_name'] for r in json.loads(catalogue.read_text())['comparators']}
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Display names cannot affect identities or numerical work.
        for mid, item in self.progress.items():
            summary = item.summary()
            enabled = self.enabled(mid, control)
            models[mid] = {**summary, 'display_name': names.get(mid, mid),
                'enabled': bool(control['models'].get(mid, {}).get('enabled')), 'active_workers': counts[mid],
                'state': 'failed' if item.error else 'completed' if summary['status'] == 'complete' else 'active' if counts[mid] else 'waiting' if enabled and item.queue else 'paused' if not enabled else 'failed' if item.failures else 'bounded_complete',
                'error': item.error, 'ready_queue': len(item.queue), 'output_directory': str(item.root),
                'in_flight_origins': [list(s['task']) for s in self.active.values() if s['mid'] == mid and s['task'] is not None]}
            if item.fingerprint is not None and summary != item.published_summary:
                original._atomic_json(item.root / 'summary.json', summary)
                original._atomic_json(item.root / 'progress.json', summary)
                item.published_summary = summary
        for row in models.values():
            row['running_origins'] = len(row['in_flight_origins'])
        enabled_models = [m for m in models.values() if m['enabled']]
        eta = {'kind': 'measured_workload_projection_not_calibrated_wall_eta',
            'scope': 'enabled queued/inflight work; excludes failed origins, setup, IO, contention and elapsed in-flight fraction',
            'assumptions': 'Per-model log2(training length), log2(native horizon) strata; nearest observed stratum scaled by n+1000H if absent. Up to128 samples/stratum. First origin of each worker excluded as warmup. Origin/asset fit costs are nonstationary.',
            'sample_count': sum(m['eta_projection']['sample_count'] for m in enabled_models),
            'mean_seconds_lower_bound': None, 'median_seconds_lower_bound': None,
            'uncertainty_interval_seconds': None,
            'blocked_model_ids': [mid for mid, row in models.items() if row['enabled'] and (row['error'] or row['failed_tasks'])]}
        for label in ('mean', 'median'):
            values = [m['eta_projection'][label + '_worker_seconds'] for m in enabled_models]
            if all(value is not None for value in values):
                eta[label + '_seconds_lower_bound'] = sum(values) / control['workers']
        status = {'scheduler': PROTOCOL, 'updated_unix': time.time(), 'models': models,
            'workers': control['workers'], 'active_workers': len(self.active),
            'free_slots': max(0, control['workers'] - len(self.active)),
            'running_origins': sum(m['running_origins'] for m in models.values()),
            'worker_assignments': [{'slot': key, 'pid': getattr(slot['process'], 'pid', None),
                'model_id': slot['mid'], 'task': list(slot['task']) if slot['task'] is not None else None,
                'ready': slot['ready'], 'stopping': slot['stopping']} for key, slot in self.active.items()],
            'enabled_model_count': len(enabled_models), 'expected_active_tasks': sum(m['expected_tasks'] for m in enabled_models),
            'completed_active_tasks': sum(m['completed_tasks'] for m in enabled_models),
            'failed_active_tasks': sum(m['failed_tasks'] for m in enabled_models),
            'ready_queue': sum(m['ready_queue'] for m in enabled_models),
            'stop_requested': control['stop'] and self.control_error is None,
            'control_error': self.control_error, 'bounded_task_selection': self.bounded,
            'selected_tasks_per_model': len(self.allowed), 'eta': eta,
            'forecast_context_calendar': CONTEXT_CALENDAR, 'training_target_calendar': TARGET_CALENDAR,
            'canonical_scores': 'separate versioned scoring reporter'}
        original._atomic_json(self.output_root / 'status.json', status)
        self.latest_status, self.last_publish = status, time.monotonic()
        if self.progress_callback is not None:
            try:
                self.progress_callback(status)
            except Exception as exc:  # noqa: BLE001 -- observer failures must not interrupt numerical work
                status['callback_error'] = f'{type(exc).__name__}: {exc}'
                original._atomic_json(self.output_root / 'status.json', status)
        return status


def wait_for_events(controller, timeout):
    connections = [slot['connection'] for slot in controller.active.values()
                   if not slot['stopping'] and not slot.get('connection_eof', False)]
    if connections:
        return wait_connections(connections, timeout=timeout)
    time.sleep(timeout)
    return []


def run_controller(panel_root, output_root, control_path, *, registry_path=None, poll_seconds=.1,
                   selected_tasks=None, max_tasks=None, progress_callback=None):
    output_root = Path(output_root)
    with exclusive_lock(output_root / '.controller.lock'):
        with exclusive_lock(output_root / '.parallel-workers.lock'):
            pass
        controller = ParallelController(panel_root, output_root, control_path, registry_path=registry_path,
            selected_tasks=selected_tasks, max_tasks=max_tasks, progress_callback=progress_callback)
        original._atomic_json(output_root / 'scheduler-provenance.json', {
            'scheduler': PROTOCOL, 'script_sha256': sha256(__file__), 'full_panel_checkpoint_imports': False,
            'bounded_task_selection': controller.bounded, 'selected_tasks_per_model': len(controller.allowed)})
        previous = {}
        def drain(signum, frame):
            controller.draining = True
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, drain)
        try:
            while True:
                status = controller.poll()
                if not controller.active and (controller.draining or controller.bounded and all(
                    item.error or len(item.allowed & (item.completed | item.failures)) == len(item.allowed)
                    for mid, item in controller.progress.items() if status['models'][mid]['enabled'])):
                    return status
                wait_for_events(controller, poll_seconds)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--tasks-json', type=Path, help='Exact list of {series_id,origin_label}; original horizons unchanged')
    parser.add_argument('--max-tasks', type=int, help='Bound declared task count per model, in class/family interleaved order')
    args = parser.parse_args()
    required_threads = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')
    if any(os.environ.get(key) != '1' for key in required_threads):
        parser.error('set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1 before launch')
    tasks = json.loads(args.tasks_json.read_text()) if args.tasks_json else None
    result = run_controller(args.panel_root, args.output_root, args.control, registry_path=args.registry,
                            selected_tasks=tasks, max_tasks=args.max_tasks)
    if result['control_error'] or any(m['error'] or m['failed_tasks'] for m in result['models'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
