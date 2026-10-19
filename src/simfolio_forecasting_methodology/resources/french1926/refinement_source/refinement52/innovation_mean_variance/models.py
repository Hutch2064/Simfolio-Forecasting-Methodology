"""Separate continuous-node center/spread corrections on unchanged RMS M261."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('node_split_frozen_joint', HERE.parent/'innovation_moments/models.py')
OLD = importlib.util.module_from_spec(spec); sys.modules[spec.name] = OLD; spec.loader.exec_module(OLD)
BASE, REF, ROOT, FROZEN = OLD.BASE, OLD.REF, OLD.ROOT, OLD.FROZEN
MODES = ('center_only', 'variance_only_preserve_mean')
DIAGNOSTICS = {}


def transform(nodes, mode):
    mean, variance = OLD.continuous_moments(nodes)
    if mode == 'center_only':
        return nodes-mean
    if mode == 'variance_only_preserve_mean':
        return (nodes-mean)/np.sqrt(variance)+mean
    if mode == 'joint_control':
        return OLD.normalize(nodes)
    raise ValueError(mode)


def forecast(task, mode):
    body = REF.MODELS.parent.body; original_nodes = body.learned.nodes

    def altered(data, n):
        nodes = original_nodes(data, n)
        mean, variance = OLD.continuous_moments(nodes)
        result = transform(nodes, mode); cm, cv = OLD.continuous_moments(result)
        DIAGNOSTICS[hashlib.sha256(data).hexdigest(), n, mode] = {
            'training_sha256': hashlib.sha256(data).hexdigest(), 'nodes': n, 'mode': mode,
            'original_nodes_sha256': hashlib.sha256(nodes.tobytes()).hexdigest(),
            'transformed_nodes_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
            'original_continuous_mean': mean, 'original_continuous_variance': variance,
            'transformed_continuous_mean': cm, 'transformed_continuous_variance': cv,
            'normalization_scope': 'pre-clipping uniform-probability moments; clipping unchanged'}
        return result

    if mode != 'original_control':
        body.learned.nodes = altered
    try:
        paths, stages, _ = BASE.forecast(task, 'deterministic_RMS')
        return paths, stages
    finally:
        body.learned.nodes = original_nodes


initialize = OLD.initialize
