"""Adapters for the existing canonical runner; no scoring changes."""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
# Spawn inherits the parent runner's modified search path. Select this suite
# explicitly before importing the common research module name in each worker.
sys.path.insert(0,str(HERE))
import models

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('learned_multiscale_offset_canonical_runner',ROOT/'tools/rough_jump_vine/run_panel.py')
runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
original_hashes=runner.source_hashes
original_evaluate=runner.evaluate
runner.models=models
runner.CANDIDATES=models.CANDIDATES


def source_hashes():
    hashes=original_hashes()
    for directory in (ROOT/'tools/rough_bayesian',ROOT/'tools/dynamic_rough_inference',ROOT/'tools/empirical_rough_noise',ROOT/'tools/conditional_residual_rough',ROOT/'tools/conditional_empirical_rough',ROOT/'tools/debiased_whittle_rough',ROOT/'tools/differenced_whittle_rough',ROOT/'tools/streamed_rough_paths',ROOT/'tools/relaxed_hurst_differenced_rough',ROOT/'tools/consistent_noise_rough',ROOT/'tools/raw_return_laplace_rough',ROOT/'tools/student_return_laplace_rough',ROOT/'tools/student_implied_noise_rough',ROOT/'tools/pathwise_multiscale_rough',ROOT/'tools/untruncated_rough_priors',ROOT/'tools/learned_multiscale_rates_rough',ROOT/'tools/predictive_loading_rates_rough',ROOT/'tools/learned_loading_full_hurst_rough',ROOT/'tools/multiscale_offset_rough',Path(__file__).resolve().parent):
        for pattern in ('*.py','*.cpp'):
            for path in sorted(directory.glob(pattern)):
                hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    backend=models.overlay.native_filter()
    if backend is not None:
        hashes['rough_filter_native_binary']=hashlib.sha256(Path(backend[0].__file__).read_bytes()).hexdigest()
    native=models.load_paths()[0]
    hashes['rough_streamed_native_binary']=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    return hashes


def evaluate(index,task):
    models.FIT_DIAGNOSTICS.clear()
    try:
        result=original_evaluate(index,task)
    except Exception as error:
        runner.atomic(Path(os.environ['ROUGH_INFERENCE_OUTPUT'])/f'failure-{index:04d}.json',
            {'task': index,'portfolio': task.portfolio_id,'origin': str(task.origin_date),
                'error_type': type(error).__name__,'message': str(error)})
        raise
    runner.atomic(Path(os.environ['ROUGH_INFERENCE_OUTPUT'])/f'task-{index:04d}-diagnostics.json',
        list(models.FIT_DIAGNOSTICS.values()))
    return result


runner.source_hashes=source_hashes
runner.evaluate=evaluate


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',required=True)
    known,_=parser.parse_known_args()
    os.environ['ROUGH_INFERENCE_OUTPUT']=str(Path(known.output).resolve())
    runner.main()
