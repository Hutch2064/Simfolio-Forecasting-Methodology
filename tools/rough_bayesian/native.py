"""Strict floating-point ARM filter; other platforms retain the Numba kernel."""
import hashlib
import importlib.util
import platform
import subprocess
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_filter():
    if platform.machine() != 'arm64':
        return None
    import fcntl

    import pybind11
    from numba.extending import get_cython_function_address
    source = Path(__file__).with_name('overlay_filter.cpp')
    flags = ['-O3', '-mcpu=native', '-std=c++17', '-ffp-contract=off',
             '-shared', '-fPIC', '-undefined', 'dynamic_lookup']
    compiler = subprocess.check_output(['clang++', '--version'])
    identity = hashlib.sha256(source.read_bytes() + compiler + repr(
        (flags, platform.platform(), sysconfig.get_config_var('SOABI'))).encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / 'simfolio-rough-native' / identity
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / ('rough_overlay_native' + sysconfig.get_config_var('EXT_SUFFIX'))
    with (directory / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            temporary = binary.with_suffix('.tmp')
            subprocess.run(['clang++', *flags, '-I' + pybind11.get_include(),
                            '-I' + sysconfig.get_path('include'), str(source), '-o', str(temporary)], check=True)
            temporary.replace(binary)
    spec = importlib.util.spec_from_file_location('rough_overlay_native', binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    address = get_cython_function_address('scipy.linalg.cython_blas', 'ddot')
    gemv = get_cython_function_address('scipy.linalg.cython_blas', 'dgemv')
    return module, address, gemv
