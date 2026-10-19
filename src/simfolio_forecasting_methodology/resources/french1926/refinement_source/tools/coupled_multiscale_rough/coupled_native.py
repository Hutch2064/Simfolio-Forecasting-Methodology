"""Stream the pinned NumPy Gaussian distribution into the unchanged OU recursion."""
import hashlib
import importlib.util
import platform
import subprocess
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_paths():
    import fcntl

    import numpy as np
    import pybind11
    from numba.extending import get_cython_function_address
    source = Path(__file__).with_name('paths.cpp')
    numpy_root = Path(np.__file__).parent
    libraries = [numpy_root/'random/lib/libnpyrandom.a', numpy_root/'_core/lib/libnpymath.a']
    flags = ['-O3','-std=c++17','-ffp-contract=off','-shared','-fPIC']
    if platform.system() == 'Darwin': flags += ['-undefined','dynamic_lookup']
    compiler = subprocess.check_output(['clang++','--version'])
    identity = hashlib.sha256(source.read_bytes()+compiler+repr((flags,np.__version__,
        platform.platform(),sysconfig.get_config_var('SOABI'))).encode()+
        b''.join(p.read_bytes() for p in libraries)).hexdigest()
    directory = Path(tempfile.gettempdir())/'simfolio-coupled-multiscale'/identity
    directory.mkdir(parents=True,exist_ok=True)
    binary = directory/('coupled_multiscale_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    with (directory/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not binary.exists():
            temporary = binary.with_suffix('.tmp')
            subprocess.run(['clang++',*flags,'-I'+pybind11.get_include(),
                '-I'+sysconfig.get_path('include'),'-I'+np.get_include(),str(source),
                *map(str,libraries),'-o',str(temporary)],check=True)
            temporary.replace(binary)
    spec = importlib.util.spec_from_file_location('coupled_multiscale_native',binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,get_cython_function_address('scipy.linalg.cython_blas','ddot')
