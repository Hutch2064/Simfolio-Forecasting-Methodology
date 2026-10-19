"""Compile the large-matrix conditional-correlation likelihood."""
import hashlib
import importlib.util
import platform
import subprocess
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load():
    import fcntl

    import pybind11
    source=Path(__file__).with_name('likelihood_lapack.cpp')
    flags=['-O3','-std=c++17','-ffp-contract=off','-shared','-fPIC']
    if platform.system()=='Darwin':flags+=['-undefined','dynamic_lookup']
    compiler=subprocess.check_output(['clang++','--version'])
    identity=hashlib.sha256(source.read_bytes()+compiler+repr((flags,platform.platform(),
        sysconfig.get_config_var('SOABI'))).encode()).hexdigest()
    directory=Path(tempfile.gettempdir())/'simfolio-cdcc-lapack'/identity
    directory.mkdir(parents=True,exist_ok=True)
    binary=directory/('cdcc_lapack_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    with (directory/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not binary.exists():
            temporary=binary.with_suffix('.tmp')
            subprocess.run(['clang++',*flags,'-I'+pybind11.get_include(),
                '-I'+sysconfig.get_path('include'),str(source),'-o',str(temporary)],check=True)
            temporary.replace(binary)
    spec=importlib.util.spec_from_file_location('cdcc_lapack_native',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from scipy.linalg import cython_blas, cython_lapack
    module.configure(cython_lapack.__pyx_capi__["dpotrf"],cython_lapack.__pyx_capi__["dpotri"],cython_lapack.__pyx_capi__["dgesv"],cython_blas.__pyx_capi__["dgemv"],cython_blas.__pyx_capi__["ddot"])
    return module
