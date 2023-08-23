import subprocess
from os import path
import pytest
import sys

fenics = pytest.importorskip("fenics")


@pytest.mark.skipif(not hasattr(fenics, "HDF5File"),
                    reason="requires hdf5 support")
@pytest.mark.xfail(reason="PointIntegralSolver is not implemented")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = [sys.executable, path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
