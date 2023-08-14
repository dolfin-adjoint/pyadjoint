import pytest
import sys
pytest.importorskip("fenics")
from os import path
import subprocess

@pytest.mark.xfail(reason="compute_gradient_tlm is not implemented")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = ["mpirun", "-n", "2", "python", path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
