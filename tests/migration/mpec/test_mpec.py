import pytest
pytest.importorskip("fenics")
from os import path
import subprocess
import pytest

@pytest.mark.xfail(reason="compute_tlm is not implemented. It is possible to get tlm values, but it involves interacting with tape and block outputs directly")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = ["python", path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
