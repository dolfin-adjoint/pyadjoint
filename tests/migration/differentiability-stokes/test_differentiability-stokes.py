from os import path
import subprocess
import pytest

@pytest.mark.skip("does not test dolfin-adjoint")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = ["python", path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
