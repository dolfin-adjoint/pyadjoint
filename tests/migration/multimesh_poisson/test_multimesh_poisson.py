from os import path
import subprocess
import pytest


@pytest.mark.skip("Not supported by dolfin master yet")
@pytest.mark.xfail(reason="multimesh is not implemented")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = [sys.executable, path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
