import pytest
import importlib

@pytest.fixture(autouse=True)
def skip_by_missing_module(request):
    if request.node.get_marker('skipif_module_is_missing'):
        to_import = request.node.get_marker('skipif_module_is_missing').args[0]
        try:
            importlib.import_module(to_import)
        except ImportError:
            pytest.skip('skipped because module {} is missing'.format(to_import))   
