import pytest
import importlib
import numpy.random
from pyadjoint import set_working_tape, Tape


@pytest.fixture(autouse=True)
def skip_by_missing_module(request):
    marker = request.node.get_closest_marker("skipif_module_is_missing")
    if marker:
        to_import = marker.args[0]
        try:
            importlib.import_module(to_import)
        except ImportError:
            pytest.skip('skipped because module {} is missing'.format(to_import))


def pytest_runtest_setup(item):
    """ Hook function which is called before every test """
    set_working_tape(Tape())

    # Fix the seed to avoid random test failures due to slight tolerance variations
    numpy.random.seed(21)
