import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *


@pytest.fixture(params=[
    "constant assign",
    "function assign",
    "project",
])
def tag(request):
    return request.param


def test_tags(tag):
    if tag == "constant assign":
        c1 = Constant(0.0)
        c2 = Constant(1.0)
        c1.assign(c2, ad_block_tag=tag)
    else:
        mesh = UnitSquareMesh(1, 1)
        V = FunctionSpace(mesh, "CG", 1)
        f1 = Function(V)
        f2 = Function(V)
        f2.vector()[:] = 1.0
        if tag == "function assign":
            f1.assign(f2, ad_block_tag=tag)
        elif tag == "project":
            f1 = project(f2, V, ad_block_tag=tag)
        else:
            raise ValueError
    tape = get_working_tape()
    tags = tape.get_tags()
    assert len(tags) == 1 and tag in tags
    blocks = tape.get_blocks(tag=tag)
    assert len(blocks) == 1
