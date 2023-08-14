import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(params=[
    "constant assign",
    "function assign",
    "interpolate",
    "project",
    "supermesh project",
    "interpolate method",
    "project method",
    "supermesh project method",
])
def tag(request):
    return request.param


def test_tags(tag):
    if tag == "constant assign":
        c1 = Constant(0.0)
        c2 = Constant(1.0)
        c1.assign(c2, ad_block_tag=tag)
    else:
        mesh = UnitSquareMesh(1, 1, diagonal='left')
        V = FunctionSpace(mesh, "CG", 1)
        f1 = Function(V)
        if "supermesh" in tag:
            mesh2 = UnitSquareMesh(1, 1, diagonal='right')
            f2 = Function(FunctionSpace(mesh, "CG", 1))
        else:
            f2 = Function(V)
        f2.assign(1.0)
        if tag == "function assign":
            f1.assign(f2, ad_block_tag=tag)
        elif tag == "interpolate":
            f1 = interpolate(f2, V, ad_block_tag=tag)
        elif tag == "interpolate method":
            f1.interpolate(f2, ad_block_tag=tag)
        elif "project method" in tag:
            f1.project(f2, ad_block_tag=tag)
        elif "project" in tag:
            f1 = project(f2, V, ad_block_tag=tag)
        else:
            raise ValueError
    tape = get_working_tape()
    tags = tape.get_tags()
    assert len(tags) == 1 and tag in tags
    blocks = tape.get_blocks(tag=tag)
    assert len(blocks) == 1
