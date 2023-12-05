import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Eagerly call PETSc finalize.

    This is required because of a diabolical ordering issue between petsc2py and
    pytest-xdist setup and teardown operations (see
    https://github.com/firedrakeproject/firedrake/issues/3245). Without this
    modification the ordering is:

        pytest init -> PETSc init -> pytest finalize -> PETSc finalize

    This is problematic because pytest finalize cleans up some state that causes
    PETSc finalize to crash. To get around this we call PETSc finalize earlier
    in the process resulting in:

        pytest init -> PETSc init -> PETSc finalize -> pytest finalize

    """
    # import must be inside the function to avoid calling petsc2py initialize here
    from petsc2py import PETSc

    PETSc._finalize()
