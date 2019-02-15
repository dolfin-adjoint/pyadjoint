Change log
==========

dolfin-adjoint 2018.2.0.dev0 [2019-02-04]
-----------------------------------------

- Support for FEniCS 2018.2.0.dev0
- Added function `taylor_to_dict`, which automatically computes shape derivatives and hessians without user input
- Added support for shape optimization. Supports "naive" shape optimization, computing shape derivatives with the ufl function `CoordinateDerivative`, as well as more advanced optimization, where only the boundary mesh nodes are the design variables. Demo can be found in `examples/stokes-shape-ad`.

dolfin-adjoint 2018.1.0 [2018-10-05]
------------------------------------

- Support for FEniCS 2018.1.0

dolfin-adjoint 2017.2.1 [2018-10-05]
------------------------------------

- Merged `taylor_test` and `taylor_test_multiple`
- Use tensorflow for tape graph visualisation
- Add ROLSolver to pyadjoint optimization package
- Renamed ReducedFunctional.optimize to optimize_tape
- Now annotates function assignment of linear combinations
- create_overloaded_object is moved to pyadjoint, introducing the OverloadedType._ad_init_object method for the same purpose.
- Added UFLConstraint to dolfin-adjoint
- BlockVariable now has an attribute `marked_in_path`, which indicates if one needs to compute (adjoint/tlm/hessian) values for this block variable.
- Block.evaluate_adj and Block.evaluate_hessian now take a new bool argument `markings`, indicating if relevant block variables are marked.
- Added alternative Block methods for evaluating adjoint/tlm/hessian values, which hides some of the technical implementation details common for most evaluate_* Block methods.
- Add the class `Placeholder` to pyadjoint
- Add the method Control.tape_value for retrieving the value of an object on the tape

dolfin-adjoint 2017.2.0 [2018-01-11]
------------------------------------

- Initial release
