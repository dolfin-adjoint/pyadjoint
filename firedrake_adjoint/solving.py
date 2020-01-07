import backend
from fenics_adjoint.solving import SolveLinearSystemBlock, SolveVarFormBlock
from fenics_adjoint.variational_solver import LinearVariationalSolveBlock, NonlinearVariationalSolveBlock


def _init_params(self, args, kwargs, varform):
    if len(self.forward_args) <= 0:
        self.forward_args = args

    if len(self.forward_kwargs) <= 0:
        self.forward_kwargs = kwargs.copy()

    if len(self.adj_args) <= 0:
        self.adj_args = self.forward_args

    if len(self.adj_kwargs) <= 0:
        self.adj_kwargs = self.forward_kwargs.copy()

        if varform:
            if "J" in self.forward_kwargs:
                self.adj_kwargs["J"] = backend.adjoint(self.forward_kwargs["J"])
            if "Jp" in self.forward_kwargs:
                self.adj_kwargs["Jp"] = backend.adjoint(self.forward_kwargs["Jp"])

            if "M" in self.forward_kwargs:
                raise NotImplementedError("Annotation of adaptive solves not implemented.")
            self.adj_kwargs.pop("appctx", None)

    if "solver_parameters" in kwargs and "mat_type" in kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = kwargs["solver_parameters"]["mat_type"]

    if varform:
        if "appctx" in kwargs:
            self.assemble_kwargs["appctx"] = kwargs["appctx"]


def __SolveLinearSystemBlock__init_params(self, args, kwargs):
    super(SolveLinearSystemBlock, self)._init_solver_parameters(args, kwargs)
    _init_params(self, args, kwargs, varform=False)


SolveLinearSystemBlock._init_solver_parameters = __SolveLinearSystemBlock__init_params


def __SolveVarFormBlock__init_params(self, args, kwargs):
    super(SolveVarFormBlock, self)._init_solver_parameters(args, kwargs)
    _init_params(self, args, kwargs, varform=True)


def __SolveVarFormBlock__assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
    return super(SolveVarFormBlock, self)._assemble_and_solve_adj_eq(dFdu_adj_form, dJdu, compute_bdy)


SolveVarFormBlock._init_solver_parameters = __SolveVarFormBlock__init_params
SolveVarFormBlock._assemble_and_solve_adj_eq = __SolveVarFormBlock__assemble_and_solve_adj_eq


def __LinearVariationalSolveBlock__init_params(self, args, kwargs):
    super(LinearVariationalSolveBlock, self)._init_solver_parameters(args, kwargs)
    _init_params(self, args, kwargs, varform=True)


LinearVariationalSolveBlock._init_solver_parameters = __LinearVariationalSolveBlock__init_params


def __NonlinearVariationalSolveBlock__init_params(self, args, kwargs):
    super(NonlinearVariationalSolveBlock, self)._init_solver_parameters(args, kwargs)
    _init_params(self, args, kwargs, varform=True)


NonlinearVariationalSolveBlock._init_solver_parameters = __NonlinearVariationalSolveBlock__init_params
