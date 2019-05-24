from fenics_adjoint.solving import SolveLinearSystemBlock, SolveVarFormBlock
from fenics_adjoint.variational_solver import LinearVariationalSolveBlock, NonlinearVariationalSolveBlock


def __SolveLinearSystemBlock__init_params(self, args, kwargs):
    super(SolveLinearSystemBlock, self)._init_solver_parameters(args, kwargs)
    if len(self.forward_args) <= 0:
        self.forward_args = args

    if len(self.forward_kwargs) <= 0:
        self.forward_kwargs = kwargs

    if "solver_parameters" in self.forward_kwargs and "mat_type" in self.forward_kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = self.forward_kwargs["solver_parameters"]["mat_type"]

    if len(self.adj_args) <= 0:
        self.adj_args = self.forward_args


SolveLinearSystemBlock._init_solver_parameters = __SolveLinearSystemBlock__init_params


def __SolveVarFormBlock__init_params(self, args, kwargs):
    super(SolveVarFormBlock, self)._init_solver_parameters(args, kwargs)
    if len(self.forward_args) <= 0:
        self.forward_args = args

    if len(self.forward_kwargs) <= 0:
        self.forward_kwargs = kwargs

    if "solver_parameters" in self.forward_kwargs and "mat_type" in self.forward_kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = self.forward_kwargs["solver_parameters"]["mat_type"]

    if len(self.adj_args) <= 0:
        # self.adj_args = tuple(self.forward_kwargs.get("solver_parameters", {}).values())
        self.adj_kwargs = self.forward_kwargs


def __SolveVarFormBlock__assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, bcs, compute_bdy=True):
    return super(SolveVarFormBlock, self)._assemble_and_solve_adj_eq(dFdu_adj_form, dJdu, bcs, compute_bdy)


SolveVarFormBlock._init_solver_parameters = __SolveVarFormBlock__init_params
SolveVarFormBlock._assemble_and_solve_adj_eq = __SolveVarFormBlock__assemble_and_solve_adj_eq


def __LinearVariationalSolveBlock__init_params(self, args, kwargs):
    super(LinearVariationalSolveBlock, self)._init_solver_parameters(args, kwargs)
    if len(self.adj_args) <= 0 and len(self.adj_kwargs) <= 0:
        self.adj_kwargs = self.solver_kwargs

    if "solver_parameters" in self.solver_kwargs and "mat_type" in self.solver_kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = self.solver_kwargs["solver_parameters"]["mat_type"]


LinearVariationalSolveBlock._init_solver_parameters = __LinearVariationalSolveBlock__init_params


def __NonlinearVariationalSolveBlock__init_params(self, args, kwargs):
    super(NonlinearVariationalSolveBlock, self)._init_solver_parameters(args, kwargs)
    if len(self.adj_args) <= 0 and len(self.adj_kwargs) <= 0:
        self.adj_kwargs = self.solver_kwargs

    if "solver_parameters" in self.solver_kwargs and "mat_type" in self.solver_kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = self.solver_kwargs["solver_parameters"]["mat_type"]


NonlinearVariationalSolveBlock._init_solver_parameters = __NonlinearVariationalSolveBlock__init_params
