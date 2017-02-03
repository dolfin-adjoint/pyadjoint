import backend
from .tape import Tape, Block, Function

def solve(eq, func, bc):
	output = backend.solve(eq, func, bc)

	tape = Tape.get_tape()
	SolveBlock.create_block(tape, eq, func, bc)

	return output



class SolveBlock(Block):
	def _init_(self, eq, func, bc):
		self.eq = eq
		self.bc = bc
		self.func = func

	def evaluate_adj(self):
		adj_var = Function(self.func.function_space())
		lhs = self.eq.lhs
		dFdu = backend.derivative(lhs, self.func, backend.TrialFunction(self.func.function_space()))
		#dFdu = backend.adjoint(dFdu)

		dFdu = backend.assemble(dFdu)
		self.bc.apply(dFdu)
		B = dFdu.copy()
		#A_mat, B_mat = backend.as_backend_type(dFdu).mat(), backend.as_backend_type(B).mat()
		#A_mat.transpose(B_mat)


		#print type(dFdu)
		#from numpy import transpose
		#adj_dFdu = transpose(dFdu.array())

		dJdu = self.func.get_adj_output()

		#J = self.func**2*backend.dx
		#dJdu = backend.derivative(J, self.func, backend.TestFunction(self.func.function_space()))
		#dJdu = backend.assemble(dJdu)

		#backend.solve(dFdu == dJdu, adj_var, self.bc, solver_parameters={'print_matrix': True, 'print_rhs': True})
		self.bc.apply(dJdu)
		#print B.array()
		backend.solve(B, adj_var.vector(), dJdu)

		for c in lhs.coefficients():
			if c != self.func:
				#dc = Function(c.function_space())
				#dc.vector()[:] = 1
				dFdm = backend.derivative(lhs, c, backend.TrialFunction(c.function_space()))
				dFdm = backend.assemble(-dFdm)
				#print dFdm.array()
				#self.bc.apply(dFdm)

				adj_dFdm = dFdm.copy()
				#print adj_dFdm.array()
				dFdm_mat, adj_dFdm_mat = backend.as_backend_type(dFdm).mat(), backend.as_backend_type(adj_dFdm).mat()
				dFdm_mat.transpose(adj_dFdm_mat)
				

				c.add_adj_output(adj_dFdm*adj_var.vector())


