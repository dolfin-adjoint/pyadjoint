import backend
from .tape import Tape, Block

def assemble(*args, **kwargs):
	tape = Tape.get_tape()

	fill_tape = kwargs.pop("fill_tape", True)
	output = backend.assemble(*args, **kwargs)

	if fill_tape:
		form = args[0]

		# Create the block and add it to the tape.
		block = AssembleBlock.create_block(tape, form)

		output = block.create_reference_type(output)
		
		# For each coefficient in the form, write the reference
		# so we know where to store the adjoint Jacobian output
		#for c in self.form.coefficients():
		#	tape.append_constant(c.adj_jac_output_ref)

		# Allocate a new variable to store adjoint Jacobian input
		#ref = tape.append_variable()

		# Write



	return output



class AssembleBlock(Block):
	def _init_(self, form):
		self.form = form
		for c in self.form.coefficients():
			self.add_dependency(c)

	def evaluate_adj(self):
		adj_input = self.fwd_outputs[0].get_adj_output()

		for c in self.dependencies:
			if isinstance(c, backend.Function):
				dc = backend.TestFunction(c.function_space())
			elif isinstance(c, backend.Constant):
				dc = backend.Constant(1)

			dform = backend.derivative(self.form, c, dc)
			output = backend.assemble(dform)
			c.add_adj_output(adj_input * output)