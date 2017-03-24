from .tape import get_working_tape


class ReducedFunctional(object):

	def __init__(self, functional, control):
		self.functional = functional
		self.control = control
		self.tape = get_working_tape()

		for i, block in enumerate(self.tape.get_blocks()):
			if self.control.original_block_output in block.get_dependencies():
				self.block_idx = i
				break

	def derivative(self):
		self.tape.reset_variables()
		self.functional.block_output.set_initial_adj_input(1.0)
		self.tape.evaluate(self.block_idx)

		return self.control.get_adj_output()

	def __call__(self, value=None):
		if value is None:
			return self.functional()

		self.control.original_block_output.output = value
		value.set_block_output(self.control.original_block_output)

		blocks = self.tape.get_blocks()
		for i in range(self.block_idx, len(blocks)):
			blocks[i].recompute()

		return self.functional()





