import backend

class Tape(object):
	
	__slots__ = ["tape", "variables", "outputs"]	

	tape_instances = []

	def __init__(self):
		# Initialize the tape
		self.tape = []

		# Initialize a list of variables on the tape.
		# This list will contain outputs/inputs that change
		# during evalutation of the adjoint/tangent linear model.
		self.variables = []

		# A list which contains the outputs to the adj/tlm.
		self.outputs = []

	def clear_tape(self):
		self.tape = []

	def add_output(self):
		self.outputs.append(0)

		return len(self.outputs)-1

	def add_block(self, block):
		"""
		Adds a block to the tape and returns the index.
		"""
		self.tape.append(block)

		# len() is computed in constant time, so this should be fine.
		return len(self.tape)-1

	def evaluate(self):
		for i in range(len(self.tape)-1, -1, -1):
			self.tape[i].evaluate_adj()

	def reset_variables(self):
		for i in range(len(self.tape)-1, -1, -1):
			self.tape[i].reset_variables()

	@classmethod
	def get_tape(cls):
		"""
		A method to obtain the current tape or create a new one.
		"""
		if not cls.tape_instances:
			ret = cls()
			cls.tape_instances.append(ret)
			return ret

		return cls.tape_instances[-1]


class Block(object):
	"""Block is a class representing one block in the tape.
	Thus each block object corresponds to a basic operation in the forward model."""
	def __init__(self, tape, *args, **kwargs):
		self.tape = tape
		self.dependencies = []
		self.fwd_outputs = []
		self._init_(*args, **kwargs)

	def _init_(self, *args, **kwargs):
		return NotImplementedError

	def add_dependency(self, dep):
		self.dependencies.append(dep)

	def create_fwd_output(self, obj):
		self.fwd_outputs.append(obj)

	def reset_variables(self):
		for dep in self.dependencies:
			dep.reset_variables()

	def create_reference_type(self, output):
		if isinstance(output, float):
			cls = AdjFloat
		elif isinstance(output, backend.Function):
			cls = Function
		else:
			return NotImplementedError

		ret = cls(output)
		self.create_fwd_output(ret)

		return ret


	@classmethod
	def create_block(cls, tape, *args, **kwargs):
		"""
		A method that creates a block and appends it on the tape.
		"""
		block = cls(tape, *args, **kwargs)
		block.tape_idx = tape.add_block(block)
		return block

class OverloadedType(object):
	def __init__(self, *args, **kwargs):
		tape = kwargs.pop("tape", None)

		if tape:
			self.tape = tape
		else:
			self.tape = Tape.get_tape()

		self.adj_value = 0

		self._init_(*args, **kwargs)

	def _init_(self):
		return NotImplementedError

	def add_adj_output(self, val):
		self.adj_value += val

	def get_adj_output(self):
		return self.adj_value

	def set_initial_adj_input(self, value):
		self.adj_value = value

	def reset_variables(self):
		self.adj_value = 0


class Function(OverloadedType, backend.Function):
	def _init_(self, *args, **kwargs):
		backend.Function.__init__(self, *args, **kwargs)

class Constant(OverloadedType, backend.Constant):
	def _init_(self, *args, **kwargs):
		backend.Constant.__init__(self, *args, **kwargs)

class AdjFloat(OverloadedType, float):
	def __new__(cls, *args, **kwargs):
		return float.__new__(cls, *args)

	def _init_(self, *args, **kwargs):
		float.__init__(self, *args, **kwargs)

	def __mul__(self, other):
		output = float.__mul__(self, other)
		if output is NotImplemented:
			return NotImplemented

		block = AdjFloat.MulBlock.create_block(self.tape, self, other)

		output = block.create_reference_type(output)

		return output 


	class MulBlock(Block):
		def _init_(self, lfactor, rfactor):
			self.lfactor = lfactor
			self.rfactor = rfactor

		def evaluate_adj(self):
			adj_input = self.fwd_outputs[0].get_adj_output()

			self.rfactor.add_adj_output(adj_input * self.lfactor)
			self.lfactor.add_adj_output(adj_input * self.rfactor)
