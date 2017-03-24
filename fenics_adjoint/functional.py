import backend
from .assembly import assemble

class Functional(object):
	
	def __init__(self, form):
		self.form = form
		self.block_output = assemble(self.form).get_block_output()

	def __call__(self):
		return self.block_output.output