import backend
from .tape import Tape

class Function(backend.Function):
    def __init__(self, *args, **kwargs):
    	adj_jac_output_ref = kwargs.pop("adj_jac_output_ref", None)
    	tape = kwargs.pop("tape", None)

    	if tape:
    		self.tape = tape
    	else:
    		self.tape = Tape.get_tape()

    	if adj_jac_output_ref:
    		self.adj_jac_output_ref = adj_jac_output_ref
    	else:
    		self.adj_jac_output_ref = self.tape.add_output()

    	return backend.Function.__init__(self, *args, **kwargs)
 
    def set_initial_adj_jac_input(self, value):
        #self.tape.write_to_variable_at_index(self.adj_jac_output_ref, value)
        pass

    def get_adj_jac_output(self):
        #return self.tape.read_variable_at_index(self.adj_jac_output_ref)
        return 1

class AdjFloat(float):
    def __new__(cls, val, adj_jac_output_ref=None):
        return float.__new__(cls, val)

    def __init__(*args, **kwargs):
        self = args[0]
        if not "adj_jac_output_ref" in kwargs:
            self.adj_jac_output_ref = tape.add_output()
        else:
            self.adj_jac_output_ref = kwargs["adj_jac_output_ref"]
            del kwargs["adj_jac_output_ref"]
        return float.__init__(*args, **kwargs)