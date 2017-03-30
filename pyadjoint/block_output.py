class BlockOutput(object):
    """References a block output variable.

    """
    id_cnt = 0

    def __init__(self, output):
        self.output = output
        self.adj_value = 0
        self.saved_output = None
        BlockOutput.id_cnt += 1
        self.id = BlockOutput.id_cnt

    def add_adj_output(self, val):
        self.adj_value += val

    def get_adj_output(self):
        #print "Bugger ut: ", self.adj_value
        #print self.output
        return self.adj_value

    def set_initial_adj_input(self, value):
        self.adj_value = value

    def reset_variables(self):
        self.adj_value = 0

    # TODO: Make this just an attribute. Extend with Property if needed later.
    def get_output(self):
        return self.output

    def save_output(self):
        # Previously I used 
        # self.saved_ouput = Function(self.output.function_space(), self.output.vector()) as
        # assign allocates a new vector (and promptly doesn't need nor 
        # modifies the old vector) However this does not work when we also want to save copies of
        # other functions, say an output function from a SolveBlock. As
        # backend.solve overwrites the vector of the solution function.

        # TODO: I just realized, this is backend.Function specific. Maybe we should
        # create some kind of copy abstract method.

        self.saved_output = self.output.copy(deepcopy=True)

    def get_saved_output(self):
        if self.saved_output:
            return self.saved_output
        else:
            return self.output

    def __str__(self):
        return str(self.output)

