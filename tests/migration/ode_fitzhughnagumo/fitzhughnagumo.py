

import numpy as _np
list_types = (_np.ndarray, list)
inf = float("infinity")


def value_formatter(value, width=0):
    """
    Return a formated string of a value

    Arguments
    ---------
    value : any
        The value which is formatted
    width : int
        A min str length value
    """
    ret = None
    if isinstance(value, list_types):
        if len(value) > 4:
            if isinstance(value[0], integers):
                formatstr = "[%d, %d, ..., %d, %d]"
            elif isinstance(value[0], scalars):
                formatstr = "[%%.%(ff)s, %%.%(ff)s, ..., %%.%(ff)s, %%.%(ff)s]" % \
                            float_format()
            else:
                formatstr = "[%s, %s, ..., %s, %s]"
            ret = formatstr % (value[0], value[1], value[-2], value[-1])
        elif len(value) == 0:
            ret = "[]"
        else:
            if isinstance(value[0], integers):
                formatstr = "%d"
            elif isinstance(value[0], scalars):
                formatstr = "%%.%(ff)s" % float_format()
            else:
                formatstr = "%s"

            formatstr = "[%s]" % (", ".join(formatstr for i in range(len(value))))
            ret = formatstr % tuple(value)

    elif isinstance(value, float):
        if value == inf:
            ret = "\xe2\x88\x9e"
        elif value == -inf:
            ret = "-\xe2\x88\x9e"

    elif isinstance(value, str):
        ret = repr(value)

    if ret is None:
        ret = str(value)

    if width == 0:
        return ret
    return VALUE_JUST(ret, width)


class Range(object):
    """
    A simple class for helping checking a given value is within a certain range
    """

    def __init__(self, ge=None, le=None, gt=None, lt=None):
        """
        Create a Range

        Arguments
        ---------
        ge : scalar (optional)
            Greater than or equal, range control of argument
        le : scalar (optional)
            Lesser than or equal, range control of argument
        gt : scalar (optional)
            Greater than, range control of argument
        lt : scalar (optional)
            Lesser than, range control of argument
        """
        ops = [ge, gt, le, lt]
        opnames = ["ge", "gt", "le", "lt"]

        # Checking valid combinations of kwargs
        if le is not None and lt is not None:
            value_error("Cannot create a 'Range' including "
                        "both 'le' and 'lt'")
        if ge is not None and gt is not None:
            value_error("Cannot create a 'Range' including "
                        "both 'ge' and 'gt'")

        # Checking valid types for 'RangeChecks'
        for op, opname in zip(ops, opnames):
            if not (op is None or isinstance(op, scalars)):
                type_error("expected a scalar for the '%s' arg" % opname)

        # get limits
        minval = gt if gt is not None else ge if ge is not None else -inf
        maxval = lt if lt is not None else le if le is not None else inf

        if minval > maxval:
            value_error("expected the maxval to be larger than minval")

        # Dict for test and repr
        range_formats = {}
        range_formats["minop"] = ">=" if gt is None else ">"
        range_formats["maxop"] = "<=" if lt is None else "<"
        range_formats["minvalue"] = str(minval)
        range_formats["maxvalue"] = str(maxval)

        # Dict for pretty print
        range_formats["minop_format"] = "[" if gt is None else "("
        range_formats["maxop_format"] = "]" if lt is None else ")"
        range_formats["minformat"] = value_formatter(minval)
        range_formats["maxformat"] = value_formatter(maxval)
        self.range_formats = range_formats

        self.range_eval_str = "lambda value : _all(value %(minop)s %(minvalue)s) "\
                              "and _all(value %(maxop)s %(maxvalue)s)" %\
                              range_formats

        self._in_range = eval(self.range_eval_str)

        # Define some string used for pretty print
        self._range_str = "%(minop_format)s%(minformat)s, "\
                          "%(maxformat)s%(maxop_format)s" % range_formats

        self._in_str = "%%s \xe2\x88\x88 %s" % self._range_str

        self._not_in_str = "%%s \xe2\x88\x89 %s" % self._range_str

        self.arg_repr_str = ", ".join("%s=%s" % (opname, op)
                                      for op, opname in zip(ops, opnames)
                                      if op is not None)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.arg_repr_str)

    def __str__(self):
        return self._range_str

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self._in_str == other._in_str

    def __contains__(self, value):
        """
        Return True of value is in range

        Arguments
        ---------
        value : scalar%s
            A value to be used in checking range
        """ % ("" if _np is None else " and np.ndarray")
        if not isinstance(value, range_types):
            type_error("only scalars%s can be ranged checked" %
                       ("" if _np is None else " and np.ndarray"))
        return self._in_range(value)

    def format(self, value, width=0):
        """
        Return a formated range check of the value

        Arguments
        ---------
        value : scalar
            A value to be used in checking range
        width : int
            A min str length value
        """
        in_range = self.__contains__(value)

        if value in self:
            return self.format_in(value, width)
        return self.format_not_in(value, width)

    def format_in(self, value, width=0):
        """
        Return a formated range check

        Arguments
        ---------
        value : scalar
            A value to be used in checking range
        width : int
            A min str length value
        """

        return self._in_str % value_formatter(value, width)

    def format_not_in(self, value, width=0):
        """
        Return a formated range check

        Arguments
        ---------
        value : scalar
            A value to be used in checking range
        width : int
            A min str length value
        """

        return self._not_in_str % value_formatter(value, width)


def init_values(**values):
    """
    Init values
    """
    # Imports
    import dolfin

    # Init values
    # s=0.0, v=-85.0
    init_values = [0.0, -85.0]

    # State indices and limit checker
    state_ind = dict(s=(0, Range()), v=(1, Range()))

    for state_name, value in list(values.items()):
        if state_name not in state_ind:
            raise ValueError("{{0}} is not a state.".format(state_name))
        ind, range = state_ind[state_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(state_name,
                                                              range.format_not_in(value)))

        # Assign value
        init_values[ind] = value
    init_values = dolfin.Constant(tuple(init_values))

    return init_values


def default_parameters(**values):
    """
    Parameter values
    """
    # Imports
    import dolfin
    import dolfin_adjoint

    # Param values
    # a=0.13, b=0.013, c_1=0.26, c_2=0.1, c_3=1.0, v_peak=40.0, v_rest=-85.0
    param_values = [0.13, 0.013, 0.26, 0.1, 1.0, 40.0, -85.0]
    param_names = ["a", "b", "c", "c_1", "c_2", "c_3", "v_peak", "v_rest"]

    # Parameter indices and limit checker
    param_ind = dict(a=(0, Range()), b=(1, Range()), c_1=(2, Range()),
                     c_2=(3, Range()), c_3=(4, Range()), v_peak=(5, Range()), v_rest=(6,
                                                                                      Range()))

    for param_name, value in list(values.items()):
        if param_name not in param_ind:
            raise ValueError("{{0}} is not a param".format(param_name))
        ind, range = param_ind[param_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(param_name,
                                                              range.format_not_in(value)))

        # Assign value
        param_values[ind] = value

    params = []
    for (val, name) in zip(param_values, param_names):
        params.append(dolfin_adjoint.Constant(val, name=name))

    return params


def rhs(states, time, parameters, dy=None):
    """
    Compute right hand side
    """
    # Imports
    import ufl_legacy as ufl
    import dolfin

    # Assign states
    assert (isinstance(states, dolfin.Function))
    assert (states.function_space().depth() == 1)
    assert (states.function_space().num_sub_spaces() == 2)
    s, v = dolfin.split(states)

    # Assign parameters
    a, b, c_1, c_2, c_3, v_peak, v_rest = parameters
    v_amp = v_peak - v_rest
    v_th = v_rest + a * v_amp

    I = (v - v_rest) * (v - v_th) * (v_peak - v) * c_1 / (v_amp * v_amp) - (v
                                                                            - v_res t) * c_2 * s / v_amp

    # Init test function
    _v = dolfin.TestFunction(states.function_space())

    # Derivative for state s
    dy = ((-c_3 * s + v - v_rest) * b) * _v[0]

    # Derivative for state v
    dy += (I) * _v[1]

    dya = dolfin.assemble(dy * dolfin.dx)

    # Return dy
    return dy
