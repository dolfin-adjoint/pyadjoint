
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
        if len(value)>4:
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

            formatstr = "[%s]" % (", ".join(formatstr for i in range(len(value))) )
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
            value_error("Cannot create a 'Range' including "\
                        "both 'le' and 'lt'")
        if ge is not None and gt is not None:
            value_error("Cannot create a 'Range' including "\
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
                              "and _all(value %(maxop)s %(maxvalue)s)"%\
                              range_formats

        self._in_range = eval(self.range_eval_str)

        # Define some string used for pretty print
        self._range_str = "%(minop_format)s%(minformat)s, "\
                          "%(maxformat)s%(maxop_format)s" % range_formats

        self._in_str = "%%s \xe2\x88\x88 %s" % self._range_str

        self._not_in_str = "%%s \xe2\x88\x89 %s" % self._range_str

        self.arg_repr_str = ", ".join("%s=%s" % (opname, op) \
                                      for op, opname in zip(ops, opnames) \
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
            type_error("only scalars%s can be ranged checked" % \
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
    # Xr1=0, Xr2=1, Xs=0, m=0, h=0.75, j=0.75, d=0, f=1, fCa=1, s=1, r=0,
    # Ca_SR=0.2, Ca_i=0.0002, g=1, Na_i=11.6, V=-86.2, K_i=138.3
    init_values = [0, 1, 0, 0, 0.75, 0.75, 0, 1, 1, 1, 0, 0.2, 0.0002, 1,\
        11.6, -86.2, 138.3]

    # State indices and limit checker
    state_ind = dict(Xr1=(0, Range()), Xr2=(1, Range()), Xs=(2, Range()),\
        m=(3, Range()), h=(4, Range()), j=(5, Range()), d=(6, Range()), f=(7,\
        Range()), fCa=(8, Range()), s=(9, Range()), r=(10, Range()),\
        Ca_SR=(11, Range()), Ca_i=(12, Range()), g=(13, Range()), Na_i=(14,\
        Range()), V=(15, Range()), K_i=(16, Range()))

    for state_name, value in list(values.items()):
        if state_name not in state_ind:
            raise ValueError("{{0}} is not a state.".format(state_name))
        ind, range = state_ind[state_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(state_name,\
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

    # Param values
    # P_kna=0.03, g_K1=5.405, g_Kr=0.096, g_Ks=0.062, g_Na=14.838,
    # g_bna=0.00029, g_CaL=0.000175, g_bca=0.000592, g_to=0.294,
    # K_mNa=40, K_mk=1, P_NaK=1.362, K_NaCa=1000, K_sat=0.1,
    # Km_Ca=1.38, Km_Nai=87.5, alpha=2.5, gamma=0.35, K_pCa=0.0005,
    # g_pCa=0.825, g_pK=0.0146, Buf_c=0.15, Buf_sr=10, Ca_o=2,
    # K_buf_c=0.001, K_buf_sr=0.3, K_up=0.00025, V_leak=8e-05,
    # V_sr=0.001094, Vmax_up=0.000425, a_rel=0.016464, b_rel=0.25,
    # c_rel=0.008232, tau_g=2, Na_o=140, Cm=0.185, F=96485.3415,
    # R=8314.472, T=310, V_c=0.016404, stim_amplitude=0,
    # stim_duration=1, stim_period=1000, stim_start=1, K_o=5.4
    param_values = [0.03, 5.405, 0.096, 0.062, 14.838, 0.00029, 0.000175,\
        0.000592, 0.294, 40, 1, 1.362, 1000, 0.1, 1.38, 87.5, 2.5, 0.35,\
        0.0005, 0.825, 0.0146, 0.15, 10, 2, 0.001, 0.3, 0.00025, 8e-05,\
        0.001094, 0.000425, 0.016464, 0.25, 0.008232, 2, 140, 0.185,\
        96485.3415, 8314.472, 310, 0.016404, 0, 1, 1000, 1, 5.4]

    # Parameter indices and limit checker
    param_ind = dict(P_kna=(0, Range()), g_K1=(1, Range()), g_Kr=(2,\
        Range()), g_Ks=(3, Range()), g_Na=(4, Range()), g_bna=(5, Range()),\
        g_CaL=(6, Range()), g_bca=(7, Range()), g_to=(8, Range()), K_mNa=(9,\
        Range()), K_mk=(10, Range()), P_NaK=(11, Range()), K_NaCa=(12,\
        Range()), K_sat=(13, Range()), Km_Ca=(14, Range()), Km_Nai=(15,\
        Range()), alpha=(16, Range()), gamma=(17, Range()), K_pCa=(18,\
        Range()), g_pCa=(19, Range()), g_pK=(20, Range()), Buf_c=(21,\
        Range()), Buf_sr=(22, Range()), Ca_o=(23, Range()), K_buf_c=(24,\
        Range()), K_buf_sr=(25, Range()), K_up=(26, Range()), V_leak=(27,\
        Range()), V_sr=(28, Range()), Vmax_up=(29, Range()), a_rel=(30,\
        Range()), b_rel=(31, Range()), c_rel=(32, Range()), tau_g=(33,\
        Range()), Na_o=(34, Range()), Cm=(35, Range()), F=(36, Range()),\
        R=(37, Range()), T=(38, Range()), V_c=(39, Range()),\
        stim_amplitude=(40, Range()), stim_duration=(41, Range()),\
        stim_period=(42, Range()), stim_start=(43, Range()), K_o=(44,\
        Range()))

    for param_name, value in list(values.items()):
        if param_name not in param_ind:
            raise ValueError("{{0}} is not a param".format(param_name))
        ind, range = param_ind[param_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(param_name,\
                range.format_not_in(value)))

        # Assign value
        param_values[ind] = value
    param_values = dolfin.Constant(tuple(param_values))

    return param_values

def rhs(states, time, parameters, dy=None):
    """
    Compute right hand side
    """
    # Imports
    import ufl_legacy as ufl
    import dolfin

    # Assign states
    assert(isinstance(states, dolfin.Function))
    assert(states.function_space().depth() == 1)
    assert(states.function_space().num_sub_spaces() == 17)
    Xr1, Xr2, Xs, m, h, j, d, f, fCa, s, r, Ca_SR, Ca_i, g, Na_i, V, K_i =\
        dolfin.split(states)

    # Assign parameters
    assert(isinstance(parameters, (dolfin.Function, dolfin.Constant)))
    if isinstance(parameters, dolfin.Function):
        assert(parameters.function_space().depth() == 1)
        assert(parameters.function_space().num_sub_spaces() == 45)
    else:
        assert(parameters.value_size() == 45)
    P_kna, g_K1, g_Kr, g_Ks, g_Na, g_bna, g_CaL, g_bca, g_to, K_mNa, K_mk,\
        P_NaK, K_NaCa, K_sat, Km_Ca, Km_Nai, alpha, gamma, K_pCa, g_pCa,\
        g_pK, Buf_c, Buf_sr, Ca_o, K_buf_c, K_buf_sr, K_up, V_leak, V_sr,\
        Vmax_up, a_rel, b_rel, c_rel, tau_g, Na_o, Cm, F, R, T, V_c,\
        stim_amplitude, stim_duration, stim_period, stim_start, K_o =\
        dolfin.split(parameters)

    # Reversal potentials
    E_Na = R*T*ufl.ln(Na_o/Na_i)/F
    E_K = R*T*ufl.ln(K_o/K_i)/F
    E_Ks = R*T*ufl.ln((Na_o*P_kna + K_o)/(Na_i*P_kna + K_i))/F
    E_Ca = 0.5*R*T*ufl.ln(Ca_o/Ca_i)/F

    # Inward rectifier potassium current
    alpha_K1 = 0.1/(1.0 + 6.14421235332821e-6*ufl.exp(0.06*V - 0.06*E_K))
    beta_K1 = (3.06060402008027*ufl.exp(0.0002*V - 0.0002*E_K) +\
        0.367879441171442*ufl.exp(0.1*V - 0.1*E_K))/(1.0 + ufl.exp(0.5*E_K -\
        0.5*V))
    xK1_inf = alpha_K1/(alpha_K1 + beta_K1)
    i_K1 = 0.430331482911935*ufl.sqrt(K_o)*(-E_K + V)*g_K1*xK1_inf

    # Rapid time dependent potassium current
    i_Kr = 0.430331482911935*ufl.sqrt(K_o)*(-E_K + V)*Xr1*Xr2*g_Kr

    # Rapid time dependent potassium current xr1 gate
    xr1_inf = 1.0/(1.0 + 0.0243728440732796*ufl.exp(-0.142857142857143*V))
    alpha_xr1 = 450.0/(1.0 + ufl.exp(-9/2 - V/10.0))
    beta_xr1 = 6.0/(1.0 + 13.5813245225782*ufl.exp(0.0869565217391304*V))
    tau_xr1 = alpha_xr1*beta_xr1

    # Rapid time dependent potassium current xr2 gate
    xr2_inf = 1.0/(1.0 + 39.1212839981532*ufl.exp(0.0416666666666667*V))
    alpha_xr2 = 3.0/(1.0 + 0.0497870683678639*ufl.exp(-0.05*V))
    beta_xr2 = 1.12/(1.0 + 0.0497870683678639*ufl.exp(0.05*V))
    tau_xr2 = alpha_xr2*beta_xr2

    # Slow time dependent potassium current
    i_Ks = (Xs*Xs)*(V - E_Ks)*g_Ks

    # Slow time dependent potassium current xs gate
    xs_inf = 1.0/(1.0 + 0.69967253737513*ufl.exp(-0.0714285714285714*V))
    alpha_xs = 1100.0/ufl.sqrt(1.0 +\
        0.188875602837562*ufl.exp(-0.166666666666667*V))
    beta_xs = 1.0/(1.0 + 0.0497870683678639*ufl.exp(0.05*V))
    tau_xs = alpha_xs*beta_xs

    # Fast sodium current
    i_Na = (m*m*m)*(-E_Na + V)*g_Na*h*j

    # Fast sodium current m gate
    m_inf = 1.0/((1.0 +\
        0.00184221158116513*ufl.exp(-0.110741971207087*V))*(1.0 +\
        0.00184221158116513*ufl.exp(-0.110741971207087*V)))
    alpha_m = 1.0/(1.0 + ufl.exp(-12.0 - V/5.0))
    beta_m = 0.1/(1.0 + 0.778800783071405*ufl.exp(0.005*V)) + 0.1/(1.0 +\
        ufl.exp(7.0 + V/5.0))
    tau_m = alpha_m*beta_m

    # Fast sodium current h gate
    h_inf = 1.0/((1.0 + 15212.5932856544*ufl.exp(0.134589502018843*V))*(1.0 +\
        15212.5932856544*ufl.exp(0.134589502018843*V)))
    alpha_h = 4.43126792958051e-7*ufl.exp(-0.147058823529412*V)/(1.0 +\
        2.3538526683702e+17*ufl.exp(1.0*V))
    beta_h = (310000.0*ufl.exp(0.3485*V) + 2.7*ufl.exp(0.079*V))/(1.0 +\
        2.3538526683702e+17*ufl.exp(1.0*V)) + 0.77*(1.0 - 1.0/(1.0 +\
        2.3538526683702e+17*ufl.exp(1.0*V)))/(0.13 +\
        0.0497581410839387*ufl.exp(-0.0900900900900901*V))
    tau_h = 1.0/(alpha_h + beta_h)

    # Fast sodium current j gate
    j_inf = 1.0/((1.0 + 15212.5932856544*ufl.exp(0.134589502018843*V))*(1.0 +\
        15212.5932856544*ufl.exp(0.134589502018843*V)))
    alpha_j = (37.78 + V)*(-6.948e-6*ufl.exp(-0.04391*V) -\
        25428.0*ufl.exp(0.2444*V))/((1.0 +\
        2.3538526683702e+17*ufl.exp(1.0*V))*(1.0 +\
        50262745825.954*ufl.exp(0.311*V)))
    beta_j = 0.6*(1.0 - 1.0/(1.0 +\
        2.3538526683702e+17*ufl.exp(1.0*V)))*ufl.exp(0.057*V)/(1.0 +\
        0.0407622039783662*ufl.exp(-0.1*V)) +\
        0.02424*ufl.exp(-0.01052*V)/((1.0 +\
        2.3538526683702e+17*ufl.exp(1.0*V))*(1.0 +\
        0.00396086833990426*ufl.exp(-0.1378*V)))
    tau_j = 1.0/(alpha_j + beta_j)

    # Sodium background current
    i_b_Na = (-E_Na + V)*g_bna

    # L type ca current
    i_CaL = 4.0*(F*F)*(-0.341*Ca_o +\
        Ca_i*ufl.exp(2.0*F*V/(R*T)))*V*d*f*fCa*g_CaL/((-1.0 +\
        ufl.exp(2.0*F*V/(R*T)))*R*T)

    # L type ca current d gate
    d_inf = 1.0/(1.0 + 0.513417119032592*ufl.exp(-0.133333333333333*V))
    alpha_d = 0.25 + 1.4/(1.0 +\
        0.0677244716592409*ufl.exp(-0.0769230769230769*V))
    beta_d = 1.4/(1.0 + ufl.exp(1.0 + V/5.0))
    gamma_d = 1.0/(1.0 + 12.1824939607035*ufl.exp(-0.05*V))
    tau_d = gamma_d + alpha_d*beta_d

    # L type ca current f gate
    f_inf = 1.0/(1.0 + 17.4117080633276*ufl.exp(0.142857142857143*V))
    tau_f = 80.0 + 165.0/(1.0 + ufl.exp(5/2 - V/10.0)) +\
        1125.0*ufl.exp(-0.00416666666666667*((27.0 + V)*(27.0 + V)))

    # L type ca current fca gate
    alpha_fCa = 1.0/(1.0 + 8.03402376701711e+27*ufl.elem_pow(Ca_i, 8.0))
    beta_fCa = 0.1/(1.0 + 0.00673794699908547*ufl.exp(10000.0*Ca_i))
    gama_fCa = 0.2/(1.0 + 0.391605626676799*ufl.exp(1250.0*Ca_i))
    fCa_inf = 0.157534246575342 + 0.684931506849315*gama_fCa +\
        0.684931506849315*beta_fCa + 0.684931506849315*alpha_fCa
    tau_fCa = 2.0
    d_fCa = (-fCa + fCa_inf)/tau_fCa

    # Calcium background current
    i_b_Ca = (V - E_Ca)*g_bca

    # Transient outward current
    i_to = (-E_K + V)*g_to*r*s

    # Transient outward current s gate
    s_inf = 1.0/(1.0 + ufl.exp(4.0 + V/5.0))
    tau_s = 3.0 + 85.0*ufl.exp(-0.003125*((45.0 + V)*(45.0 + V))) + 5.0/(1.0 +\
        ufl.exp(-4.0 + V/5.0))

    # Transient outward current r gate
    r_inf = 1.0/(1.0 + 28.0316248945261*ufl.exp(-0.166666666666667*V))
    tau_r = 0.8 + 9.5*ufl.exp(-0.000555555555555556*((40.0 + V)*(40.0 + V)))

    # Sodium potassium pump current
    i_NaK = K_o*Na_i*P_NaK/((K_mk + K_o)*(Na_i + K_mNa)*(1.0 +\
        0.0353*ufl.exp(-F*V/(R*T)) + 0.1245*ufl.exp(-0.1*F*V/(R*T))))

    # Sodium calcium exchanger current
    i_NaCa = (-(Na_o*Na_o*Na_o)*Ca_i*alpha*ufl.exp((-1.0 + gamma)*F*V/(R*T))\
        + (Na_i*Na_i*Na_i)*Ca_o*ufl.exp(F*V*gamma/(R*T)))*K_NaCa/((1.0 +\
        K_sat*ufl.exp((-1.0 + gamma)*F*V/(R*T)))*((Na_o*Na_o*Na_o) +\
        (Km_Nai*Km_Nai*Km_Nai))*(Km_Ca + Ca_o))

    # Calcium pump current
    i_p_Ca = Ca_i*g_pCa/(K_pCa + Ca_i)

    # Potassium pump current
    i_p_K = (-E_K + V)*g_pK/(1.0 +\
        65.4052157419383*ufl.exp(-0.167224080267559*V))

    # Calcium dynamics
    i_rel = ((Ca_SR*Ca_SR)*a_rel/((Ca_SR*Ca_SR) + (b_rel*b_rel)) + c_rel)*d*g
    i_up = Vmax_up/(1.0 + (K_up*K_up)/(Ca_i*Ca_i))
    i_leak = (-Ca_i + Ca_SR)*V_leak
    g_inf = (1.0 - 1.0/(1.0 + 0.0301973834223185*ufl.exp(10000.0*Ca_i)))/(1.0 +\
        1.97201988740492e+55*ufl.elem_pow(Ca_i, 16.0)) + 1.0/((1.0 +\
        0.0301973834223185*ufl.exp(10000.0*Ca_i))*(1.0 +\
        5.43991024148102e+20*ufl.elem_pow(Ca_i, 6.0)))
    d_g = (-g + g_inf)/tau_g
    Ca_i_bufc = 1.0/(1.0 + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i)))
    Ca_sr_bufsr = 1.0/(1.0 + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr +\
        Ca_SR)))

    # Sodium dynamics

    # Membrane
    i_Stim = -(1.0 - 1.0/(1.0 + ufl.exp(-5.0*stim_start +\
        5.0*time)))*stim_amplitude/(1.0 + ufl.exp(-5.0*stim_start + 5.0*time\
        - 5.0*stim_duration))

    # Potassium dynamics

    # The ODE system: 17 states

    # Init test function
    _v = dolfin.TestFunction(states.function_space())

    # Derivative for state Xr1
    dy = ((-Xr1 + xr1_inf)/tau_xr1)*_v[0]

    # Derivative for state Xr2
    dy += ((-Xr2 + xr2_inf)/tau_xr2)*_v[1]

    # Derivative for state Xs
    dy += ((-Xs + xs_inf)/tau_xs)*_v[2]

    # Derivative for state m
    dy += ((-m + m_inf)/tau_m)*_v[3]

    # Derivative for state h
    dy += ((-h + h_inf)/tau_h)*_v[4]

    # Derivative for state j
    dy += ((j_inf - j)/tau_j)*_v[5]

    # Derivative for state d
    dy += ((d_inf - d)/tau_d)*_v[6]

    # Derivative for state f
    dy += ((-f + f_inf)/tau_f)*_v[7]

    # Derivative for state fCa
    dy += ((1.0 - 1.0/((1.0 + ufl.exp(60.0 + V))*(1.0 + ufl.exp(-10.0*fCa +\
        10.0*fCa_inf))))*d_fCa)*_v[8]

    # Derivative for state s
    dy += ((-s + s_inf)/tau_s)*_v[9]

    # Derivative for state r
    dy += ((-r + r_inf)/tau_r)*_v[10]

    # Derivative for state Ca_SR
    dy += ((-i_leak + i_up - i_rel)*Ca_sr_bufsr*V_c/V_sr)*_v[11]

    # Derivative for state Ca_i
    dy += ((-i_up - (i_CaL + i_p_Ca + i_b_Ca - 2.0*i_NaCa)*Cm/(2.0*F*V_c) +\
        i_leak + i_rel)*Ca_i_bufc)*_v[12]

    # Derivative for state g
    dy += ((1.0 - 1.0/((1.0 + ufl.exp(60.0 + V))*(1.0 + ufl.exp(-10.0*g +\
        10.0*g_inf))))*d_g)*_v[13]

    # Derivative for state Na_i
    dy += ((-3.0*i_NaK - 3.0*i_NaCa - i_Na - i_b_Na)*Cm/(F*V_c))*_v[14]

    # Derivative for state V
    dy += (-i_Ks - i_to - i_Kr - i_p_K - i_NaK - i_NaCa - i_Na - i_p_Ca -\
        i_b_Na - i_CaL - i_Stim - i_K1 - i_b_Ca)*_v[15]

    # Derivative for state K_i
    dy += ((-i_Ks - i_to - i_Kr - i_p_K - i_Stim - i_K1 +\
        2.0*i_NaK)*Cm/(F*V_c))*_v[16]

    # Return dy
    return dy
