from fenics import *
from fenics_adjoint import *
import ufl_legacy as ufl
import numpy as np
from numpy.random import randn, random
import pickle


class ANN(object):
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            # Load weights.
            return cls.load(args[0])
        else:
            return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.mesh = kwargs.get("mesh")
        if isinstance(args[0], str):
            # Pickle has loaded the weights.
            self.load_weights()
        else:
            layers = args[0]
            bias = kwargs.get("bias")
            sigma = kwargs.get("sigma", ufl.tanh)
            init_method = kwargs.get("init_method", "normal")
            self.weights = generate_weights(self.mesh, layers, bias, init_method=init_method)
            self.layers = layers
            self.bias = bias
            self.sigma = sigma
            self.ctrls = None
            self.backup_weights_flat = None

    def save(self, name):
        with open(f"weights/{name}", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name):
        with open(f"weights/{name}", "rb") as f:
            return pickle.load(f)

    def load_weights(self):
        weights = generate_weights(self.mesh, self.layers, self.bias)
        for weight, sweight in zip(weights, self.weights):
            for w, sw in zip(weight["coefficient"], sweight["coefficient"]):
                w.vector()[:] = sw
            if "bias" in weight:
                weight["bias"].vector()[:] = sweight["bias"]
        self.weights = weights

    def __getstate__(self):
        r = self.__dict__.copy()
        weights = []
        for weight in self.weights:
            app = {}
            coefficient = []
            for w in weight["coefficient"]:
                if not isinstance(w, np.ndarray):
                    w = w.vector().get_local()
                coefficient.append(w)
            app["coefficient"] = coefficient
            if "bias" in weight:
                bias = weight["bias"]
                if not isinstance(bias, np.ndarray):
                    bias = bias.vector().get_local()
                app["bias"] = bias
            weights.append(app)
        r["weights"] = weights
        r["mesh"] = None
        r["ctrls"] = None
        r["backup_weights_flat"] = None
        return r

    def __call__(self, inputs):
        return NN(inputs, self.weights, self.sigma)

    def weights_flat(self):
        ctrls = self.weights_ctrls()
        r = []
        for ctrl in ctrls:
            r.append(ctrl.tape_value())
        return r

    def weights_ctrls(self):
        if self.ctrls is None:
            r = []
            for weight in self.weights:
                for w in weight["coefficient"]:
                    r.append(Control(w))
                if "bias" in weight:
                    r.append(Control(weight["bias"]))
            self.ctrls = r
        return self.ctrls

    def opt_callback(self, *args, **kwargs):
        r = []
        for ctrl in self.weights_ctrls():
            r.append(ctrl.tape_value()._ad_create_checkpoint())
        self.backup_weights_flat = r

    def set_weights(self, weights):
        i = 0
        for weight in self.weights:
            for w in weight["coefficient"]:
                w.vector()[:] = weights[i].vector()
                w.block_variable.save_output()
                i += 1
            if "bias" in weight:
                weight["bias"].vector()[:] = weights[i].vector()
                weight["bias"].block_variable.save_output()
                i += 1


def generate_weights(mesh, layers, bias, init_method="normal"):
    init_method = init_method.lower()
    assert init_method in ["normal", "uniform"]

    weights = []
    for i in range(len(layers) - 1):
        R_layer, vector_layer = max(layers[i + 1], layers[i]), min(layers[i + 1], layers[i])
        R = VectorFunctionSpace(mesh, "R", 0, dim=R_layer)
        ws = []
        weight = {}
        for _ in range(vector_layer):
            W = Function(R)
            if init_method == "uniform":
                W.vector()[:] = random(R.dim())
            elif init_method == "normal":
                W.vector()[:] = np.sqrt(2 / vector_layer) * randn(R.dim())
            ws.append(W)
        weight["coefficient"] = ws
        if bias[i]:
            R = VectorFunctionSpace(mesh, "R", 0, dim=layers[i + 1])
            weight["bias"] = Function(R)
        weights.append(weight)
    return weights


def NN(inputs, weights, sigma):
    r = as_vector(inputs)
    depth = len(weights)
    for i, weight in enumerate(weights):
        l = []
        for w in weight["coefficient"]:
            l.append(w)
        vec = as_vector(l)
        if w.ufl_shape[0] != r.ufl_shape[0]:
            vec = ufl.transpose(vec)
        term = vec * r
        if "bias" in weight:
            term += weight["bias"]
        if i + 1 >= depth:
            r = term
        else:
            r = nonlin_function(term, func=sigma)

    if r.ufl_shape[0] == 1:
        return r[0]
    return r


def identity(x):
    return x


class ELU(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, *args, **kwargs):
        return conditional(ufl.gt(x, 0), x, self.alpha * (ufl.exp(x) - 1))


def sigmoid(x):
    return 1 / (1 + ufl.exp(-x))


def nonlin_function(vec, func=ufl.tanh):
    v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
    return ufl.as_vector(v)
