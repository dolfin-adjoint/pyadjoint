import backend
import ufl


class UFLFormCoefficient(ufl.coefficient.Coefficient):
    def __init__(self, v):
        self.v = v
        self._hash = hash(v)
        self._count = v.count()
        self._repr = repr(v)
        self._str = str(v)
        self._record_items = False

    def __getattr__(self, attr):
        return getattr(self.v, attr)

    def __getitem__(self, item):
        return type(self.v).__getitem__(self, item)

    @property
    def __class__(self):
        return self.v.__class__

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self._str

    def __truediv__(self, other):
        return self.v / other


class UFLForm(object):
    def __init__(self, form, coefficient_map):
        self.form = form
        self.subforms = {}
        self.coefficient_map = coefficient_map

    def derivative(self, coefficient, direction=None):
        subforms = self.subforms
        if coefficient not in subforms:
            subforms[coefficient] = [None, None]
        subform = subforms[coefficient]
        i = 0 if direction is None or isinstance(direction, ufl.Argument) else 1
        if subform[i] is None:
            if i == 1:
                placeholder = UFLFormCoefficient(direction)
            else:
                placeholder = direction
            if coefficient in self.coefficient_map:
                mapped_coefficient = self.coefficient_map[coefficient]
            else:
                mapped_coefficient = coefficient
            subform[i] = (UFLForm(ufl.algorithms.expand_derivatives(
                backend.derivative(self.form, mapped_coefficient, placeholder)),
                                     self.coefficient_map),
                          placeholder)

        subform, *placeholder = subform[i]
        if i == 1:
            placeholder[0].v = direction
        return subform

    def adjoint(self):
        subforms = self.subforms
        if "__adjoint__" not in subforms:
            subforms["__adjoint__"] = UFLForm(backend.adjoint(self.form),
                                                 self.coefficient_map)
        return subforms["__adjoint__"]

    def action(self, coefficient):
        subforms = self.subforms
        if "__action__" not in subforms:
            placeholder = UFLFormCoefficient(coefficient)
            subforms["__action__"] = (UFLForm(backend.action(self.form, placeholder),
                                                 self.coefficient_map),
                                      placeholder)
        subform, placeholder = subforms["__action__"]
        placeholder.v = coefficient
        return subform

    def replace(self, replace_map):
        for coeff in replace_map:
            self.coefficient_map[coeff].v = replace_map[coeff]

    def replace_coefficient(self, coefficient, new_coefficient):
        self.coefficient_map[coefficient].v = new_coefficient

    def replace_with_argument(self, coeff, argument):
        subforms = self.subforms
        if "__replace_with_argument__" not in subforms:
            subforms["__replace_with_argument__"] = UFLForm(ufl.replace(self.form,
                                                                               {
                                                                                   self.coefficient_map[coeff]:
                                                                                       argument
                                                                               }),
                                                               self.coefficient_map)
        return subforms["__replace_with_argument__"]

    def lhs(self):
        subforms = self.subforms
        if "__lhs__" not in subforms:
            subforms["__lhs__"] = UFLForm(backend.lhs(self.form), self.coefficient_map)
        return subforms["__lhs__"]

    def rhs(self):
        subforms = self.subforms
        if "__rhs__" not in subforms:
            subforms["__rhs__"] = UFLForm(backend.rhs(self.form), self.coefficient_map)
        return subforms["__rhs__"]

    def __getattr__(self, item):
        return getattr(self.form, item)

    @property
    def __class__(self):
        return self.form.__class__

    def __hash__(self):
        return hash(self.form)

    def __str__(self):
        return str(self.form)

    def __neg__(self):
        subforms = self.subforms
        if "__neg__" not in subforms:
            subforms["__neg__"] = UFLForm(-self.form, self.coefficient_map)
        return subforms["__neg__"]

    def __eq__(self, other):
        return self.form == other

    def __add__(self, other):
        return UFLForm(self.form + other, self.coefficient_map)

    def __radd__(self, other):
        return self.__add__(other)

    def replace_variables(self):
        for variable in self.variables:
            coefficient = variable.output
            self.coefficient_map[coefficient].v = variable.saved_output

    @staticmethod
    def create(form, coefficient_map):
        placeholder_coeff_map = {}
        for coefficient in coefficient_map:
            placeholder_coeff_map[coefficient] = UFLFormCoefficient(coefficient_map[coefficient])
        form = ufl.replace(form, placeholder_coeff_map)
        return UFLForm(form, placeholder_coeff_map)
