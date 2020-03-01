"""Convolution-based filter operators."""

import numpy as np
from dedalus.core.field import Operand
from dedalus.core.operators import Operator, FutureField


class Convolve(Operator, FutureField):
    """Basic convolution operator."""

    name = 'Conv'

    def meta_constant(self, axis):
        return (self.args[0].meta[axis]['constant'] and
                self.args[1].meta[axis]['constant'])

    def check_conditions(self):
        # Coefficient layout
        arg0, arg1 = self.args
        return ((arg0.layout == self._coeff_layout) and
                (arg1.layout == self._coeff_layout))

    def operate(self, out):
        arg0, arg1 = self.args
        arg0.require_coeff_space()
        arg1.require_coeff_space()
        # Multiply coefficients
        out.layout = self._coeff_layout
        np.multiply(arg0.data, arg1.data, out=out.data)


def build_filter(domain, N):
    """Build sharp spectral filter field."""
    kmax = (N - 1) // 2
    eta = domain.new_field(name='eta')
    kx = domain.elements(0)
    ky = domain.elements(1)
    eta['c'] = 1
    eta['c'] *= np.abs(kx) <= kmax
    eta['c'] *= np.abs(ky) <= kmax
    Filter = lambda field, eta=eta: Convolve(eta, field)
    return Filter