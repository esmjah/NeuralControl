import numpy as np
from casadi import *


class CstrModel:

    Tau = 60
    C1 = 5000
    C2 = 1e6
    Cp = 1000
    E1 = 10000
    E2 = 15000
    R = 1.987
    DHrx = 5000
    rho = 1

    def cstr_model(self, c_a, c_b, tem, tem_i, c_ai, c_bi):
        
        c_a = SX.sym("c_a")
        c_b = SX.sym("c_b")
        tem = SX.sym("tem")

        k1 = self.C1 * np.exp(-self.E1 / (self.R * tem))
        k2 = self.C2 * np.exp(-self.E2 / (self.R * tem))
        r = k1 * c_a - k2 * c_b
        J = -(2.009 * c_b - (0.001657 * tem_i) ^ 2)

        der_c_a = (c_ai - c_a) / self.Tau - r
        der_c_b = (c_bi - c_b) / self.Tau + r
        der_T = (tem_i - tem) / self.Tau + self.DHrx / (self.rho * self.Cp) * r

        return der_c_a, der_c_b, der_T, J

    def cstr_model_aug(self, c_ai, c_bi, c_a, c_b, tem, tem_i):

        k1 = self.C1 * np.exp(-self.E1 / (self.R * tem))
        k2 = self.C2 * np.exp(-self.E2 / (self.R * tem))
        r = k1 * c_a - k2 * c_b
        J = -(2.009 * c_b - (0.001657 * tem_i) ^ 2)

        der_c_ai = 0
        der_c_bi = 0
        der_c_a = (c_ai - c_a) / self.Tau - r
        der_c_b = (c_bi - c_b) / self.Tau + r
        der_T = (tem_i - tem) / self.Tau + self.DHrx / (self.rho * self.Cp) * r

        return der_c_ai, der_c_bi, der_c_a, der_c_b, der_T, J

    def ode_model(self, t, y, ydot, u, d):

        c_a = y[0]
        c_b = y[1]
        tem = y[2]
        tem_i = u
        c_ai = d[0]
        c_bi = d[1]

        der_c_a, der_c_b, der_T, J = self.cstr_model(c_a, c_b, tem, tem_i, c_ai, c_bi)
        ydot = [der_c_a, der_c_b, der_T]
        return ydot
