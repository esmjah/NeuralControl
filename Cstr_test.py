import numpy as np
from casadi import *



Tau = 60
C1 = 5000
C2 = 1e6
Cp = 1000
E1 = 10000
E2 = 15000
R = 1.987
DHrx = 5000
rho = 1

c_ai = u[0]
c_bi = u[1]
tem_i = u[2]


c_a = SX.sym("c_a")
c_b = SX.sym("c_b")
tem = SX.sym("tem")

k1 = C1 * np.exp(-E1 / (R * tem))
k2 = C2 * np.exp(-E2 / (R * tem))
r = k1 * c_a - k2 * c_b
J = -(2.009 * c_b - (0.001657 * tem_i) ^ 2)

der_c_a = (c_ai - c_a) / Tau - r
der_c_b = (c_bi - c_b) / Tau + r
der_T = (tem_i - tem) / Tau + DHrx / (rho * Cp) * r

