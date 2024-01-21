import random

from qiskit import *
from qiskit import QuantumCircuit
from qiskit.opflow import X, Y, Z, I
from qiskit.opflow.state_fns import StateFn
from qiskit.opflow import CircuitOp
from qiskit.opflow.expectations import PauliExpectation
from qiskit.opflow.converters import CircuitSampler
from qiskit.opflow import CircuitStateFn
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics


def mixer(beta, num_qubit):
    codeword = QuantumRegister(num_qubit, 'codeword')

    recieved = QuantumRegister(num_qubit, 'recieved')
    circ = QuantumCircuit(codeword, recieved)

    # circ.h(1)

    circ.rx(2 * beta, 1)

    # circ.h(1)

    circ.barrier()

    circ.barrier()
    return circ


def cost_unitary(gamma, num_qubit):
    codeword = QuantumRegister(num_qubit, 'codeword')

    recieved = QuantumRegister(num_qubit, 'recieved')

    circ = QuantumCircuit(codeword, recieved)
    for i in range(num_qubit):
        circ.cx(codeword[i], recieved[i])
        circ.rz((-gamma), recieved[i])
        circ.cx(codeword[i], recieved[i])
        circ.barrier()

    return circ


# gamma = Parameter("$\\gamma$")

# beta = Parameter("$\\beta$")

def create_qaoa_circ(theta, num_qubit, p, r):
    beta = theta[0]
    gamma = theta[1]

    codeword = QuantumRegister(num_qubit, 'codeword')
    recieved = QuantumRegister(num_qubit, 'recieved')

    circ = QuantumCircuit(codeword, recieved)

    # if the codeword recieved is r = |0010>, in qiskit the 1 is in the 1st position of recieved register.

    for i in range(len(r)):
        if r[i] == '1':
            circ.x(recieved[i])

    circ.h(range(2))

    circ.cx(0, 2)

    circ.barrier()

    for i in range(p):
        circ.append(mixer(beta, num_qubit), range(2 * num_qubit))
        circ.append(cost_unitary(gamma, num_qubit), range(2 * num_qubit))

    return circ

data = {}

layers = [2,3,4,5,6,7,8,9,10]

gamma = random.uniform(0.0,2*np.pi)

for p in layers:
    def expectation(theta0, theta1):
        beta = theta0
        gamma = theta1

        psi = create_qaoa_circ([beta, gamma], num_qubit=3, p, r='011')

        psi = CircuitStateFn(psi)

        H = (0.5 * I ^ I ^ I ^ I ^ I ^ I) + (0.5 * I ^ I ^ I ^ I ^ I ^ I) + (0.5 * I ^ I ^ I ^ I ^ I ^ I) + (
                    0.5 * I ^ I ^ I ^ I ^ I ^ I) \
            - (0.5 * Z ^ I ^ I ^ Z ^ I ^ I) - (0.5 * I ^ Z ^ I ^ I ^ Z ^ I) - (0.5 * I ^ I ^ Z ^ I ^ I ^ Z)

        measurable_expression = StateFn(H, is_measurement=True).compose(psi)

        expect = PauliExpectation().convert(measurable_expression)

        simulator = QasmSimulator()

        q_instance = QuantumInstance(simulator, shots=2000)

        sampler = CircuitSampler(q_instance).convert(expect)

        r = sampler.eval().real

        return r


    beta = np.arange(0.0,np.pi,0.01)
    # gamma = np.arange(0.0,2*np.pi,0.01)

    x, y = np.meshgrid(beta,gamma)
    f = np.vectorize(expectation)

    z = f(x,y)

    var = statistics.variance(z)

    data["Layer:"+str(p)] = var

data["layer"] = layers

df = pd.DataFrame(data)
df.to_excel("321_uni_landscape_barrenplot.xlsx", index=False)

# fig = plt.figure(figsize = (12,12))
# ax = plt.axes(projection='3d')
#
#
# # Creating color map
# my_cmap = plt.get_cmap('nipy_spectral')          # 'nipy_spectral' 'hsv' 'turbo' 'hot''ocean' 'rainbow' twilight
#
# surf = ax.plot_surface(x,y,z,cmap = my_cmap,
#                        edgecolor ='none',shade = True)
#
#
#
# #fig.colorbar(surf, ax, shrink=0.5, aspect=5)
#
# # Set axes label
# ax.set_xlabel("$\\beta$",fontsize = 15)
# ax.set_ylabel("$\\gamma$",fontsize = 15)
# ax.set_zlabel('$<H_f>$',fontsize = 15)
# ax.set_title('Optimization Landscape of $H_f$',fontsize = 25)
# fig.colorbar(surf)
#
# plt.show()

# plt.savefig('land321_300.png',dpi = 300)
