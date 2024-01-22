import numpy as np
import random
from qiskit import *
from qiskit.visualization import plot_histogram
import pandas as pd

from qiskit.circuit import Parameter

import matplotlib.pyplot as plt

def reverse(string):
    str = ""
    for i in string:
        str = i + str
    return str


def mixer(beta, num_qubit):
    codeword = QuantumRegister(num_qubit, 'codeword')

    recieved = QuantumRegister(num_qubit, 'recieved')
    circ = QuantumCircuit(codeword, recieved)

    circ.h(0)
    circ.h(1)
    circ.h(2)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.rz(-2 * beta, 2)
    circ.cx(1, 2)
    circ.cx(0, 1)
    circ.h(0)
    circ.h(1)
    circ.h(2)

    circ.barrier()

    circ.h(0)
    circ.h(4)
    circ.h(5)
    circ.cx(0, 4)
    circ.cx(4, 5)
    circ.rz(-2 * beta, 5)
    circ.cx(4, 5)
    circ.cx(0, 4)
    circ.h(0)
    circ.h(4)
    circ.h(5)

    circ.barrier()

    circ.h(2)
    circ.h(3)
    circ.h(4)
    circ.cx(2, 3)
    circ.cx(3, 4)
    circ.rz(-2 * beta, 4)
    circ.cx(3, 4)
    circ.cx(2, 3)
    circ.h(2)
    circ.h(3)
    circ.h(4)

    circ.barrier()

    circ.h(1)
    circ.h(3)
    circ.h(5)
    circ.cx(1, 3)
    circ.cx(3, 5)
    circ.rz(-2 * beta, 5)
    circ.cx(3, 5)
    circ.cx(1, 3)
    circ.h(1)
    circ.h(3)
    circ.h(5)

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


gamma1 = Parameter("$\\gamma$")  # np.pi/2

beta1 = Parameter("$\\beta$")  # np.pi/3


################ theta is the initial parameters [beta,gamma] :

def create_qaoa_circ(optm_param, theta, num_qubit, p, r):  # beta and gamma are lists here

    beta1 = theta[0]
    gamma1 = theta[1]

    beta = []
    gamma = []

    if p > 1:
        j = 0
        while j < ((2 * p) - 2):  # for i in range(len(theta)-1,2):
            beta.append(optm_param[j])
            gamma.append(optm_param[j + 1])
            j = j + 2

    beta.append(beta1)
    gamma.append(gamma1)

    codeword = QuantumRegister(num_qubit, 'codeword')
    recieved = QuantumRegister(num_qubit, 'recieved')

    circ = QuantumCircuit(codeword, recieved)

    # if the codeword recieved is r = |0010>, in qiskit the 1 is in the 1st position of recieved register.

    for i in range(len(r)):
        if r[i] == '1':
            circ.x(recieved[i])

    circ.h(range(3))

    circ.cx(0, 4)
    circ.cx(0, 5)

    circ.cx(1, 3)
    circ.cx(1, 5)

    circ.cx(2, 3)
    circ.cx(2, 4)

    circ.barrier()

    for i in range(p):
        circ.append(cost_unitary(gamma[i], num_qubit), range(2 * num_qubit))
        circ.append(mixer(beta[i], num_qubit), range(2 * num_qubit))


    circ.measure_all()

    return circ


################ theta is the initial parameters [beta,gamma] :

def create_qaoa_circ1(optm_param, num_qubit, p, r):  # beta and gamma are lists here

    beta = []
    gamma = []

    if p > 1:
        j = 0
        while j < (2 * p):  # for i in range(len(theta)-1,2):
            beta.append(optm_param[j])
            gamma.append(optm_param[j + 1])
            j = j + 2
    if p == 1:
        beta = [optm_param[0]]
        gamma = [optm_param[1]]
    codeword = QuantumRegister(num_qubit, 'codeword')
    recieved = QuantumRegister(num_qubit, 'recieved')
    cbit = ClassicalRegister(num_qubit)
    circ = QuantumCircuit(codeword, recieved, cbit)

    # if the codeword recieved is r = |0010>, in qiskit the 1 is in the 1st position of recieved register.

    for i in range(len(r)):
        if r[i] == '1':
            circ.x(recieved[i])

    circ.h(range(3))

    circ.cx(0, 4)
    circ.cx(0, 5)

    circ.cx(1, 3)
    circ.cx(1, 5)

    circ.cx(2, 3)
    circ.cx(2, 4)

    circ.barrier()

    for i in range(p):
        circ.append(mixer(beta[i], num_qubit), range(2 * num_qubit))
        circ.append(cost_unitary(gamma[i], num_qubit), range(2 * num_qubit))

    circ.measure(range(num_qubit), range(num_qubit))
    # circ.measure_all()

    return circ


# define a function which calculate the function value for a measurement
def hamming_weight(x, r):
    weight = 0
    for i in range(len(r)):
        if x[i] != r[i]:
            weight = weight + 1
    return weight



# define a function for calculating the average
def compute_expectation(counts, r):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = hamming_weight(reverse(bitstring), r)  # r is |0010>
        avg = avg + (obj * count)
        sum_count = sum_count + count

    expectn = avg / sum_count

    return expectn


def get_expectation(r, p, optm_param):
    simulator = Aer.get_backend('qasm_simulator')
    shots = 1000
    num_qubit = 6

    def execute_circ(theta):
        circ = create_qaoa_circ(optm_param, theta, num_qubit, p, r)
        job = execute(circ, backend=simulator, shots=2000)
        counts = job.result().get_counts()

        return compute_expectation(counts, r)

    return execute_circ


from scipy.optimize import minimize


def main():
    Data = {}
    no = 10
    legend = []
    sim_average = 0.0
    for xxx in range(no):
        data = []
        y_data = []
        layer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Data["p"] = layer
        shots = 2000
        for p in layer:
            w_min = 1.0
            r = '111011'

            optm_param = []

            for k in range(p):
                alpha = []  # function value
                param1 = []
                param2 = []
                approx_ratio = []

                for repeatation in range(5):
                    expectation = get_expectation(r, k+1, optm_param)

                    res = minimize(expectation, [random.uniform(0, np.pi),
                                                 random.uniform(0, np.pi)],
                                   method='COBYLA')  # method='trust-constr'’

                    alpha.append(res.fun)
                    param1.append(res.x[0])
                    param2.append(res.x[1])

                for j in range(len(alpha)):
                    ratio = alpha[j] / w_min
                    approx_ratio.append(ratio)

                min_approx_ratio = min(approx_ratio)
                for j in range(len(alpha)):
                    if approx_ratio[j] == min_approx_ratio:
                        optm_param.append(param1[j])
                        optm_param.append(param2[j])
                # print("Simulation:"+str(xxx)+"For p =" + str(k+1) + "len of optimum param is" + str(len(optm_param)))
            num_qubit = 6

            optm_circ = create_qaoa_circ1(optm_param, num_qubit, p, r)

            simulator = Aer.get_backend('qasm_simulator')

            job = execute(optm_circ, backend=simulator, shots=2000)

            counts = job.result().get_counts()

            data.append(reverse(max(counts, key=lambda x: counts[x])))

            # Target string
            ts = '011011'

            y_data.append(counts[reverse(ts)])

        print("Done Simulation Number" + str(xxx + 1))

        plot_data = []

        for item in y_data:
            plot_data.append(item / shots)

        avg = sum(plot_data) / len(layer)
        sim_average += avg
        plt.plot(layer, plot_data, marker="o")
        Data["Simulation" + str(xxx + 1)] = plot_data
        legend.append("Simulation" + str(xxx + 1))

    sim_average = sim_average / no
    Data["Simulation_avg"] = sim_average

    df = pd.DataFrame(Data)
    df.to_excel("633_fixed_succ_rate_data.xlsx", index=False)

    plt.axhline(y=sim_average, color="darkgoldenrod", linestyle='-')

    legend.append("Average")

    plt.legend(legend, loc="lower right")

    plt.title('Success rate of approximate solution state $|011011>$')

    plt.xlabel('p (No. of Unitary layer)')

    plt.ylabel('Achieved solution rate')

    plt.grid(linestyle='--', axis='y')

    # plt.savefig('633fixed_soln_rate2', dpi=600)

    plt.show()

if __name__ == "__main__":
    main()