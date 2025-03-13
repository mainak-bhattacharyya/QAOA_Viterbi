import numpy as np
import random
from qiskit import *
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter

import pandas as pd


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
    circ.h(3)
    circ.h(4)
    circ.h(5)

    circ.cx(0, 1)
    circ.cx(1, 3)
    circ.cx(3, 4)
    circ.cx(4, 5)
    circ.rz(2 * beta, 5)
    circ.cx(4, 5)
    circ.cx(3, 4)
    circ.cx(1, 3)
    circ.cx(0, 1)

    circ.h(5)
    circ.h(4)
    circ.h(3)
    circ.h(1)
    circ.h(0)
    circ.barrier()
    circ.h(2)
    circ.h(3)
    circ.h(5)
    circ.h(6)
    circ.h(7)

    circ.cx(2, 3)
    circ.cx(3, 5)
    circ.cx(5, 6)
    circ.cx(6, 7)
    circ.rz(2 * beta, 7)
    circ.cx(6, 7)
    circ.cx(5, 6)
    circ.cx(3, 5)
    circ.cx(2, 3)

    circ.h(2)
    circ.h(3)
    circ.h(5)
    circ.h(6)
    circ.h(7)
    circ.barrier()
    circ.h(4)
    circ.h(5)
    circ.h(7)
    circ.h(8)
    circ.h(9)

    circ.cx(4, 5)
    circ.cx(5, 7)
    circ.cx(7, 8)
    circ.cx(8, 9)
    circ.rz(2 * beta, 9)
    circ.cx(8, 9)
    circ.cx(7, 8)
    circ.cx(5, 7)
    circ.cx(4, 5)

    circ.h(4)
    circ.h(5)
    circ.h(7)
    circ.h(8)
    circ.h(9)
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


gamma = Parameter("$\\gamma$")

beta = Parameter("$\\beta$")


def create_qaoa_circ(theta, num_qubit, p, r):
    beta = theta[0]
    gamma = theta[1]

    codeword = QuantumRegister(num_qubit, 'codeword')
    recieved = QuantumRegister(num_qubit, 'recieved')
    cbit = ClassicalRegister(num_qubit)
    circ = QuantumCircuit(codeword, recieved, cbit)

    # if the codeword recieved is r = |0010>, in qiskit the 1 is in the 1st position of recieved register.

    for i in range(len(r)):
        if r[i] == '1':
            circ.x(recieved[i])

    circ.x(codeword[0])
    circ.x(codeword[1])
    circ.x(codeword[3])
    circ.x(codeword[4])
    circ.x(codeword[5])

    for i in range(p):
        circ.append(mixer(beta, num_qubit), range(2 * num_qubit))
        circ.append(cost_unitary(gamma, num_qubit), range(2 * num_qubit))

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


# print(hamming_weight('0110','1110'))

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


def get_expectation(p, r):
    simulator = Aer.get_backend('qasm_simulator')
    # shots = 1000

    num_qubit = 10

    def execute_circ(theta):
        #         num_qubits = 6

        #         from qiskit.providers.ibmq import least_busy
        #         possible_devices = provider.backends(filters=lambda x:
        #                                              x.configuration().n_qubits >= num_qubits
        #                                                and
        #                                              x.configuration().simulator == False)
        #         qcomp = least_busy(possible_devices)

        circ = create_qaoa_circ(theta, num_qubit, p, r)
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
            r = '0001110111'
            f_max = 1.0
            optm_param = []

            param_beta = []
            param_gamma = []

            func = []
            approx_ratio = []
            param_1 = []
            param_2 = []
            for repeatation in range(5):
                expectation = get_expectation(p, r)

                res = minimize(expectation,
                               [random.uniform(0, 20.0), random.uniform(0, 50 * np.pi)], method='COBYLA')
                func.append(res.fun)
                param_1.append(res.x[0])
                param_2.append(res.x[1])
            for i in range(len(func)):
                approx_ratio.append(func[i] / f_max)
            min_approx_ratio = min(approx_ratio)

            for i in range(len(func)):
                if approx_ratio[i] == min_approx_ratio:
                    param_beta.append(param_1[i])
                    param_gamma.append(param_2[i])

            num_qubit = 10

            ip = [param_beta[0], param_gamma[0]]

            optm_circ = create_qaoa_circ(ip, num_qubit, p, r)

            simulator = Aer.get_backend('qasm_simulator')

            job = execute(optm_circ, backend=simulator, shots=2000)

            counts = job.result().get_counts()



            data.append(reverse(max(counts, key=lambda x: counts[x])))

            # Target string
            ts = '0000110111'

            y_data.append(counts[reverse(ts)])  # count of target string as layer increases..

            # print("Simulation done for p="+str(p))

            # print('Resulting counts',y_data)

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
    df.to_excel("conv_uni_succ_rate_data.xlsx", index=False)

    print("EXCEL data book is generated!")

    plt.axhline(y=sim_average, color="darkgoldenrod", linestyle='-')

    legend.append("Average")

    plt.legend(legend, loc="lower right")


    plt.xlabel('p (No. of Unitary layer)')

    plt.ylabel('Achieved solution rate')

    plt.grid(linestyle='--', axis='y')


    plt.show()


if __name__ == "__main__":
    main()
