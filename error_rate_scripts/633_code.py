from tqdm import tqdm
import joblib as jl

from qiskit import *
from qiskit import QuantumCircuit

from qiskit_aer import QasmSimulator
from qiskit import transpile 

import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

# distance = num_qubit for repetition code.

def reverse(string):
    """Reverses a string"""
    return string[::-1]

# define a function which calculate the function value for a measurement
def hamming_weight(x, r):
    """Computes the Hamming weight
    Args:
        x: type str.
        r: type str.    
    """
    return sum(xi != ri for xi, ri in zip(x, r))

# define a function for calculating the average
def compute_expectation(counts, r):
    """Computes expectation value
    Args:
        counts: stores vector and their no of occurences (type: dict).
        r: received vector (type: str).
    """
    total = sum(counts.values())
    return sum(hamming_weight(reverse(bitstring), r) * count for bitstring, count in counts.items()) / total

def get_expectation(num_qubit, r, p, shots):

    def execute_circ(theta):
        circ = create_qaoa_circ(theta, num_qubit, p, r)

        simulator = QasmSimulator()

        compiled_circ = transpile(circ, simulator)

        job = simulator.run(compiled_circ, shots=shots)

        counts = job.result().get_counts()

        return compute_expectation(counts, r)

    return execute_circ


def mixer(beta,distance):
    """Mixer unitary for repetition code"""

    num_qubit =distance

    codeword = QuantumRegister(num_qubit,'codeword')
    
    recieved = QuantumRegister(num_qubit,'recieved')

    circ = QuantumCircuit(codeword,recieved)

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

def cost_unitary(gamma,num_qubit):
    """Cost unitary"""

    codeword = QuantumRegister(num_qubit,'codeword')
    recieved = QuantumRegister(num_qubit,'recieved')
    circ = QuantumCircuit(codeword,recieved)
    for i in range(num_qubit):
        circ.cx(codeword[i], recieved[i])
        circ.rz((-gamma), recieved[i])
        circ.cx(codeword[i], recieved[i])
        circ.barrier()
    
    return circ


#theta is the initial parameters [beta,gamma]
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

    # encoding circuit:
    circ.h(range(3))

    circ.cx(0, 4)
    circ.cx(0, 5)

    circ.cx(1, 3)
    circ.cx(1, 5)

    circ.cx(2, 3)
    circ.cx(2, 4)

    circ.barrier()

    # QAOA layers: cost_unitary + mixer_unitary
    for i in range(p):
        circ.append(cost_unitary(gamma, num_qubit), range(2 * num_qubit))
        circ.append(mixer(beta, num_qubit), range(2 * num_qubit))

    circ.measure(range(num_qubit), range(num_qubit))

    return circ

def get_wer(num_qubit, no_layer, p, simulation_shots):
    """
    Args:
        no_layer: QAOA layers.
        p: float, probabilty of error.
    """
    # No of training samples from which the best parameter is sampled from:
    no_train_samples = 5

    errors = np.random.binomial(1, p, (simulation_shots, num_qubit))

    word_error = np.zeros(simulation_shots, dtype=bool)

    simulator = QasmSimulator()

    for i in range(simulation_shots):
        r = ''.join(map(str, errors[i]))
        func_vals, params = [], []
        expectation = get_expectation(num_qubit, r, no_layer, shots=1000)
        for _ in range(no_train_samples):
            res = minimize(expectation, [random.uniform(0, 20.0), random.uniform(0, 50 * np.pi)], method='COBYLA')
            func_vals.append(res.fun)
            params.append(res.x)
        best_idx = np.argmin(func_vals)
        optm_circ = create_qaoa_circ(params[best_idx], num_qubit, no_layer, r)
        job = simulator.run(transpile(optm_circ, simulator), shots=1000)
        counts = job.result().get_counts()
        word_error[i] = reverse(max(counts, key=counts.get)) != '0' * num_qubit
    return np.mean(word_error)


if __name__ == "__main__":
    num_qubit = 6
    noise_prob = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025,
                           0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0])#np.arange(0.001,1.0,0.1)
    no_layer = 3
    # samples to stimate WER
    simulation_shots = 1000
    results = np.zeros((1,len(noise_prob)))

    results[0,:] = jl.Parallel(n_jobs=-1,backend="multiprocessing")(
                        jl.delayed(get_wer)(
                            num_qubit,
                            no_layer,
                            p,
                            simulation_shots
                        )
                        for p in tqdm(noise_prob)
    )

    print(results)

    plt.figure(figsize=(10,6))

    plt.plot(noise_prob, results[0], label="p=3")
    
    plt.legend()

    plt.savefig('data/wer_633.png',dpi = 300)

    plt.show()
