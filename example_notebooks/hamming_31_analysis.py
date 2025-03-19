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


def mixer(beta,num_qubit):
    """Mixer unitary for repetition code"""

    codeword = QuantumRegister(num_qubit,'codeword')
    
    recieved = QuantumRegister(num_qubit,'recieved')

    circ = QuantumCircuit(codeword,recieved)

    min_word_matrix = np.loadtxt("data/Hamming_code_[15,11]_min_cwds.txt",dtype=int)#np.loadtxt("data/Hamming_code_[15,11]_min_cwds.txt",dtype=int)
    
    for cwd in min_word_matrix:
        mixer_qubits = np.where(cwd == 1)[0]

        circ.h(mixer_qubits)

        for i in range(len(mixer_qubits)-1):
            circ.cx(mixer_qubits[i],mixer_qubits[i+1])

        circ.rz(-2 * beta, max(mixer_qubits))

        for i in range(len(mixer_qubits) - 1, 0, -1):
            circ.cx(mixer_qubits[i-1], mixer_qubits[i])
    
        circ.h(mixer_qubits)

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

    k = 11

    min_word_matrix = np.loadtxt("data/Hamming_code_[15,11]_min_cwds.txt",dtype=int)
    standard_g = np.loadtxt("data/Hamming_code_[15,11]_standard.txt",dtype=int)
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
    circ.h(range(k))

    for control in range(k):
        for target in range(k,num_qubit):
            if standard_g[control, target] == 1:
                circ.cx(control, target)

        circ.barrier()

    # QAOA layers: cost_unitary + mixer_unitary
    for i in range(p):
        circ.append(cost_unitary(gamma, num_qubit), range(2 * num_qubit))
        circ.append(mixer(beta, num_qubit), range(2 * num_qubit))

    circ.measure(range(num_qubit), range(num_qubit))

    return circ


def get_target_state_counts(expectation, num_qubit, no_layer, received_state, target_reversed, simulator, shots, no_train_samples):
    """Function to optimize QAOA parameters and count occurrences of target state in one simulation instance."""
    # Optimize parameters for QAOA
    results = [
        minimize(expectation, [random.uniform(0, 20.0), random.uniform(0, 50 * np.pi)], method='COBYLA')
        for _ in range(no_train_samples)
    ]

    # Get the best result
    best_result = min(results, key=lambda r: r.fun)

    # Create and run the optimized QAOA circuit
    opt_circuit = create_qaoa_circ(best_result.x, num_qubit, no_layer, received_state)
    job = simulator.run(transpile(opt_circuit, simulator), shots=shots)
    counts = job.result().get_counts()

    # Return occurrences of the target state (default to 0 if not found)
    print(counts.get(target_reversed, 0))
    return counts.get(target_reversed, 0)

if __name__ == "__main__":
    # Constants
    simulator = QasmSimulator()
    shots = 1000
    num_qubit = 15
    no_instances = 6
    no_train_samples = 50
    no_layer = 5
    target_state = "111111111111111"
    received_state = "111111111111110"

    # Precompute the expectation function
    expectation = get_expectation(num_qubit, received_state, no_layer, shots=shots)

    # Precompute reversed target state
    target_reversed = reverse(target_state)

    # Run simulations in parallel
    occurrences = jl.Parallel(n_jobs=-1)(
        jl.delayed(get_target_state_counts)(expectation, num_qubit, no_layer, received_state, target_reversed, simulator, shots, no_train_samples)
        for _ in tqdm(range(no_instances))
    )
    print(occurrences)
    # Plot the results
    plt.bar(range(no_instances), occurrences, color='darkblue')
    plt.title(fr"Distribution of approximate target state $|{target_state}\rangle$")
    plt.xlabel('Simulation Instances')
    plt.ylabel('Number of Occurrences')
    plt.grid(linestyle='--', axis='y')
    plt.ylim(0, shots)
    plt.savefig('test.png', dpi=300)