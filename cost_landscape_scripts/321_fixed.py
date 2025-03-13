from qiskit import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mixer(beta,num_qubit):
    """Mixer unitary"""

    codeword = QuantumRegister(num_qubit,'codeword')
    
    recieved = QuantumRegister(num_qubit,'recieved')
    circ = QuantumCircuit(codeword,recieved)
  
    circ.rx(2*beta,1)

    circ.barrier()
    
    circ.barrier()

    return circ

def cost_unitary(gamma,num_qubit):
    """Cost unitary"""
    codeword = QuantumRegister(num_qubit,'codeword')
    
    recieved = QuantumRegister(num_qubit,'recieved')
    
    circ = QuantumCircuit(codeword,recieved)

    for i in range(num_qubit):
        circ.cx(codeword[i],recieved[i])
        circ.rz((-gamma),recieved[i])
        circ.cx(codeword[i],recieved[i])
        circ.barrier()
    
    return circ

# beta and gamma is a list of length p
# Try for fixed parameter:
def create_qaoa_circ(beta, gamma, num_qubit, p, r):
    """The QAOA circuit of p layers"""

    codeword = QuantumRegister(num_qubit,'codeword')
    recieved = QuantumRegister(num_qubit,'recieved')

    circ = QuantumCircuit(codeword,recieved)

    # if the codeword recieved is r = |0010>, in qiskit the 1 is in the 1st position of recieved register.
    
    for i in range(len(r)):
        if r[i] == '1':
            circ.x(recieved[i])
    
    circ.h(range(2))
    
    circ.cx(0,2)
    
    circ.barrier()
    
    for i in range(p):
        
        circ.append(mixer(beta[i],num_qubit),range(2*num_qubit))
        circ.append(cost_unitary(gamma[i],num_qubit),range(2*num_qubit))

    return circ


# for p = 3. beta = [beta1,beta2,beta3] and gamma = [gamma1,gamma2,gamma3]

def expectation(theta_beta,theta_gamma):
    beta = [10.718126298624355,17.033301639108966,theta_beta]
    gamma = [112.2159171046257,122.4372617692916,theta_gamma]
    
    # No of layers p = 3 in the QAOA circuit
    psi = create_qaoa_circ(beta,gamma,num_qubit = 3,p = 3,r = "011")
    
    H = SparsePauliOp.from_list([
                                    ("IIIIII", 0.5),
                                    ("IIIIII", 0.5),
                                    ("IIIIII", 0.5),
                                    ("IIIIII", 0.5),
                                    ("ZIIZII", -0.5),
                                    ("IZIIZI", -0.5),
                                    ("IIZIIZ", -0.5)
                                ])

    estimator = StatevectorEstimator()
    
    result = estimator.run([(psi,[H])]).result()[0]
    
    return float(result.data.evs[0])

beta = np.arange(0.0,np.pi,0.05)
gamma = np.arange(0.0,2*np.pi,0.05)

x, y = np.meshgrid(beta,gamma)
f = np.vectorize(expectation)

z = f(x,y)

# Find the max length to standardize the size
max_length = max(len(beta), len(gamma), z.shape[0])

# Pad shorter arrays with NaN
beta_padded = np.pad(beta, (0, max_length - len(beta)), constant_values=np.nan)
gamma_padded = np.pad(gamma, (0, max_length - len(gamma)), constant_values=np.nan)
z_padded = np.pad(z, ((0, max_length - z.shape[0]), (0, 0)), constant_values=np.nan)

# Create a DataFrame
df = pd.DataFrame({
    "beta": beta_padded,
    "gamma": gamma_padded,
})

# Add z columns with formatted names
for i in range(z.shape[1]):  
    df[f"exp[beta{i}]"] = z_padded[:, i]

# Save to CSV
df.to_csv("data/land_321_fixed.csv", index=False)


# plot
fig = plt.figure(figsize = (12,12))
ax = plt.axes(projection='3d')


# Creating color map
my_cmap = plt.get_cmap('nipy_spectral')          # 'nipy_spectral' 'hsv' 'turbo' 'hot''ocean' 'rainbow' twilight

surf = ax.plot_surface(x,y,z,cmap = my_cmap,
                       edgecolor ='none',shade = True)


# Set axes label
ax.set_xlabel("$\\beta$",fontsize = 15)
ax.set_ylabel("$\\gamma$",fontsize = 15)
ax.set_zlabel('$<H_f>$',fontsize = 15)
ax.set_title('Fixed',fontsize = 25)
fig.colorbar(surf)

plt.savefig('data/land321_fixed.png',dpi = 300)