# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:14:18 2024

@author: Krishnan Suresh
"""

import numpy as np

from qiskit.quantum_info import Statevector
from IPython.display import display
import matplotlib.pyplot as plt

plt.close('all')

#%% Qubits
ket0 = np.array([1, 0]) # define ket 0
ket1 =  np.array([0, 1]) # define ket 1
display(Statevector(ket0).draw('latex')) # for display, we use Qiskit library

ket00 = np.kron(ket0,ket0)
ket01 = np.kron(ket0,ket1)
ket10 = np.kron(ket1,ket0)
ket11 = np.kron(ket1,ket1)

phi = 1/2*ket00 +   1j/np.sqrt(2)*ket10 +  (np.sqrt(3)+1j)/4*ket11
display(Statevector(phi).draw('latex'))

#%%
import pennylane as qml

#%% Creating a Hadamard circuit
dev=qml.device("default.qubit",wires = 1)
@qml.qnode(dev)
def hadamardCircuit(): 
    qml.Hadamard(wires=0)
    return qml.probs(wires=[0])
qml.draw_mpl(hadamardCircuit)()

#%% Executing the circuit
probabilities = hadamardCircuit(shots = 1000)
print("\nProbabilities for 0 and 1 are:\n",probabilities)

#%%

dev = qml.device("default.qubit")
@qml.qnode(dev)
def RYCircuit(theta): 
    qml.RY(theta,wires=0)
    return qml.probs(wires=[0])
theta = np.pi/3
qml.draw_mpl(RYCircuit)(theta)

#%%
dev=qml.device("default.qubit",wires = 3)
@qml.qnode(dev)
def stateCircuit(): 
	qml.PauliX(wires = 0)
	qml.Identity(wires = 1)
	qml.Hadamard(wires = 2)
	return qml.state()
qml.draw_mpl(stateCircuit)()
phi = stateCircuit()
display(Statevector(phi).draw('latex'))



