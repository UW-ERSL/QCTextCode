# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:16:44 2024

@author: Krishnan Suresh
"""


#%% Qiskit authentication and test
from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum",
# token="Your API here",set_as_default=True, overwrite=True)


#%% Sample Qiskit code to run on a simulator
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
circuit = QuantumCircuit(1, 1) 
circuit.h(0) # apply H to qubit 0
# measure and place result in classical bit
circuit.measure(0, 0) 
circuit.draw('mpl') 

backend = Aer.get_backend('qasm_simulator')
transpiled_circuit = transpile(circuit, backend)
job = backend.run(transpiled_circuit,shots = 1000)
counts = job.result().get_counts(circuit)
print("Counts:\n",counts)

#%% Run on a real IBM quantum machine
from qiskit_ibm_runtime import SamplerV2 as Sampler
if (0): # Change to 1 to run on a real IBM quantum machine
	service = QiskitRuntimeService()
	backend = service.least_busy(operational=True, simulator=False)
	print(backend)
	circuit = QuantumCircuit(1)
	circuit.h(0)
	circuit.measure_all()
	transpiled_circuit = transpile(circuit, backend)
	sampler = Sampler(backend)
	job = sampler.run([transpiled_circuit],shots = 1000)
	print(f"job id: {job.job_id()}")
	result = job.result()
	print(result)
#%% Modules needed for al examples below
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from IPython.display import display
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate, HamiltonianGate
from qiskit.circuit.library import QFT, PhaseEstimation
from qiskit.circuit.library.standard_gates.u import UGate
import matplotlib.pyplot as plt
import numpy as np

#%% Complex
x = 1 + 3j # note the 3j
print("The real part is: ", x.real)
print("The imaginary part is: ", x.imag)
print("The absolute value is: ", abs(x))

#%% Qubits
ket0 = Statevector([1, 0]) # define ket 0
ket1 = Statevector([0, 1]) # define ket 1
display(ket0.draw('latex'))

ket00 = ket0.tensor(ket0)
ket01 = ket0.tensor(ket1)
ket10 = ket1.tensor(ket0)
ket11 = ket1.tensor(ket1)

phi = 1/2*ket00 +   1j/np.sqrt(2)*ket10 +  (np.sqrt(3)+1j)/4*ket11
display(phi.draw('latex'))

#%% Hadamard Operator, statevector
circuit = QuantumCircuit(1) # 1 qubit
circuit.h(0) # apply H to qubit 0
psi = Statevector(circuit) #extract the state
display(psi.draw('latex')) # print


#%%  Sampling a circuit on a simulator
def simulateCircuit(circuit,nShots=1000):
	backend = Aer.get_backend('qasm_simulator')
	new_circuit = transpile(circuit, backend)
	job = backend.run(new_circuit,shots = nShots)
	counts = job.result().get_counts(circuit)
	return counts

#%% Hadamard Operator, measure
# 1 qubit, 1 classical bit
circuit = QuantumCircuit(1, 1) 
circuit.h(0) # apply H to qubit 0
# measure and place result in classical bit
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% RX Operator
#1 qubit, 1 classical bit
circuit = QuantumCircuit(1, 1)  
circuit.rx(np.pi/3,0) # apply Rx to qubit 0
#  measure and place result in classical bit
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% State preparation 
circuit = QuantumCircuit(1, 1)  
q = Statevector([np.sqrt(8)/3, (1j)/3]) 
circuit.prepare_state(q,0,'Prepare q')
circuit.x(0) 
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Multi-qubit circuit
circuit = QuantumCircuit(3, 3)  
circuit.x(0)
circuit.id(1)
circuit.h(2)
circuit.measure([0,1,2], [0,1,2]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Multi-qubit circuit, theoretical state
circuit = QuantumCircuit(3, 3)  
circuit.x(0)
circuit.id(1)
circuit.h(2)
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% Unitary operator
circuit = QuantumCircuit(1, 1) 
UMatrix = 1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]) 
circuit.unitary(UMatrix,0,'myU')
circuit.measure(0,0) 
# To see the theoretical state, comment the previous line and uncomment next 2 lines
# psi = Statevector(circuit)
# display(psi.draw('latex'))
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% Universal operator
def UniversalOperator(theta,phi,lambdaAngle):
	U = np.array([[np.cos(theta/2),-np.exp(1j*lambdaAngle)*np.sin(theta/2)],
			      [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lambdaAngle))*np.cos(theta/2)]])
	return U

U = UniversalOperator(np.pi/2,np.pi,np.pi)
print(U)
U = UniversalOperator(np.pi,0,0)
print(U)


#%% Simple CNOT
circuit = QuantumCircuit(2, 2)  
circuit.x(1) # try id(1), h(1)
circuit.cx(1,0)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Simple CH
circuit = QuantumCircuit(2, 2)  
circuit.id(1) # try id(1), h(1)
circuit.ch(1,0)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% CNOT and controlled Hadamard
circuit = QuantumCircuit(3, 3) 
circuit.y(0) 
circuit.rx(np.pi/3,1) 
circuit.h(2) 
circuit.cx(2,0)
circuit.ch(2,1)
circuit.barrier()
psi = Statevector(circuit)
display(psi.draw('latex'))
circuit.measure([0,1,2], [0,1,2]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,10000)
print('Counts:',counts)
plot_histogram(counts)


#%% cp
circuit = QuantumCircuit(2, 2)  
circuit.x(0) 
circuit.x(1) 
circuit.cp(-np.pi/2,0,1)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% Simple swap  
circuit = QuantumCircuit(2, 2)  
circuit.x(1) 
circuit.swap(1,0)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% controlled Unitary
circuit = QuantumCircuit(2, 2) 
UMatrix = 1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]) 
U = UnitaryGate(UMatrix,'myU')
UControl = U.control(1)
circuit.append(UControl,[1,0])
circuit.measure([0,1],[0,1]) 
circuit.draw('mpl') 


#%% state controlled U Gate
circuit = QuantumCircuit(4, 1) 
cu_gate = UGate(np.pi, 0, 0).control(3, ctrl_state = '101')
# Not sure why the following does not work
# circuit.cry(theta = np.pi/3,target_qubit= 0,
#      control_qubit = [1,2,3], ctrl_state = '101')
circuit.append(cu_gate,[1,2,3,0])
circuit.measure([0],[0]) 
circuit.draw('mpl') 


#%% continuous signal
def continuousSignal(t):
	s = 0.25 + 0.5*np.sin(2*2*np.pi*t) - 0.3*np.cos(5*2*np.pi*t)
	return s

nContinuousSamples = 1000 # for plotting
t = np.zeros((nContinuousSamples,1))
s = np.zeros((nContinuousSamples,1))
for i in range(nContinuousSamples):
	t[i] = i/nContinuousSamples
	s[i]= continuousSignal(t[i])

plt.close('all')
plt.plot(t,s)
plt.axhline(0, color='black')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Signal', fontsize=14)
plt.grid(visible=True)

#%% discrete signal
M = 32
t = np.zeros(M)
y = np.zeros(M)
for i in range(M):
	t[i] = i/M
	y[i]= continuousSignal(t[i])
plt.figure()
plt.axhline(0, color='black')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Signal', fontsize=14)
plt.grid(visible=True)
plt.plot(t,y,'*')


#%% DFT operation
def createDFTMatrix(M):
    DFTMatrix = np.zeros((M,M), dtype=complex)
    omega = np.exp(1j*(2*np.pi/M))
    for i in range(M):
        for j in range(M):
            DFTMatrix[i][j] = omega**(-i*j)
    return DFTMatrix 
def processDFTResult(phi):
	cosineTerms = (phi[1:int(M/2)]).real+(phi[M-1:int(M/2):-1]).real;
	cosineTerms = np.insert(cosineTerms, 0,phi[0].real)
	sineTerms =  -(((phi[1:int(M/2)])-(phi[M-1:int(M/2):-1]))/(1j)).real;
	sineTerms = np.insert(sineTerms, 0,0)
	return [cosineTerms/M, sineTerms/M]

DFTMatrix = createDFTMatrix(M)

phi = np.matmul(DFTMatrix,y)
[cosineTerms, sineTerms] = processDFTResult(phi)
plt.figure()
plt.bar(list(range(0,int(M/2))),cosineTerms, label ='cos')
plt.bar(list(range(0,int(M/2))),sineTerms, label ='sin')
plt.legend( fontsize=14)
plt.axhline(0, color='black')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)


#%% QFT-2 
circuit = QuantumCircuit(2, 2)  
circuit.h(1)
circuit.cp(np.pi/2,0,1) 
circuit.h(0) 
circuit.swap(0,1)
#np.set_printoptions(precision =3,suppress=True)
print(np.array(Operator(circuit).data))
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% QFT-8
n = 3 # number of qubits
circuit = QuantumCircuit(n, n)  
qft = QFT(num_qubits=n).to_gate()
circuit.append(qft, qargs=list(range(n)))
circuit.measure(list(range(n)),list(range(n))) 
circuit.decompose(reps=2).draw('mpl') 


#%% QFT-8
n = 3 # number of qubits
circuit = QuantumCircuit(n, n) 
circuit.x(0)
circuit.x(1)
circuit.x(2)
qft = QFT(num_qubits=n).to_gate()
circuit.append(qft, qargs=list(range(n)))
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% IQFT-8
n = 3 # number of qubits
circuit = QuantumCircuit(n, n) 
iqft = QFT(num_qubits=n,inverse=True).to_gate()
iqft._name = 'IQFT'
circuit.append(iqft, qargs=list(range(n)))
circuit.draw('mpl') 
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% Hamiltonian
A = np.array([[2,-1],[-1,2]])
f = 0.5
lambdaHat = 3
t = -2*np.pi*f/lambdaHat #Note negative
U_A = HamiltonianGate(A, time=t,label = 'UA')
print(np.array(U_A.to_matrix()))
v = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
circuit = QuantumCircuit(1)
circuit.prepare_state(Statevector(v) ,0,'Prepare v')
circuit.append(U_A, qargs=[0])
circuit.draw('mpl') 
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% Single digit QPE with single qubit v
def myQPE1(A,v,f,lambdaHat):
	circuit = QuantumCircuit(2,1)
	circuit.h(0)
	circuit.prepare_state(Statevector(v),[1],' v')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1)
	circuit.append(UControl,[0,1])
	iqft = QFT(num_qubits=1,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [0])
	circuit.measure([0], [0]) 
	circuit.draw('mpl') 
	counts = simulateCircuit(circuit,nShots=1)
	return counts

#%% Single digit QPE with multiple qubits v
def myQPE2(A,v,f,lambdaHat):
	n = int(np.log2(v.shape[0]))
	circuit = QuantumCircuit(n+1,1)
	circuit.h(0)
	circuit.prepare_state(Statevector(v),[*range(1, n+1)],'v')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1) # only 1 control qubit
	circuit.append(UControl,[*range(0, n+1)])
	iqft = QFT(num_qubits=1,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [0])
	circuit.measure([0], [0]) 
	return simulateCircuit(circuit,nShots=1)


#%% Multiple digit QPE with multiple qubits v
def myQPE3(A,v,f,lambdaHat,m=1):
	N = v.shape[0]
	n = int(np.log2(N))
	phase_qubits = QuantumRegister(m, '\u03B8')
	input_qubits = QuantumRegister(n, 'b')
	phase_measurements = ClassicalRegister(m, '\u0398')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	for i in range(m):
		circuit.h(i)
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'b')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	for i in range(m):
		U_A_pow = U_A.power(2**i) 
		UControl = U_A_pow.control(1) # only 1 control qubit
		circuit.append(UControl,[i,*range(m, n+m)])
	iqft = QFT(num_qubits=m,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [*range(0,m)])
	circuit.measure([*range(0,m)], [*range(0,m)]) 
	circuit.draw('mpl') 
	counts = simulateCircuit(circuit,nShots=50)
	return counts

#%% Utility function processCounts for QPE
def processCounts(counts):
	# Input:  counts from circuit simulation 
	# Return: decimal values sorted by descending probability
	# First sort descending using 2nd item in dictionary

	countsSorted = sorted(counts.items(),
		key=lambda item: item[1],reverse=True)
	m = len(countsSorted[0][0]) # length of bit string
	values = []
	for i in range(len(countsSorted)):
		string = countsSorted[i][0]
		values.append(int(string, 2)/(2**m))
	return np.array(values)


#%% Test cases for QPE
plt.close('all')
example = 2
if (example == 1):
	A = np.array([[1,0],[0,0.75]])
	v0 = np.array([1,0])
	v1 = np.array([0,1])
	a0 = 1/2
	a1 = np.sqrt(3)/2
	v = a0*v0 +  a1*v1
	f = 0.5
	lambdaHat = 1
	m = 2
	counts = myQPE3(A,v,f,lambdaHat,m=m)
elif (example == 2):
	A = np.array([[2,-1],[-1,2]])
	v0 = np.array([1/np.sqrt(2),1/np.sqrt(2)])
	v1 = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
	a0 = 1
	a1 = 0
	v = a0*v0 +  a1*v1
	f = 0.5
	lambdaHat = 3
	m = 10
	counts = myQPE3(A,v,f,lambdaHat,m=m)
elif (example == 3):
	A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
	v0 = np.array([1/np.sqrt(2),0,0,-1/np.sqrt(2)])
	v1 = np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)])
	v2 = np.array([0,1,0,0])
	v3 = np.array([0,0,1,0])
	a = [1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4)]
	#a = [1,0,0,0]
	v = a[0]*v0 + a[1]*v1 + a[2]*v2 + a[3]*v3
	f = 0.5
	lambdaHat = 1.5
	m = 2
	counts = myQPE3(A,v,f,lambdaHat,m=m)
elif (example == 4):
	A = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
	v = np.random.rand(4)
	v = v/np.linalg.norm(v)
	f = 0.5
	lambdaHat = 4
	m = 2
	counts = myQPE3(A,v,f,lambdaHat,m=m)
	
print("counts:", counts)
thetaTilde = processCounts(counts)
print("thetaTilde:", thetaTilde)
print("EigenvalueTilde:",thetaTilde*lambdaHat/f)

#%% Using the Built in QPE
m = 3
A = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
v = np.random.rand(4)
v = v/np.linalg.norm(v)
f = 0.5
lambdaHat = 4

t = -2*np.pi*f/lambdaHat #Note negative
U_A = HamiltonianGate(A, time=t,label = 'UA')
iqft = QFT(num_qubits=m,inverse=True).to_gate()
iqft._name = 'IQFT'
qpe = PhaseEstimation(m,U_A,iqft)

#%%  Controlled Rotation
circuit = QuantumCircuit(2,1)
theta = np.pi/3
circuit.cry(theta,1, 0)
circuit.draw('mpl') 

#%%

def myQPECircuit(A,b,f,lambdaHat,m):
	N = A.shape[0]
	n = int(np.log2(N))
	phase_qubits = QuantumRegister(m, '\u03B8')
	b_qubits = QuantumRegister(n, 'b')
	ancilla_qubit = QuantumRegister(1,'a')
	cl = ClassicalRegister(m,'cl')
	offset = 1
	circuit = QuantumCircuit(ancilla_qubit, phase_qubits, b_qubits,cl)
		
	for i in range(m):
		circuit.h(offset+i)
	if (len(b) != 0):
		circuit.prepare_state(Statevector(b),[*range(m+offset, n+m+offset)],'b')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	for i in range(m):
		U_A_pow = U_A.power(2**i) 
		UControl = U_A_pow.control(1) # only 1 control qubit
		circuit.append(UControl,[i+offset,*range(m+offset, n+m+offset)])
	iqft = QFT(num_qubits=m,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [*range(offset,m+offset)])
	return circuit
	

#%% QPE with Controlled Rotation
def controlledRotationQPE(A,v,f,lambdaHat,m):
	N = v.shape[0]
	n = int(np.log2(N))
	# construct  QPE circuit with controlled Rotation
	qpeCircuit = myQPECircuit(A,v,f,lambdaHat,m)
	qpeCircuit.measure([*range(1,m+1)], [*range(0,m)]) 

	counts = simulateCircuit(qpeCircuit,nShots=50)
	countsSorted = sorted(counts.items(),
		key=lambda item: item[1],reverse=True)

	thetaTilde = processCounts(counts)[0:N]
	lambdaTilde = thetaTilde*lambdaHat/f
	print(thetaTilde)
	C = 0.99*min(lambdaTilde)
	print(C)
	HHLCircuit = myQPECircuit(A,v,f,lambdaHat,m)
	HHLCircuit.barrier()
	
	nControlledRotations = min(len(countsSorted),N)
	for i in range(nControlledRotations):
		string = countsSorted[i][0] 
		theta = np.pi/(i+1)
		cu_gate = UGate(theta, 0, 0).control(m, ctrl_state = string) 
		HHLCircuit.append(cu_gate,[*range(1,1+m),0])
	
	qpeInverse = myQPECircuit(A,[],f,lambdaHat,m).inverse()
	qpeInverse._name = 'IQPE'
	qpeInverse.draw('mpl')
	HHLCircuit.barrier()
	HHLCircuit.compose(qpeInverse,[*range(0,1+m+n)], [*range(0,m)],inplace = True)
	HHLCircuit.draw('mpl')
	
	return 

#%%
plt.close('all')
m=2
counts = controlledRotationQPE(A,v,f,lambdaHat,m)