import numpy as np
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.opflow import Z, StateFn, PauliExpectation, CircuitSampler
from GPyOpt.methods import BayesianOptimization

# Define a quantum circuit for a simple molecule model (e.g., H2)
def build_quantum_circuit(params):
    qc = QuantumCircuit(2)
    ansatz = RealAmplitudes(num_qubits=2, reps=1, entanglement='linear')
    qc.compose(ansatz, inplace=True)
    qc = qc.bind_parameters(params)
    return qc

# Define the target function to optimize
def target_function(x):
    params = x[0]      
    qc = build_quantum_circuit(params)      
    observable = Z ^ Z    
    backend = Aer.get_backend('statevector_simulator')
    qi = QuantumInstance(backend=backend)
    exp_val = PauliExpectation().convert(~StateFn(observable) @ StateFn(qc))
    sampler = CircuitSampler(qi).convert(exp_val)
    result = sampler.eval().real
    return -result

# Define the bounds for the parameters to optimize
bounds = [{'name': 'x', 'type': 'continuous', 'domain': (0, np.pi)} for _ in range(2)]

# Perform Bayesian Optimization
optimizer = BayesianOptimization(f=target_function, domain=bounds)
optimizer.run_optimization(max_iter=50)

# Output the optimal parameters
optimal_params = optimizer.X[np.argmin(optimizer.Y)]
optimal_value = np.min(optimizer.Y)

print(f"Optimal Parameters: {optimal_params}")
print(f"Optimal Value: {optimal_value}")
