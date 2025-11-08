"""Test state preparation in QPE."""
import numpy as np
import cirq

# Test Ry rotation for preparing |0⟩
initial_state = np.array([1.0, 0.0])
norm = np.linalg.norm(initial_state)
normalized = initial_state / norm

# Calculate rotation angle
theta = 2 * np.arctan2(abs(normalized[1]), abs(normalized[0]))
print(f"Initial state: {initial_state}")
print(f"Normalized: {normalized}")
print(f"Rotation angle: θ = {theta} rad = {np.degrees(theta)} deg")

# Test what Ry(θ) produces
q = cirq.LineQubit(0)
circuit = cirq.Circuit()
if abs(theta) > 1e-10:
    circuit.append(cirq.ry(theta)(q))
else:
    print("θ ≈ 0, no rotation needed, state remains |0⟩")

# Simulate
sim = cirq.Simulator()
result = sim.simulate(circuit)
final_state = result.final_state_vector

print(f"\nFinal state: {final_state}")
print(f"Expected: |0⟩ = [1, 0]")
print(f"Match? {np.allclose(final_state, [1, 0])}")

# Now test with |ψ⟩ = [1, 1]/√2 (should give |+⟩)
print("\n" + "="*60)
initial_state2 = np.array([1.0, 1.0]) / np.sqrt(2)
normalized2 = initial_state2 / np.linalg.norm(initial_state2)
theta2 = 2 * np.arctan2(abs(normalized2[1]), abs(normalized2[0]))

print(f"Initial state: [1, 1]/√2")
print(f"Rotation angle: θ = {theta2} rad = {np.degrees(theta2)} deg")

circuit2 = cirq.Circuit()
circuit2.append(cirq.ry(theta2)(q))

result2 = sim.simulate(circuit2)
final_state2 = result2.final_state_vector

print(f"\nFinal state: {final_state2}")
print(f"Expected: |+⟩ = [1/√2, 1/√2] = {np.array([1, 1])/np.sqrt(2)}")
print(f"Match? {np.allclose(final_state2, np.array([1, 1])/np.sqrt(2))}")

# Test H gate  
print("\n" + "="*60)
print("Compare with H gate:")
circuit_H = cirq.Circuit(cirq.H(q))
result_H = sim.simulate(circuit_H)
print(f"H|0⟩ = {result_H.final_state_vector}")
