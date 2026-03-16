---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What additional options are available when using PT?

Pauli Twirling in Mitiq targets CNOT and CZ gates via
{func}`.generate_pauli_twirl_variants`. This section covers:

- Which noise channels are tailored by PT (with PTM heatmaps)
- How the number of twirled circuit variants affects the outcome
- Stacking PT with a quantum error mitigation technique
- Adding custom noise channels to twirled circuits with {func}`.add_noise_to_two_qubit_gates`

## Which noise channels does PT tailor?

PT transforms an arbitrary noise channel acting on a two-qubit gate into a
stochastic Pauli channel by averaging over random Pauli conjugations.
The Pauli Transfer Matrix (PTM) provides a convenient way to visualize
this transformation: off-diagonal PTM elements encode coherent errors,
while diagonal elements encode incoherent (Pauli) errors.

The table below summarizes the effect of PT on common noise channels.

| Noise channel | Before PT | After PT | Effect of twirling |
|---|---|---|---|
| Coherent (e.g. $R_y$ over-rotation) | Off-diagonal PTM elements present | Diagonal PTM (Pauli channel) | Off-diagonal terms average to zero |
| Depolarizing | Already diagonal PTM | Diagonal PTM (unchanged) | No change --- already a Pauli channel |
| Bit-flip | Already diagonal PTM | Diagonal PTM (unchanged) | No change --- already a Pauli channel |
| Amplitude damping | Non-unital, off-diagonal elements | Closer to diagonal after averaging | Partially tailored; non-unital component remains |

The following code generates PTM heatmaps for a CNOT gate under these
noise channels, before and after Pauli Twirling.

```{code-cell} ipython3
import cirq
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from cirq import LineQubit, Circuit, CNOT, Ry, depolarize, bit_flip, amplitude_damp
from itertools import product
from functools import reduce

from mitiq.pec.channels import _circuit_to_choi, choi_to_super
from mitiq.utils import matrix_to_vector, vector_to_matrix
from mitiq.pt import generate_pauli_twirl_variants

Pauli_unitary_list = [
    cirq.unitary(cirq.I),
    cirq.unitary(cirq.X),
    cirq.unitary(cirq.Y),
    cirq.unitary(cirq.Z),
]


def n_qubit_paulis(num_qubits: int) -> list[npt.NDArray[np.complex64]]:
    """Get a list of n-qubit Pauli unitaries."""
    return [
        reduce(lambda a, b: np.kron(a, b), combo)
        for combo in product(Pauli_unitary_list, repeat=num_qubits)
    ]


def ptm_matrix(circuit: Circuit, num_qubits: int) -> npt.NDArray[np.complex64]:
    """Compute the Pauli Transfer Matrix (PTM) of a circuit."""
    superop = choi_to_super(_circuit_to_choi(circuit))
    paulis = n_qubit_paulis(num_qubits)
    vec_paulis = []
    for p in paulis:
        vec_paulis.append(matrix_to_vector(np.transpose(p)))

    d = 4**num_qubits
    ptm = np.zeros([d, d], dtype=complex)
    for i in range(d):
        sp = np.matmul(superop, vec_paulis[i])
        sp_mat = np.transpose(vector_to_matrix(sp))
        for j in range(d):
            ptm[j, i] = (0.5**num_qubits) * np.trace(
                np.matmul(paulis[j], sp_mat)
            )
    return ptm
```

```{code-cell} ipython3
q0, q1 = LineQubit.range(2)
cnot_circuit = Circuit(CNOT(q0, q1))

noise_configs = [
    ("Coherent ($R_y$)", Ry(rads=np.pi / 8)),
    ("Depolarizing", depolarize(p=0.1)),
    ("Bit-flip", bit_flip(p=0.1)),
    ("Amplitude damping", amplitude_damp(gamma=0.1)),
]

fig, axes = plt.subplots(len(noise_configs), 3, figsize=(14, 16))
cbar_kw = dict(vmin=-1, vmax=1, cmap="PiYG")

for row, (label, noise) in enumerate(noise_configs):
    # Ideal CNOT
    ptm_ideal = ptm_matrix(cnot_circuit, 2)
    sns.heatmap(ptm_ideal.real, ax=axes[row, 0], linewidth=0.3, **cbar_kw, cbar=False)
    axes[row, 0].set_title(f"Ideal CNOT" if row == 0 else "")
    axes[row, 0].set_ylabel(label, fontsize=11)

    # Noisy CNOT (before PT)
    noisy_circuit = cnot_circuit.with_noise(noise)
    ptm_noisy = ptm_matrix(noisy_circuit, 2)
    sns.heatmap(ptm_noisy.real, ax=axes[row, 1], linewidth=0.3, **cbar_kw, cbar=False)
    axes[row, 1].set_title("Before PT" if row == 0 else "")

    # Noisy CNOT (after PT averaging over 100 twirled circuits)
    twirled = generate_pauli_twirl_variants(cnot_circuit, num_circuits=100)
    noisy_twirled_ptms = []
    for tc in twirled:
        noisy_tc = tc.with_noise(noise)
        noisy_twirled_ptms.append(ptm_matrix(noisy_tc, 2).real)
    avg_ptm = np.mean(noisy_twirled_ptms, axis=0)
    sns.heatmap(avg_ptm, ax=axes[row, 2], linewidth=0.3, **cbar_kw, cbar=False)
    axes[row, 2].set_title("After PT (avg. 100)" if row == 0 else "")

fig.suptitle("PTM heatmaps: effect of PT on different noise channels", fontsize=14, y=1.01)
fig.tight_layout()
plt.show()
```

The key observation: for coherent noise (top row), PT eliminates the
off-diagonal PTM elements, converting the noise to a diagonal Pauli channel.
For noise that is already a Pauli channel (depolarizing, bit-flip),
PT has no effect. For amplitude damping, the non-unital diagonal
structure is partially preserved, but off-diagonal coherent components
are suppressed.

## How does the number of twirled circuits affect the outcome?

The `num_circuits` parameter in {func}`.generate_pauli_twirl_variants`
controls how many independently twirled circuit variants are generated.
The averaged expectation value over these variants converges to the
Pauli-twirled result. With too few circuits, the off-diagonal PTM
components are only partially averaged out, and residual coherent
errors remain.

The following example demonstrates this convergence.

```{code-cell} ipython3
from numpy import pi
from cirq import CircuitOperation, CXPowGate, CZPowGate, DensityMatrixSimulator, Rx, H
from cirq.devices.noise_model import GateSubstitutionNoiseModel
from functools import partial
from mitiq import Executor

q0, q1, q2 = LineQubit.range(3)
circuit = Circuit(
    H(q0),
    CNOT.on(q0, q1),
    cirq.CZ.on(q1, q2),
    CNOT.on(q0, q2),
)


def get_noise_model(noise_level: float) -> GateSubstitutionNoiseModel:
    rads = pi / 2 * noise_level
    def noisy_gate(op):
        if isinstance(op.gate, (CXPowGate, CZPowGate)):
            return CircuitOperation(
                Circuit(
                    op.gate.on(*op.qubits),
                    Rx(rads=rads).on_each(op.qubits),
                ).freeze())
        return op
    return GateSubstitutionNoiseModel(noisy_gate)


def execute(circuit: Circuit, noise_level: float = 0.15):
    return (
        DensityMatrixSimulator(noise=get_noise_model(noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )


ideal_value = execute(circuit, noise_level=0.0)
noisy_value = execute(circuit, noise_level=0.15)

num_circuits_list = [1, 2, 5, 10, 25, 50, 100, 200]
errors = []

for n in num_circuits_list:
    twirled = generate_pauli_twirl_variants(circuit, num_circuits=n)
    vals = Executor(partial(execute, noise_level=0.15)).evaluate(twirled)
    avg = np.average(vals)
    errors.append(abs(ideal_value - avg))

plt.figure(figsize=(8, 4))
plt.plot(num_circuits_list, errors, "o-", color="#2ca02c")
plt.axhline(y=abs(ideal_value - noisy_value), color="#1f77b4", linestyle="--",
            label="Error without PT")
plt.xlabel("Number of twirled circuit variants")
plt.ylabel("Absolute error |ideal - result|")
plt.title("Convergence of PT with increasing number of twirled circuits")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

```{note}
The exact convergence depends on the circuit, noise model, and noise strength.
For strong coherent noise, more twirled circuits are needed to average out the
off-diagonal PTM components. As a rule of thumb, 50--100 variants is
a reasonable starting point for most applications.
```

## Stacking PT with a quantum error mitigation technique

PT is a noise *tailoring* technique, not a noise *reduction* technique.
Its primary value comes from composing it with a dedicated quantum error
mitigation (QEM) method that benefits from structured Pauli noise.

The general stacking workflow is:

```{figure} ../img/pt_qem_workflow.svg
---
width: 800px
name: pt-qem-workflow
---
Workflow of PT combined with a QEM technique (e.g. ZNE). PT tailors the noise,
then the QEM method extrapolates or cancels the tailored noise.
```

### Example: PT + ZNE

[Zero-Noise Extrapolation](zne.md) (ZNE) amplifies noise by scaling
the circuit and extrapolates to the zero-noise limit. ZNE performs
best when the noise is stochastic, because unitary folding amplifies
Pauli noise more predictably than coherent noise.

The correct order for combining PT with ZNE is:

1. Generate noise-scaled circuits via {func}`.zne.construct_circuits`.
2. For each noise-scaled circuit, generate PT variants via
   {func}`.generate_pauli_twirl_variants`.
3. Execute all variants and average over the PT variants for each scale factor.
4. Extrapolate to zero noise using the averaged values.

This ensures the twirling gates are **not** amplified by the ZNE folding.

```{code-cell} ipython3
from mitiq import zne

scale_factors = [1, 3, 5]
noise_scaled_circuits = zne.construct_circuits(circuit, scale_factors)

NUM_PT_VARIANTS = 50
noisy_executor = Executor(partial(execute, noise_level=0.15))

noise_scaled_expvals = []
for nsc in noise_scaled_circuits:
    pt_variants = generate_pauli_twirl_variants(nsc, num_circuits=NUM_PT_VARIANTS)
    noise_scaled_expvals.append(np.average(noisy_executor.evaluate(pt_variants)))

extrapolation = zne.inference.RichardsonFactory(scale_factors=scale_factors).extrapolate
pt_zne_result = zne.combine_results(scale_factors, noise_scaled_expvals, extrapolation)

zne_only = zne.execute_with_zne(circuit, partial(execute, noise_level=0.15))

pt_variants = generate_pauli_twirl_variants(circuit, num_circuits=NUM_PT_VARIANTS)
pt_only = np.average(noisy_executor.evaluate(pt_variants))

print(f"Error (no mitigation):  {abs(ideal_value - noisy_value):.4f}")
print(f"Error (ZNE only):       {abs(ideal_value - zne_only):.4f}")
print(f"Error (PT only):        {abs(ideal_value - pt_only):.4f}")
print(f"Error (PT + ZNE):       {abs(ideal_value - pt_zne_result):.4f}")
```

For a more detailed walkthrough, see the
[ZNE with Pauli Twirling example](../examples/pt_zne.md).

## Adding noise to twirled circuits with `add_noise_to_two_qubit_gates`

The function {func}`.add_noise_to_two_qubit_gates` inserts a noise
channel after every CNOT and CZ gate in a twirled circuit. This is
useful for simulating the effect of PT on circuits with specific noise
models.

Two noise channels are built in: `"bit-flip"` and `"depolarize"`.

```{code-cell} ipython3
from mitiq.pt.pt import add_noise_to_two_qubit_gates

twirled = generate_pauli_twirl_variants(cnot_circuit, num_circuits=1)[0]

noisy_twirled = add_noise_to_two_qubit_gates(twirled, "depolarize", p=0.05)
print("Twirled circuit with depolarizing noise:\n")
print(noisy_twirled)
```

### Extending the noise dictionary

To add a custom noise channel, update the `CIRQ_NOISE_OP` dictionary
before calling {func}`.generate_pauli_twirl_variants` with a `noise_name`.

```{code-cell} ipython3
from mitiq.pt.pt import CIRQ_NOISE_OP

CIRQ_NOISE_OP["phase-flip"] = cirq.phase_flip

noisy_twirled_custom = add_noise_to_two_qubit_gates(
    twirled, "phase-flip", p=0.03
)
print("Twirled circuit with custom phase-flip noise:\n")
print(noisy_twirled_custom)
```

The same approach works for generating variants with built-in noise
via the `noise_name` argument:

```{code-cell} ipython3
twirled_with_noise = generate_pauli_twirl_variants(
    cnot_circuit, num_circuits=3, noise_name="phase-flip", p=0.03
)
print(f"Generated {len(twirled_with_noise)} twirled circuits with phase-flip noise.")
print(f"\nExample:\n{twirled_with_noise[0]}")
```

```{tip}
Any callable with signature `(float) -> cirq.Gate` can be added to the
dictionary. This covers any single-parameter Cirq noise channel.
```
