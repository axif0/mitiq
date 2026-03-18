---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is the theory behind Pauli Twirling?

Pauli Twirling (PT) {cite}`Wallman_2016_PRA, Hashim_2021_PRX, Urbanek_2021_PRL, Saki_2023_arxiv`
is a quantum noise tailoring technique that transforms an arbitrary noise
channel into a stochastic Pauli channel by randomly conjugating operations
with Pauli operators and averaging over the results.

## Overview

1. PT is a noise-agnostic tailoring technique designed to be composed with
   more direct mitigation methods.

2. For Markovian noise, PT symmetrizes the noise channel
   (analogous to dynamical decoupling {cite}`Viola_1998_PRA, Viola_1999_PRL, Zhang_2014_PRL`).

3. The success of PT depends on the noise characteristics and the quantum
   system. While PT generally simplifies the noise channel, it can in
   some cases produce a completely depolarizing channel with total loss of
   quantum information.

In the context of quantum error mitigation, PT is closer to [DDD](ddd-5-theory.md)
but stands apart as a noise tailoring technique. PT's distinguishing characteristics:

- It does not reduce noise on its own, but tailors the noise so that downstream
  techniques can mitigate it more effectively.

- It generates multiple circuits with random Pauli modifications and averages
  over their results. There is no need for linear combinations of noisy results
  as in [ZNE](zne-5-theory.md), [PEC](pec-5-theory.md), and [CDR](cdr-5-theory.md).

- The error mitigation overhead is minimal: there is no increase in statistical
  uncertainty in the final result beyond the standard sampling variance.

## Mathematical formulation

### The Pauli twirl of a channel

Let $\mathcal{P}_n = \{I, X, Y, Z\}^{\otimes n}$ be the $n$-qubit Pauli group
(with $|\mathcal{P}_n| = 4^n$ elements, ignoring phases). Given an arbitrary
CPTP map $\mathcal{E}$ with Kraus representation

$$
\mathcal{E}(\rho) = \sum_j K_j \rho K_j^\dagger,
$$

the **Pauli twirl** of $\mathcal{E}$ is defined as

$$
\mathcal{T}_P[\mathcal{E}](\rho)
= \frac{1}{4^n} \sum_{\sigma \in \mathcal{P}_n}
  \sigma \,\mathcal{E}(\sigma \rho \sigma^\dagger)\, \sigma^\dagger.
$$(pauli_twirl_def)

### Key theorem: twirling produces a Pauli channel

For any CPTP map $\mathcal{E}$, the Pauli twirl $\mathcal{T}_P[\mathcal{E}]$
is a **Pauli channel**:

$$
\mathcal{T}_P[\mathcal{E}](\rho) = \sum_k p_k \, \sigma_k \rho \sigma_k^\dagger,
$$(pauli_channel_result)

where the probabilities are

$$
p_k = \frac{1}{4^n} \sum_j \left| \mathrm{Tr}(\sigma_k K_j) \right|^2.
$$

This follows from the Schur orthogonality relation for the Pauli group acting by conjugation
{cite}`Wallman_2016_PRA`.

### Gate twirling (dressed twirling)

Directly applying Eq. {math:numref}`pauli_twirl_def` to a noisy gate
$\mathcal{E} = \mathcal{N} \circ U$ would destroy the ideal unitary $U$.
Instead, Mitiq implements **dressed twirling**: for each Pauli $P_i$,
find $P_i'$ such that $U P_i = P_i' U$ (up to a global phase). The
twirled circuit is then

$$
P_i' \cdot U \cdot P_i = U \quad \text{(ideal effect preserved)},
$$

while the noise $\mathcal{N}$ gets twirled. For Clifford gates (CNOT, CZ),
the conjugation $U P_i U^\dagger$ always yields another Pauli, making this
straightforward. The valid twirl pairs $(P, Q, R, S)$ for CNOT and CZ gates
are stored in {attr}`mitiq.pt.pt.CNOT_twirling_gates` and
{attr}`mitiq.pt.pt.CZ_twirling_gates`.

Each two-qubit gate is sandwiched by a randomly sampled Pauli pair $(P, Q)$
before and a corresponding pair $(R, S)$ after, chosen so that the ideal
gate is preserved and only the noise gets twirled. The valid pairs
$(P, Q, R, S)$ for CNOT and CZ gates are stored in
{attr}`mitiq.pt.pt.CNOT_twirling_gates` and {attr}`mitiq.pt.pt.CZ_twirling_gates`.

## Pauli Transfer Matrix representation

### Definition

For an $n$-qubit channel $\mathcal{E}$, the Pauli Transfer Matrix (PTM)
$\Lambda$ has entries

$$
\Lambda_{ij} = \frac{1}{2^n} \mathrm{Tr}\!\left[ P_i \,\mathcal{E}(P_j) \right],
$$(ptm_definition)

where $P_i, P_j$ are $n$-qubit Pauli operators. All PTM entries are real
and lie in $[-1, 1]$.

### Effect of Pauli twirling on the PTM

**Before twirling:** An arbitrary noise channel has a general PTM with
off-diagonal elements. These off-diagonal entries encode coherent errors ---
unitary rotations that mix Pauli components.

**After twirling:** The PTM becomes **diagonal**:

$$
\Lambda_{\text{twirled}} = \mathrm{diag}(1, \lambda_1, \lambda_2, \ldots, \lambda_{4^n - 1}),
$$

where each $\lambda_k$ relates to the Pauli channel probabilities $p_k$.
The off-diagonal elements vanish because the twirl average enforces
$\Lambda_{ij} = 0$ for $i \neq j$ via the same Schur orthogonality argument.

### Visualizing the PTM with heatmaps

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

### Convergence with number of twirled circuits

The `num_circuits` parameter controls how many independently twirled variants
are generated and averaged. With too few circuits, the off-diagonal PTM
components are only partially averaged out, and residual coherent errors
remain. The following example shows this convergence.

```{code-cell} ipython3
from numpy import pi
from cirq import CircuitOperation, CXPowGate, CZPowGate, DensityMatrixSimulator, Rx, H
from cirq.devices.noise_model import GateSubstitutionNoiseModel
from functools import partial, reduce
from mitiq import Executor

q0, q1, q2 = LineQubit.range(3)
conv_circuit = Circuit(
    H(q0),
    CNOT.on(q0, q1),
    cirq.CZ.on(q1, q2),
    CNOT.on(q0, q2),
)


def get_coherent_noise_model(noise_level: float) -> GateSubstitutionNoiseModel:
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


def conv_execute(circuit: Circuit, noise_level: float = 0.15):
    return (
        DensityMatrixSimulator(noise=get_coherent_noise_model(noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )


ideal_value = conv_execute(conv_circuit, noise_level=0.0)
noisy_value = conv_execute(conv_circuit, noise_level=0.15)

num_circuits_list = [1, 2, 5, 10, 25, 50, 100, 200]
errors = []

for n in num_circuits_list:
    twirled = generate_pauli_twirl_variants(conv_circuit, num_circuits=n)
    vals = Executor(partial(conv_execute, noise_level=0.15)).evaluate(twirled)
    errors.append(abs(ideal_value - np.average(vals)))

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

### Connection to error scaling

The PTM representation makes it straightforward to see why coherent noise
accumulates more severely than incoherent noise. For a coherent error such as
a small rotation by angle $\theta$, the PTM has off-diagonal entries of order
$\theta$. When gates are composed, these off-diagonal terms can interfere
constructively, causing the worst-case error to scale as
$\sqrt{r(\mathcal{E})} \propto \theta$ {cite}`Wallman_2014`.

For an incoherent (Pauli) channel, the PTM is diagonal and all entries
deviate from the ideal by $O(\theta^2)$, so errors accumulate linearly:
$r(\mathcal{E}) \propto \theta^2$. After Pauli twirling, the off-diagonal
terms vanish and the coherent quadratic penalty is removed.
