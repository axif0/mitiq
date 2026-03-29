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

The sections below define the Pauli twirl, describe dressed twirling as 
implemented in Mitiq, use the Pauli transfer matrix (PTM) to visualize the 
effect of twirling on noise, and briefly discuss finite sampling and error-rate 
scaling.

## The Pauli twirl of a channel

Let $P_n = \{I, X, Y, Z\}^{\otimes n}$ be the $n$-qubit Pauli group
(with $|P_n| = 4^n$ elements, ignoring phases). Given an arbitrary
CPTP map $\mathcal{E}$ with Kraus representation

$$
\mathcal{E}(\rho) = \sum_j K_j \rho K_j^\dagger,
$$

the **Pauli twirl** of $\mathcal{E}$ is defined as

$$
\mathcal{T}_P[\mathcal{E}](\rho)
= \frac{1}{4^n} \sum_{\sigma \in P_n}
  \sigma \,\mathcal{E}(\sigma \rho \sigma^\dagger)\, \sigma^\dagger.
$$(pauli_twirl_def)

### Twirling produces a Pauli channel

For any CPTP map $\mathcal{E}$, the Pauli twirl $\mathcal{T}_P[\mathcal{E}]$
is a **Pauli channel**:

$$
\mathcal{T}_P[\mathcal{E}](\rho) = \sum_k p_k \, \sigma_k \rho \sigma_k^\dagger,
$$(pauli_channel_result)

where the probabilities are

$$
p_k = \frac{1}{4^n} \sum_j \left| \mathrm{Tr}(\sigma_k K_j) \right|^2.
$$

This follows from the Schur orthogonality relation for the Pauli group acting
by conjugation {cite}`Wallman_2016_PRA`. The key consequence is that
*coherent* (unitary) error components are converted into *incoherent*
(stochastic) Pauli errors, which are generally easier for downstream
error mitigation techniques to suppress.

## Gate twirling (dressed twirling)

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
straightforward. Each two-qubit gate is sandwiched by a randomly sampled
Pauli pair $(P, Q)$ before and a corresponding pair $(R, S)$ after, chosen
so that the ideal gate is preserved and only the noise is twirled. The valid
pairs $(P, Q, R, S)$ for CNOT and CZ gates are stored in
{attr}`mitiq.pt.pt.CNOT_twirling_gates` and
{attr}`mitiq.pt.pt.CZ_twirling_gates`.

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

For a general channel, $\Lambda$ has off-diagonal entries that encode coherent
mixing between Pauli components. Pauli twirling is the same group average as in
Eq. {math:numref}`pauli_twirl_def`, so in the PTM basis the twirled channel is
**diagonal**:

$$
\Lambda_{\text{twirled}} = \mathrm{diag}(1, \lambda_1, \lambda_2, \ldots, \lambda_{4^n - 1}),
$$

with $\Lambda_{ij} = 0$ for $i \neq j$ and each $\lambda_k$ fixed by the Pauli
weights $p_k$ {cite}`Wallman_2016_PRA`.

The following code generates PTM heatmaps for a CNOT gate under several noise 
channels.

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

Coherent $R_y$ noise (top row) shows large off-diagonal PTM entries that are
suppressed after averaging over twirled circuits. Depolarizing and bit-flip
noise are already Pauli channels, so their PTMs are essentially unchanged by
PT. Amplitude damping is non-unital; PT suppresses coherent cross-terms but does
not turn it into a Pauli channel in the same way as for coherent gate noise.

## Convergence with number of twirled circuits

The twirl in Eq. {math:numref}`pauli_twirl_def` is estimated by drawing $N$
twirled circuits independently and averaging expectation values. Too small $N$
leaves a biased estimate of the fully twirled channel.

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
Convergence speed depends on the circuit, noise model, and noise strength.
Strong coherent noise typically needs more twirled circuits; 50–100 variants is
often used as a starting point in practice.
```

## Connection to error scaling

Small coherent rotations contribute PTM off-diagonal terms of order $\theta$
that can add constructively across gates, so worst-case error measures can
scale like $\sqrt{r(\mathcal{E})} \propto \theta$ {cite}`Wallman_2014`. Pauli
noise corresponds to a diagonal PTM with deviations $O(\theta^2)$ from the
identity, which accumulates more favorably. After twirling, the coherent
cross-terms are removed in the averaged channel, which is why PT is composed
with QEM methods that assume stochastic Pauli-like noise.
