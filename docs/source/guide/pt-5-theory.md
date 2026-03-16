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

Let $\mathcal{G}_n = \{I, X, Y, Z\}^{\otimes n}$ be the $n$-qubit Pauli group
(with $|\mathcal{G}_n| = 4^n$ elements, ignoring phases). Given an arbitrary
CPTP map $\mathcal{E}$ with Kraus representation

$$
\mathcal{E}(\rho) = \sum_j K_j \rho K_j^\dagger,
$$

the **Pauli twirl** of $\mathcal{E}$ is defined as

$$
\mathcal{T}_P[\mathcal{E}](\rho)
= \frac{1}{4^n} \sum_{\sigma \in \mathcal{G}_n}
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

**Proof sketch.** Expand each Kraus operator in the Pauli basis:
$K_j = \sum_k c_{jk} \sigma_k$. The twirl average eliminates all
cross-terms $c_{jk} c_{jl}^*$ for $k \neq l$ due to the Schur
orthogonality relation for the Pauli group acting by conjugation:

$$
\frac{1}{4^n} \sum_{\sigma \in \mathcal{G}_n}
\sigma \, \sigma_k \, \sigma \, \sigma_l
= \delta_{kl} \, \sigma_k.
$$

Only the $k = l$ terms survive, yielding a diagonal Pauli channel.

### Gate twirling (dressed twirling)

Directly applying Eq. {math:numref}`pauli_twirl_def` to a noisy gate
$\mathcal{E} = \mathcal{N} \circ U$ would destroy the ideal unitary $U$.
Instead, Mitiq implements **dressed twirling**: for each Pauli $P_i$,
find $P_i'$ such that $U P_i = P_i' U$ (up to a global phase). The
twirled circuit is then

$$
P_i' \circ U \circ P_i = U \quad \text{(ideal effect preserved)},
$$

while the noise $\mathcal{N}$ gets twirled. For Clifford gates (CNOT, CZ),
the conjugation $U P_i U^\dagger$ always yields another Pauli, making this
straightforward. The valid twirl pairs $(P, Q, R, S)$ for CNOT and CZ gates
are stored in {attr}`mitiq.pt.pt.CNOT_twirling_gates` and
{attr}`mitiq.pt.pt.CZ_twirling_gates`.

The dressed twirling circuit for a two-qubit gate looks like:

$$
\begin{array}{c}
-\!- P -\!- \bullet -\!- R -\!- \\
\qquad\qquad\; | \\
-\!- Q -\!- \oplus -\!- S -\!-
\end{array}
$$

where $(P, Q, R, S)$ is randomly sampled from the twirl group for each gate instance.

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

### Connection to error scaling

The PTM makes it straightforward to see why coherent noise is worse than
incoherent noise. Consider a single-qubit noisy rotation $R_{Y_\theta}$.
Its PTM is

$$
R_{R_{Y_\theta}} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & 0 & \sin\theta \\
0 & -\sin\theta & 0 & \cos\theta \\
0 & 0 & 0 & 1
\end{pmatrix}.
$$

In the small-angle limit, $\cos\theta \approx 1 - \theta^2/2$ and
$\sin\theta \approx \theta$. The off-diagonal terms (coherent errors)
scale as $\theta$, while the diagonal deviation from identity scales as
$\theta^2$. This means the worst-case coherent error
$\sqrt{r(\mathcal{E})} \propto \theta$ is quadratically worse than the
incoherent error $r(\mathcal{E}) \propto \theta^2$.

After Pauli twirling, the off-diagonal terms vanish, and the remaining
diagonal errors scale as $\theta^2$ --- the quadratic penalty is removed.

## Relationship to other techniques

PT is a **preprocessing** step that restructures noise for downstream
mitigation. It composes naturally with:

- **[ZNE](zne-5-theory.md)**: PT converts coherent noise to Pauli noise;
  ZNE then amplifies and extrapolates the structured noise more predictably.
- **[PEC](pec-5-theory.md)**: A Pauli noise model is easier to learn
  and decompose for probabilistic error cancellation.
- **[DDD](ddd-5-theory.md)**: DDD suppresses idle-time decoherence while
  PT tailors gate noise. They target different error sources.

See [What additional options are available?](pt-3-options.md) for a
worked example of stacking PT with ZNE.
