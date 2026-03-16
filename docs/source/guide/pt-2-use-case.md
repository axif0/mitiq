---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use PT?

## Advantages

Pauli Twirling is a technique devised to tailor noise towards Pauli channels.
More details on the theory of Pauli Twirling are given in the section
[What is the theory behind PT?](pt-5-theory.md).

### Converts coherent noise to incoherent noise

Quantum noise broadly falls into two categories: **coherent** and **incoherent**.

**Coherent noise** consists of unwanted unitary rotations (e.g. gate over-rotations,
cross-talk). While reversible in principle, coherent errors are particularly
damaging because they accumulate *quadratically* with circuit depth.
If $r(\mathcal{E})$ is the average gate infidelity, the worst-case error
for coherent noise scales as $\sqrt{r(\mathcal{E})}$
{cite}`Wallman_2014`, which is far worse than
the average error rate would suggest.

In the Pauli Transfer Matrix (PTM) representation, coherent noise
manifests as **off-diagonal elements**. These off-diagonal terms cause
errors from successive gates to interfere constructively, compounding
the damage across the circuit.

**Incoherent noise** (e.g. depolarizing, dephasing, amplitude damping) introduces
classical randomness and decoherence. It is irreversible, but it accumulates only
*linearly* with circuit depth: the worst-case error scales as $r(\mathcal{E})$ directly.
In the PTM representation, a stochastic Pauli channel is **diagonal** --- there
are no off-diagonal terms to interfere constructively.

Pauli Twirling converts arbitrary noise channels (including coherent errors)
into stochastic Pauli channels by randomly conjugating each noisy gate with
Pauli operators and averaging over the results. The off-diagonal PTM entries
cancel under this averaging, leaving only the diagonal Pauli channel.
This trades a quadratically-scaling worst-case error for a linearly-scaling one.

### Noise-agnostic and easy to use

PT does not require knowledge of the underlying noise model.
It works with any Markovian noise channel and requires
only the ability to insert single-qubit Pauli gates around two-qubit gates.

### Enables downstream error mitigation

Because PT converts noise into structured Pauli channels, it makes subsequent
quantum error mitigation (QEM) techniques more effective. Techniques like
[ZNE](zne.md), [PEC](pec.md), and [CDR](cdr.md) benefit from noise
that is stochastic and well-characterized. See the
[stacking PT with QEM section](pt-3-options.md) for details.

## Disadvantages

### Increased circuit depth without compilation

Pauli Twirling inserts additional single-qubit Pauli gates before and after
each target two-qubit gate. In hardware implementations, these are typically
absorbed into adjacent single-qubit layers by a compilation pass. Mitiq does
not currently provide this compilation, so circuit depth increases by the
additional gates. In practice, single-qubit gate errors are typically much
smaller than two-qubit gate errors, so this overhead is often negligible.

### Risk of fully depolarizing the noise

PT does not reduce the total error rate --- it restructures the noise.
In some cases, the tailored noise channel can approach a
**completely depolarizing channel**, where all quantum information is lost.
This happens when the original coherent errors are large relative to the
gate fidelity, and the Pauli-averaged channel distributes error
weight roughly equally across all Pauli operators.

For this reason, PT should be treated as a noise *tailoring* technique
rather than a noise *reduction* technique, and is most effective when
composed with a dedicated error mitigation method.
