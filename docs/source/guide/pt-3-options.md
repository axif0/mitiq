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
{func}`.generate_pauli_twirl_variants`. This page covers how to add custom
noise channels to twirled circuits using {func}`.add_noise_to_two_qubit_gates`.


## Adding noise to twirled circuits with `add_noise_to_two_qubit_gates`

The function {func}`.add_noise_to_two_qubit_gates` inserts a noise
channel after every CNOT and CZ gate in a twirled circuit. This is
useful for simulating the effect of PT on circuits with specific noise
models.

Two noise channels are built in: `"bit-flip"` and `"depolarize"`.

```{code-cell} ipython3
import cirq
from cirq import LineQubit, Circuit, CNOT
from mitiq.pt import generate_pauli_twirl_variants
from mitiq.pt.pt import add_noise_to_two_qubit_gates

q0, q1 = LineQubit.range(2)
cnot_circuit = Circuit(CNOT(q0, q1))

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

## How the number of generated Pauli twirled circuits affects the outcome

The number of twirled circuits generated (controlled by `num_circuits`) determines how well the physical average approximates the ideal, intended Pauli channel. 

- **Small `num_circuits` (e.g., 1-10):** The variance in the expectation values will be quite high because the Pauli group hasn't been adequately sampled over. The resulting effective noise channel might still contain coherent properties.
- **Large `num_circuits` (e.g., 20-100+):** By the law of large numbers, the averaged results converge toward the exact stochastic Pauli channel. However, generating and evaluating too many circuits increases the simulation or execution cost. 

In practice, executing 20 to 50 twirl variants per expectation value provides a good balance between sufficient noise tailoring and execution overhead on physical hardware.

## Which noise channels are tailored by PT?

PT aims to convert coherent or generally asymmetric noise into purely stochastic Pauli noise. Any noise channel with off-diagonal terms in its Pauli Transfer Matrix (PTM) benefits from twirling, which zeroes out those off-diagonals. 

The following heatmaps provide a visual representation of how different original noise channels are transformed (tailored) after Pauli Twirling.

```{figure} ../img/pt_ptm_heatmaps.png
---
width: 600px
name: pt-heatmaps
---
PTM heatmaps before and after PT. The off-diagonal block terms of Coherent over-rotations (top) and Amplitude Damping (middle) are perfectly zeroed out. Depolarizing noise (bottom), which is already stochastic Pauli noise, remains unchanged.
```

## Stacking PT with other QEM techniques

Because PT does not remove noise but rather structures it favorably (eliminating coherent worst-case scaling), it is almost always best used as a **preprocessing step** before applying a dedicated quantum error mitigation technique like [Zero-Noise Extrapolation (ZNE)](zne.md) or [Probabilistic Error Cancellation (PEC)](pec.md).

For example, ZNE relies on the assumption that noise scales predictably as the circuit depth increases. Coherent errors violate this assumption by scaling quadratically or constructively. By applying PT *at each noise scale level*, the noise becomes stochastic, guaranteeing a much smoother extrapolation curve. 

When stacking PT with a QEM technique:
1. Generate the base mitigating circuits (e.g. at various scale factors for ZNE).
2. Apply PT variants around the multi-qubit gates in *each* of the scaled circuits.
3. Compute the average expectation value for the PT variants at each scale level.
4. Feed those partially noise-tailored expectation values into the QEM inference engine.

```{figure} ../img/pt_qem_workflow.svg
---
width: 700px
name: pt-qem-workflow
---
Workflow demonstrating the stacking of PT before inference in a broader QEM pipeline.
```
