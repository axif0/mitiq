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
