# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark mitiq performance on benchmarking circuits.

Uses the Calibrator with a single ZNE configuration across 5 distinct
benchmark circuits and computes the average improvement factor.

Reference: https://github.com/unitaryfund/mitiq/issues/2828
"""

import cirq
import numpy as np

from mitiq import Calibrator, MeasurementResult, Settings
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global

NOISE_LEVEL = 0.01
SHOTS = 1_000


def noisy_execute(circuit: cirq.Circuit) -> MeasurementResult:
    """Execute a circuit with depolarizing noise and return bitstring counts.

    The Calibrator requires an executor that returns MeasurementResult
    (bitstrings/counts), not expectation values.
    """
    noisy_circuit = circuit.with_noise(cirq.depolarize(NOISE_LEVEL))
    result = cirq.DensityMatrixSimulator().run(
        noisy_circuit, repetitions=SHOTS
    )
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


benchmark_settings = Settings(
    benchmarks=[
        {
            "circuit_type": "ghz",
            "num_qubits": 2,
        },
        {
            "circuit_type": "w",
            "num_qubits": 2,
        },
        {
            "circuit_type": "rb",
            "num_qubits": 2,
            "circuit_depth": 5,
        },
        {
            "circuit_type": "mirror",
            "num_qubits": 2,
            "circuit_depth": 5,
            "circuit_seed": 1,
        },
        {
            "circuit_type": "rb",
            "num_qubits": 2,
            "circuit_depth": 10,
        },
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
    ],
)


def compute_improvement_factors(cal: Calibrator) -> np.ndarray:
    """Extract per-circuit improvement factors from calibration results.

    Improvement factor = noisy_error / mitigated_error, where
    noisy_error = |ideal - noisy| and mitigated_error = |ideal - mitigated|.
    """
    results = cal.results
    factors = np.zeros(results.num_problems)
    for problem in cal.problems:
        noisy_err, mitigated_err = results._get_errors(
            strategy_id=0, problem_id=problem.id
        )
        factors[problem.id] = (
            noisy_err / mitigated_err if mitigated_err > 0 else np.inf
        )
    return factors


def main() -> None:
    cal = Calibrator(
        noisy_execute, frontend="cirq", settings=benchmark_settings
    )

    print(f"Cost estimate: {cal.get_cost()}")
    print(f"Number of benchmark circuits: {len(cal.problems)}")
    print(f"Number of strategies: {len(cal.strategies)}")
    print()

    cal.run(log="flat")
    print()

    improvement_factors = compute_improvement_factors(cal)

    print("Per-circuit improvement factors:")
    print("-" * 50)
    for problem in cal.problems:
        factor = improvement_factors[problem.id]
        noisy_err, mitigated_err = cal.results._get_errors(
            strategy_id=0, problem_id=problem.id
        )
        improved = "YES" if factor > 1.0 else "NO"
        print(
            f"  {problem.type:>8s} (depth={problem.circuit_depth:>3d}): "
            f"improvement={factor:.4f}  "
            f"noisy_err={noisy_err:.4f}  "
            f"mitigated_err={mitigated_err:.4f}  "
            f"[{improved}]"
        )

    avg_factor = np.mean(improvement_factors)
    print("-" * 50)
    print(f"Average improvement factor: {avg_factor:.4f}")

    best = cal.best_strategy()
    print(f"\nBest strategy: {best.to_dict()}")


if __name__ == "__main__":
    main()
