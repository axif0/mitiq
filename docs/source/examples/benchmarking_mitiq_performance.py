# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark mitiq performance on benchmarking circuits.

Uses the Calibrator with a single ZNE configuration across 5 distinct
benchmark circuits and computes the average improvement factor.

"""

import cirq
import matplotlib.pyplot as plt
import numpy as np

from mitiq import Calibrator, MeasurementResult, Settings
from mitiq.pec.representations import (
    represent_operation_with_local_depolarizing_noise,
)
from mitiq.zne.inference import ExpFactory, LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
    insert_id_layers,
)

NOISE_LEVEL_1Q = 0.001
NOISE_LEVEL_2Q = 0.01
SHOTS = 5_000


def noisy_execute(circuit: cirq.Circuit) -> MeasurementResult:
    """Execute a circuit with depolarizing noise and return bitstring counts.

    The Calibrator requires an executor that returns MeasurementResult
    (bitstrings/counts), not expectation values.
    """
    noisy_circuit = cirq.Circuit()
    for op in circuit.all_operations():
        noisy_circuit.append(op)
        if isinstance(op.gate, cirq.MeasurementGate):
            continue

        # In reality, 2-qubit gates are usually far noisier
        # than 1-qubit gates. We model this realistic hardware constraint by
        # injecting different depolarizing error rates directly after
        # each operation!
        if len(op.qubits) == 1:
            noisy_circuit.append(
                cirq.depolarize(NOISE_LEVEL_1Q).on(*op.qubits)
            )
        elif len(op.qubits) == 2:
            noisy_circuit.append(
                cirq.depolarize(NOISE_LEVEL_2Q, n_qubits=2).on(*op.qubits)
            )

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
            "scale_noise": fold_global,  # Performs fine on shallow circuits!
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,  # It behaves much more predictably on
            # short circuits when avoiding
            # fractions or even numbers.
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        #  asymptote=0.25: Since we use a depolarizing
        # channel on 2-qubit gates, the signal physically decays exponentially
        # toward a fully mixed random state (1/4 probability). Pinning the
        # asymptote prevents convergence failures and lets the math perfectly
        # fit the deep circuit data!
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": ExpFactory([1.0, 3.0, 5.0], asymptote=0.25),
        },
        # Instead of scaling the entire circuit globally at the end,
        # fold logic gates at random internally.
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
        # Rather than applying inverse logic gates, this simply idles
        # the qubits to accumulate pure time-decoherence.
        {
            "technique": "zne",
            "scale_noise": insert_id_layers,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        # Instead of scaling, learn a quasi-probability representation
        # for gates. This pays the resource cost in terms of shots
        # (samples) rather than circuit depth.
        {
            "technique": "pec",
            "representation_function": (
                represent_operation_with_local_depolarizing_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": NOISE_LEVEL_2Q,
            "num_samples": 100,
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

    # Scatter plot: improvement factor vs estimated resource cost.
    # ZNE cost  = circuit_depth × sum(scale_factors) # extra gate overhead.
    # PEC cost  = circuit_depth × (1 + num_samples/SHOTS) # extra shot overhead
    #             normalised into "effective depth" units so both axes are
    #             comparable on the same plot.
    plt.figure(figsize=(10, 6))
    for strategy_id, strategy in enumerate(cal.strategies):
        s_dict = strategy.to_dict()
        tech = s_dict.get("technique", "")

        resources = []
        improvements = []
        for problem in cal.problems:
            noisy_err, mitigated_err = cal.results._get_errors(
                strategy_id=strategy_id, problem_id=problem.id
            )
            factor = noisy_err / mitigated_err if mitigated_err > 0 else np.inf

            if tech == "ZNE":
                scale_factors = s_dict.get("scale_factors", [1.0])
                cost = problem.circuit_depth * sum(scale_factors)
            elif tech == "PEC":
                num_samples = s_dict.get("num_samples", 1)
                cost = problem.circuit_depth * (1 + num_samples / SHOTS)
            else:
                cost = problem.circuit_depth

            resources.append(cost)
            improvements.append(factor)

        if tech == "PEC":
            rep_name = s_dict.get("representation_function", "Unknown")
            if "depolarizing_noise" in rep_name:
                rep_name = "local_depolarizing"
            samples = s_dict.get("num_samples", "Unknown")
            label = f"PEC ({rep_name}, {samples} samples)"
        else:
            factory_name = s_dict.get("factory", "Unknown")
            scale_name = s_dict.get("scale_method", "Unknown")
            label = f"{factory_name} ({scale_name})"

        avg_resource = np.mean(resources)
        avg_improvement = np.mean(improvements)

        plt.scatter(
            avg_resource, avg_improvement, label=label, s=80, alpha=0.7
        )

    plt.axhline(y=1.0, color="r", linestyle="--", label="No Improvement")
    plt.xlabel(
        "Estimated Resource Cost\n"
        "(ZNE: avg depth × Σ scale factors  |  "
        "PEC: avg depth × (1 + samples/shots))",
        fontsize=11,
    )
    plt.ylabel("Improvement Factor", fontsize=12)
    plt.title("ZNE & PEC Strategy Performance vs Resource Cost", fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
    # If we want to save the plot, we can just uncomment the following lines:
    # plt.savefig("zne_pec_strategy_performance.png")
    # print("\nSaved performance plot to 'zne_pec_strategy_performance.png'")


if __name__ == "__main__":
    main()
