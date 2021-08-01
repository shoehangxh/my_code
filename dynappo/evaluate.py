
from typing import Callable, List, Tuple


import baselines


def robustness(
    landscape,
    make_explorer,
    signal_strengths: List[float] = [0, 0.5, 0.75, 0.9, 1],
    verbose: bool = True,
):
    results = []
    for ss in signal_strengths:
        print(f"Evaluating for robustness with model accuracy; signal_strength: {ss}")

        model = baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)
        explorer = make_explorer(model, ss)
        res = explorer.run(landscape, verbose=verbose)

        results.append((ss, res))

    return results


def efficiency(
    landscape,
    make_explorer,
    budgets: List[Tuple[int, int]] = [
        (100, 500),
        (100, 5000),
        (1000, 5000),
        (1000, 10000),
    ],
):
    results = []
    for sequences_batch_size, model_queries_per_batch in budgets:
        print(
            f"Evaluating for sequences_batch_size: {sequences_batch_size}, "
            f"model_queries_per_batch: {model_queries_per_batch}"
        )
        explorer = make_explorer(sequences_batch_size, model_queries_per_batch)
        res = explorer.run(
            landscape
        )  # TODO: is this being logged? bc the last budget pair would take very long

        results.append(((sequences_batch_size, model_queries_per_batch), res))

    return results


def adaptivity(
    landscape,
    make_explorer,
    num_rounds: List[int] = [1, 10, 100],
    total_ground_truth_measurements: int = 1000,
    total_model_queries: int = 10000,
):
    results = []
    for rounds in num_rounds:
        print(f"Evaluating for num_rounds: {rounds}")
        explorer = make_explorer(
            rounds,
            int(total_ground_truth_measurements / rounds),
            int(total_model_queries / rounds),
        )
        res = explorer.run(landscape)

        results.append((rounds, res))

    return results
