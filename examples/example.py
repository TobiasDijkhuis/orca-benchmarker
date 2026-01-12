import os
from pathlib import Path

from ase.io import read

from orca_benchmarker.benchmark_functionals import BenchmarkFunctionals

if __name__ == "__main__":
    trimer_benchmarks = BenchmarkFunctionals.from_yaml("benchmarks.yaml")

    trimer_path = Path(os.path.dirname(__file__)) / "optimization.traj"
    trimer = read(trimer_path, index=-1)

    # Run the benchmarks
    trimer_benchmarks.run(
        trimer,
        directory="results",
        charge=0,
        mult=1,
    )

    # Get the Results
    benchmark_results, reference_result = trimer_benchmarks.read_results(
        directory="results"
    )
