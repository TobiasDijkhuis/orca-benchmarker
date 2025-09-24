import os
from pathlib import Path

from ase.calculators.orca import OrcaProfile
from ase.io import read

from orca_benchmarker.benchmark_parser import BenchmarkFunctionals

if __name__ == "__main__":
    trimer_benchmarks = BenchmarkFunctionals.from_yaml("benchmarks.yaml")

    use_slurm = True
    slurm_script_path = "calc.sh"

    trimer_path = Path(os.path.dirname(__file__)) / "optimization.traj"
    trimer = read(trimer_path, index=-1)
    trimer_benchmarks.run(
        trimer,
        profile=OrcaProfile("/opt/orca-6.1.0/orca"),
        directory="results",
        use_slurm=use_slurm,
        slurm_script_path=slurm_script_path,
    )

    trimer_benchmarks.wait()

    benchmark_results, reference_result = trimer_benchmarks.read_results(
        directory="results"
    )
