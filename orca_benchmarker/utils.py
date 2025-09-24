import os
import re
from pathlib import Path
from shutil import copyfile
from subprocess import run

from ase.atoms import Atoms
from ase.calculators.orca import ORCA
from ase.io import read

from .result import Result


def get_job_state(job_id: int | str) -> str:
    command = f"sacct --noheader --jobs {job_id} --format=jobid,state"

    stdout = run(command.split(), capture_output=True, text=True).stdout
    while stdout == "":
        # It is possible that sacct has some trouble finding the job.
        # If that is the case, wait for a bit, and then try again.
        sleep(0.1)
        stdout = run(command.split(), capture_output=True, text=True).stdout
    state = stdout.split("\n")[0].split()[1]
    return state


def wait_until_job_done(job_id: int | str, sleep_interval: int = 0.5) -> None:
    # possible job state codes: https://slurm.schedmd.com/job_state_codes.html

    # wait until the calculation is done.
    job_state = get_job_state(job_id)
    while job_state in ["RUNNING", "PENDING"]:
        sleep(sleep_interval)
        job_state = get_job_state(job_id)


def slurm_calculator(
    calculator: ORCA, atoms: Atoms, slurm_script_path: str | Path
) -> int:  # Result:
    calculator.write_inputfiles(atoms, "energy")

    # Copy slurm script
    slurm_script_destination = calculator.directory / "calc.sh"
    copyfile(slurm_script_path, slurm_script_destination)

    try:
        from slurm_manager.job import SlurmJob

        job = SlurmJob.start_from_command(
            "sbatch calc.sh", directory=calculator.directory
        )
        job_id = job.job_id
    except (ImportError, ModuleNotFoundError):
        init_dir = os.getcwd()

        # Change to correct directory
        os.chdir(calculator.directory)

        # Start the calculation, and wait until it is done
        process = run(["sbatch", "calc.sh"], capture_output=True, text=True)
        job_id = process.stdout.split()[-1]
        # wait_until_job_done(job_id)

        # Change back to original directory
        os.chdir(init_dir)
    return int(job_id)
    # output_path = calculator.directory / "orca.out"
    # return Result.from_orca_output_file(output_path)


def safe_calculate(
    calculator: ORCA,
    atoms: Atoms,
    force: bool = False,
    use_slurm: bool = False,
    slurm_script_path: str | Path | None = None,
) -> int | None:
    output_path = Path(calculator.directory) / "orca.out"
    if not force and output_path.is_file():
        result = Result.from_orca_output_file(output_path)
        if result.completed:
            return None
        print("\tCalculation started but did not finish. Restarting")

    print("Starting calculation for", calculator.directory.parts[-1])
    if use_slurm:
        if slurm_script_path is None:
            msg = "If use_slurm is True, slurm_script_path needs to be set but was None"
            raise ValueError(msg)
        job_id = slurm_calculator(calculator, atoms, slurm_script_path)
        return job_id

    try:
        energy = calculator.get_potential_energy(atoms)
        try:
            forces = calculator.get_forces(atoms)
        except PropertyNotImplementedError:
            forces = None
    except CalledProcessError as e:
        print(f"\tError running calculation for benchmark {calculator.directory}:")
        print("\t\t" + str(e))
        print(f"\t\tInspect output file {output_path}")
    return None


def replace_brackets_parenthesis_by_lowercase(string: str) -> str:
    """Replace something in parentheses and capitalized by same thing lowercase and without parentheses.
    Example:
        CCSD(T) -> CCSDt
    """
    return re.sub(
        r"[(][A-Z]+[)]",
        lambda m: m.group(0).replace("(", "").replace(")", "").lower(),
        string,
    )


def convert_benchmark_to_dir(benchmark: str, reference=False) -> Path:
    """Converts the benchmark string to a valid directory"""
    benchmark_dir = benchmark.replace(" ", "_").replace("/", "")
    benchmark_dir = replace_brackets_parenthesis_by_lowercase(str(benchmark_dir))
    if reference:
        return Path("reference_" + benchmark_dir)
    return Path(benchmark_dir)
