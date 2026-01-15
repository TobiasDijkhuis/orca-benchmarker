import os
import re
from pathlib import Path
from subprocess import run
from typing import Any, Type

from ase.calculators.calculator import BadConfiguration
from ase.calculators.orca import ORCA

# def get_job_state(job_id: int | str) -> str:
#     command = f"sacct --noheader --jobs {job_id} --format=jobid,state"
#
#     stdout = run(command.split(), capture_output=True, text=True).stdout
#     while stdout == "":
#         # It is possible that sacct has some trouble finding the job.
#         # If that is the case, wait for a bit, and then try again.
#         sleep(0.1)
#         stdout = run(command.split(), capture_output=True, text=True).stdout
#     state = stdout.split("\n")[0].split()[1]
#     return state
#
#
# def wait_until_job_done(job_id: int | str, sleep_interval: int = 0.5) -> None:
#     # possible job state codes: https://slurm.schedmd.com/job_state_codes.html
#
#     # wait until the calculation is done.
#     job_state = get_job_state(job_id)
#     while job_state in ["RUNNING", "PENDING"]:
#         sleep(sleep_interval)
#         job_state = get_job_state(job_id)
#
#
# def slurm_calculator(
#     calculator: ORCA, atoms: Atoms, slurm_script_path: str | Path
# ) -> int:  # Result:
#     calculator.write_inputfiles(atoms, "energy")
#
#     # Copy slurm script
#     slurm_script_destination = calculator.directory / "calc.sh"
#     copyfile(slurm_script_path, slurm_script_destination)
#
#     try:
#         from slurm_manager.job import SlurmJob
#
#         job = SlurmJob.start_from_command(
#             "sbatch calc.sh", directory=calculator.directory
#         )
#         job_id = job.job_id
#     except (ImportError, ModuleNotFoundError):
#         init_dir = os.getcwd()
#
#         # Change to correct directory
#         os.chdir(calculator.directory)
#
#         # Start the calculation, and wait until it is done
#         process = run(["sbatch", "calc.sh"], capture_output=True, text=True)
#         job_id = process.stdout.split()[-1]
#         # wait_until_job_done(job_id)
#
#         # Change back to original directory
#         os.chdir(init_dir)
#     return int(job_id)


def check_type(name: str, variable: Any, expected_type: Type[Any]) -> None:
    """Check that the type is a certain expected type.

    Args:
        name (str): name of the variable to check
        variable (Any): variable to check type
        expected_type (Type[Any]): type for variable to be

    Raises:
        TypeError: if variable is not an instance of expected_type
    """
    if not isinstance(variable, expected_type):
        raise TypeError(
            f"{name} was expected to be {expected_type}, but was {type(variable)}"
        )


# def safe_calculate(
#     calculator: ORCA,
#     atoms: Atoms,
#     force: bool = False,
#     use_slurm: bool = False,
#     slurm_script_path: str | Path | None = None,
# ) -> int | None:
#     output_path = Path(calculator.directory) / "orca.out"
#     print("Starting calculation for", calculator.directory.parts[-1])
#     if not force and output_path.is_file():
#         result = Result.from_orca_output_file(output_path)
#         if result.completed:
#             return None
#
#         print("\tCalculation started but did not finish. Restarting")
#
#     if use_slurm:
#         if slurm_script_path is None:
#             msg = "If use_slurm is True, slurm_script_path needs to be set but was None"
#             raise ValueError(msg)
#         # We Check whether the job is running, by finding whether one of the jobs running or in the queue has the
#         # same bash calc file as this one would have. By running squeue -u --format="%A,%o
#         # See https://slurm.schedmd.com/squeue.html for more info
#         job_id = slurm_calculator(calculator, atoms, slurm_script_path)
#         return job_id
#
#     try:
#         energy = calculator.get_potential_energy(atoms)
#         try:
#             forces = calculator.get_forces(atoms)
#         except PropertyNotImplementedError:
#             forces = None
#     except CalledProcessError as e:
#         print(f"\tError running calculation for benchmark {calculator.directory}:")
#         print("\t\t" + str(e))
#         print(f"\t\tInspect output file {output_path}")
#     return None


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


def remove_libxc_from_string(string: str) -> str:
    return re.sub(
        r"LibXC[(][a-zA-Z0-9]+[)]",
        lambda m: m.group(0).replace("LibXC(", "").replace(")", ""),
        string,
    )


def convert_benchmark_to_dir(benchmark: str, reference: bool = False) -> Path:
    """Converts the benchmark string to a valid directory"""
    benchmark_dir = benchmark.replace(" ", "_").replace("/", "")
    benchmark_dir = remove_libxc_from_string(benchmark_dir)
    benchmark_dir = replace_brackets_parenthesis_by_lowercase(str(benchmark_dir))
    if reference:
        return Path("reference_" + benchmark_dir)
    return Path(benchmark_dir)


def start_job(
    submission_script_path: str | Path, wait_until_done: bool = False
) -> None:
    """Start job by running `sbatch {submission_script_path}`.

    Args:
        submission_script_path (str | Path): path to submission script file
        wait_until_done (bool): wait until the job is done. Default: False

    Notes:
        This first changes to the submission script's directory, and then
            changes back to the initial working directory.
    """

    wd = os.getcwd()
    directory = Path(submission_script_path).parents[0]
    command = f"sbatch {submission_script_path.parts[-1]}".split()
    if wait_until_done:
        command.insert(1, "--wait")
    os.chdir(directory)
    run(command)
    os.chdir(wd)


def is_slurm_active() -> bool:
    """Runs a command to find out if slurm is installed and active.

    Returns:
        bool: whether slurm is installed and active.
    """
    try:
        command = run(["sacct", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return not command.stderr


def get_orca_path_from_ase_configuration() -> str | None:
    """Get the default ORCA path from the config.ini file

    Returns:
        str if `orca` `command` is defined in the ASE configuration,
            and None if it is not."""
    try:
        return ORCA().profile.command
    except BadConfiguration:
        return None


def get_orca_path(orca_path: str | Path | None) -> Path:
    """Get the ORCA path, either from the argument passed to this function or
    from the ASE configuration.

    Args:
        orca_path (str | Path | None): Path to ORCA executable

    Returns:
        Path: path to ORCA executable

    Raises:
        ValueError if both orca_path supplied to this function is None, and the
            `orca` `command` in the ASE configuration is not defined.
    """
    if orca_path is not None:
        return Path(orca_path)

    orca_path = get_orca_path_from_ase_configuration()
    if orca_path is not None:
        return Path(orca_path)

    raise ValueError(
        "Orca path was not set in ASE configuration, but orca_path given as kwarg was also None"
    )
