import os
import re
from pathlib import Path
from shutil import copyfile
from subprocess import run

from ase.atoms import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.orca import ORCA
from ase.io import read
from textwrap import dedent

from typing import Any
from orca_benchmarker.result import Result


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


def safe_calculate(
    calculator: ORCA,
    atoms: Atoms,
    force: bool = False,
    use_slurm: bool = False,
    slurm_script_path: str | Path | None = None,
) -> int | None:
    output_path = Path(calculator.directory) / "orca.out"
    print("Starting calculation for", calculator.directory.parts[-1])
    if not force and output_path.is_file():
        result = Result.from_orca_output_file(output_path)
        if result.completed:
            return None

        print("\tCalculation started but did not finish. Restarting")

    if use_slurm:
        if slurm_script_path is None:
            msg = "If use_slurm is True, slurm_script_path needs to be set but was None"
            raise ValueError(msg)
        # We Check whether the job is running, by finding whether one of the jobs running or in the queue has the
        # same bash calc file as this one would have. By running squeue -u --format="%A,%o
        # See https://slurm.schedmd.com/squeue.html for more info
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


def remove_libxc_from_string(string: str) -> str:
    return re.sub(
        r"LibXC[(][a-zA-Z0-9]+[)]",
        lambda m: m.group(0).replace("LibXC(", "").replace(")", ""),
        string,
    )


def convert_benchmark_to_dir(benchmark: str, reference=False) -> Path:
    """Converts the benchmark string to a valid directory"""
    benchmark_dir = benchmark.replace(" ", "_").replace("/", "")
    benchmark_dir = remove_libxc_from_string(benchmark_dir)
    benchmark_dir = replace_brackets_parenthesis_by_lowercase(str(benchmark_dir))
    if reference:
        return Path("reference_" + benchmark_dir)
    return Path(benchmark_dir)


def create_job_array(
    calculators: list[ORCA],
    atoms: Atoms,
    orca_path: str | Path = "orca",
    slurm_parameters: dict[str, Any] = {},
    output_file: str | Path = "run.sh",
    parameter_file: str | Path = "directories.txt",
) -> None:
    if isinstance(calculators, ORCA):
        calculators = [calculators]

    n_calculators = len(calculators)
    if n_calculators == 0:
        raise ValueError(f"At least one instance of ORCA should be passed")

    with open(parameter_file, "w") as file:
        for calculator_idx, calculator in enumerate(calculators):
            if not isinstance(calculator, ORCA):
                raise TypeError(
                    f"Calculator at index {calculator_idx} was not of type ORCA"
                )
            calculator.write_inputfiles(atoms, all_properties)
            file.write(f"{calculator.directory}\n")

    with open(output_file, "w") as file:
        file.write("#!/bin/bash\n\n")
        for key, value in slurm_parameters.items():
            file.write(f"#SBATCH --{key}={value}\n")

        file.write(
            dedent(f"""
                #SBATCH --array=1-{n_calculators}

                calculation_dirs_file={parameter_file}
                calculation_dir=$(sed "{{SLURM_ARRAY_TASK_ID}}q;d" $calculation_dirs_file)

                # Replace NPROCS_HERE by the amount of tasks per node.
                sed -i -e "s/NPROCS_HERE/${{SLURM_NTASKS_PER_NODE}}/g" ${{calculation_dir}}/orca.inp
                
                # Always run stuff in local scratch on node for job and not in ${{HOME}} to avoid NFS network slowness
                scratch_dir="/scratch/${{SLURM_JOB_USER}}/${{SLURM_JOB_ID}}"
                cd $scratch_dir
                
                # ORCA has the tendency to put temporary files in the same directory as the input file,
                # so we copy the input file from the calculation directory to the scratch directory
                cp ${{calculation_dir}}/orca.inp ${{scratch_dir}}/orca.inp
                
                # Run calculation. Take input file from scratch directory, and put output there too.
                {orca_path} ${{scratch_dir}}/orca.inp > ${{calculation_dir}}/orca.out

                # Remove temporary stuff in case of a failed calculation
                rm *tmp*

                # Remove unnecessary stuff
                rm orca.densities
                rm orca.gbw

                # Copy relevant things to calculation directory
                cp ${{scratch_dir}}/orca.* ${{calculation_dir}}/.""")
        )
