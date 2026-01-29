from pathlib import Path
from textwrap import dedent

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.orca import ORCA
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Bohr, Hartree

from orca_benchmarker.result import Result, ResultInfo
from orca_benchmarker.utils import check_type, get_orca_path


def read_orca_generated_trajectory(trajectory_path: str | Path) -> list[Atoms]:
    """Read ORCA generated xyz trajectory. Similar to ase.io.read, except also
    reads the energy in the comment line.

    Args:
        trajectory_path (str | Path): path to xyz file.

    Returns:
        atoms (list[Atoms]): list of frames, with energies attached by
            SinglePointCalculator instances.

    """
    with open(trajectory_path) as file:
        lines = file.readlines()
    lines = list(filter(lambda line: line != ">\n", lines))
    natoms = int(lines[0].strip())

    nframes = len(lines) // (natoms + 2)
    atoms = [None] * nframes
    for frame_idx in range(nframes):
        frame_lines = lines[frame_idx * (natoms + 2) : (frame_idx + 1) * (natoms + 2)]

        symbols = [""] * natoms
        positions = [[0.0, 0.0, 0.0]] * natoms
        for atom_idx, line in enumerate(frame_lines[2:]):
            split_line = line.split()
            symbols[atom_idx] = split_line[0]
            positions[atom_idx] = split_line[1:4]
        atoms[frame_idx] = Atoms(symbols=symbols, positions=positions)

        energy = float(frame_lines[1].split()[-1])  # Energy in Hartree
        atoms[frame_idx].calc = SinglePointCalculator(
            atoms[frame_idx], energy=energy * Hartree
        )
    return atoms


def write_orca_array_submission_script(
    parameter_file: str | Path = "directories.txt",
    output_file: str | Path = "run.sh",
    sbatch_kwargs: dict[str, str] | None = None,
    orca_path: str | Path | None = None,
    remove_gbw: bool = False,
    environment_commands: str | list[str] | None = None,
    scratch_dir: str = "$SCRATCH",
) -> None:
    """Write a job array submission script that runs ORCA in directories written in
    `parameter_file`.

    Args:
        parameter_file (str | Path):
        output_file (str | Path):
        sbatch_kwargs (dict[str, str] | None): keyword arguments written to the
            job submission script as "#SBATCH --{key}={value}". Default = None
        environment_commands (str | list[str] | None): commands written to the
            job submission script to set up the environment. This is written in
            order in the list, and can be something like
            `['module purge', 'module load OpenMPI/4.1.6-GCC-0.3.2']`.
            Default = None
        remove_gbw (bool): whether to remove the orca.gbw file to save storage.
            Default = False
        scratch_dir (str): scratch directory where to run the calculation.
            Outputs are copied to the calculation directory after completion.
            Default: `$SCRATCH`, such that Slurm evaluates it, which does not work
            on all clusters.
        orca_path (str | Path | None): path to the `orca` executable.
            If None, the path is inferred from the ASE configuration, see
            https://ase-lib.org/ase/calculators/calculators.html#calculator-configuration
            This can be set in `~/.config/ase/config.ini` as

                [orca]
                command = /path/to/orca/executable

            Default = None
    """
    orca_path = get_orca_path(orca_path)

    num_jobs_in_array = sum(1 for _ in open(parameter_file))

    with open(output_file, "w") as file:
        file.write("#!/bin/bash\n\n")
        if sbatch_kwargs is not None:
            for key, value in sbatch_kwargs.items():
                file.write(f"#SBATCH --{key}={value}\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --cpus-per-task=1\n")
        file.write("#SBATCH --threads-per-core=1\n\n")
        file.write(f"#SBATCH --array=1-{num_jobs_in_array}\n\n")

        if environment_commands is not None:
            if isinstance(environment_commands, str):
                environment_commands = [environment_commands]
            check_type("environment_commands", environment_commands, list)
            for environment_command in environment_commands:
                check_type("environment_command", environment_command, str)
                file.write(f"{environment_command}\n")

        nl = "\n"
        string = dedent(f"""

            calculation_dirs_file={parameter_file.resolve().as_posix()}
            calculation_dir=$(sed "${{SLURM_ARRAY_TASK_ID}}q;d" $calculation_dirs_file)
            calculation_dir=$(realpath $calculation_dir)
            
            # Replace NPROCS_HERE by the amount of tasks per node.
            sed -i -e "s/NPROCS_HERE/${{SLURM_NTASKS_PER_NODE}}/g" ${{calculation_dir}}/orca.inp
            
            # Always run stuff in local scratch on node for job and not in ${{HOME}} to avoid NFS network slowness
            scratch_dir={scratch_dir}
            cd $scratch_dir
            
            # ORCA has the tendency to put temporary files in the same directory as the input file,
            # so we copy the input file from the calculation directory to the scratch directory
            cp ${{calculation_dir}}/orca.inp orca.inp
            
            # Run calculation. Take input file from scratch directory, and put output there too.
            {orca_path} orca.inp > ${{calculation_dir}}/orca.out

            # Remove temporary stuff in case of a failed calculation
            # The "-f" suppresses the error if no files matching "*tmp*" exist
            rm -f *tmp*

            # Remove unnecessary stuff
            {f"# This file was written with 'remove_gbw = False'{nl}            # " if not remove_gbw else ""}rm orca.gbw; rm orca.densities*
            rm orca.bibtex
            rm orca.property.txt

            # Copy relevant things to calculation directory
            cp orca.* ${{calculation_dir}}/.""")
        file.write(string)


def create_parameters_from_calculator_list(
    calculator_list: list[ORCA] | ORCA,
    atoms: Atoms,
    parameter_file: str | Path = "directories.txt",
) -> None:
    if isinstance(calculator_list, ORCA):
        calculator_list = [calculator_list]

    n_calculators = len(calculator_list)
    if n_calculators == 0:
        raise ValueError("At least one instance of ORCA should be passed")

    check_type("atoms", atoms, Atoms)
    for calculator in calculator_list:
        check_type("calculator", calculator, ORCA)

    with open(parameter_file, "w") as file:
        for calculator_idx, calculator in enumerate(calculator_list):
            calculator.write_inputfiles(atoms, all_properties)
            file.write(f"{calculator.directory.resolve().as_posix()}\n")


def create_parameters_from_atoms_list(
    calculator: ORCA,
    atoms_list: list[Atoms] | Atoms,
    parameter_file: str | Path = "directories.txt",
) -> None:
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    n_atoms = len(atoms_list)
    if n_atoms == 0:
        raise ValueError("At least one instance of Atoms should be passed")

    check_type("calculator", calculator, ORCA)
    for atoms in atoms_list:
        check_type("atoms", atoms, Atoms)

    base_directory = Path(calculator.directory.resolve().as_posix())
    with open(parameter_file, "w") as file:
        for atoms_idx, atoms in enumerate(atoms_list):
            # Start from index 1 instead of 0 to have slurm array task ids
            # and directories line up.
            calculator.directory = base_directory / str(atoms_idx + 1)
            calculator.write_inputfiles(atoms, all_properties)
            file.write(f"{calculator.directory.resolve().as_posix()}\n")


def remove_finished_calculators(calculators: list[ORCA]) -> list[ORCA]:
    """Check which ORCA instances are finished, and remove those from the list.

    Args:
        calculators (list[ORCA]): list of ORCA instances with correct `directory`
            attributes.

    Returns:
        list[ORCA]: list of ORCA instances with the instances that have completed
            `calc.directory / "orca.out"` files removed.
    """
    to_keep = []
    for i, calculator in enumerate(calculators):
        output_file = Path(calculator.directory) / "orca.out"
        if not output_file.is_file():
            to_keep.append(i)
            continue

        result = read_orca_output(output_file)
        if not result.info.completed:
            to_keep.append(i)
    return [calculators[i] for i in to_keep]


def remove_finished_atoms(calculator: ORCA, atoms_list: list[Atoms]) -> list[Atoms]:
    """Remove Atoms instances from atoms_list that have already been calculated
    with this calculator. The order of Atoms in atoms_list is important here,
    as they are checked by index."""

    calc_directory = calculator.directory
    to_keep = []
    for i, atoms in enumerate(atoms_list):
        output_file = calc_directory / str(i + 1) / "orca.out"
        if not output_file.is_file():
            # Calculation not started yet
            to_keep.append(i)
            continue

        result = read_orca_output(output_file)
        if not result.info.completed:
            # Calculation did not finish
            to_keep.append(i)
            continue

        if not np.allclose(result.info.positions, atoms.positions) or not np.all(
            result.info.symbols == atoms.symbols
        ):
            # Not the same structure
            to_keep.append(i)
            continue
    return [atoms_list[i] for i in to_keep]


def read_orca_output(filepath: str | Path) -> Result:
    """Read output file generated by run of ORCA.

    Args:
        filepath (str | Path): path to `orca.out` file.

    Returns:
        Result: instance of Result with energies, forces and vibrations if
            available.
    """

    filepath = Path(filepath)
    with open(filepath) as file:
        lines = file.readlines()
    nlines = len(lines)

    completed = False
    for line in lines[::-1]:
        if "ORCA TERMINATED NORMALLY" in line:
            completed = True
            break

    if not completed:
        info = ResultInfo(path=filepath, completed=False, n_atoms=None)
        return Result(info=info)

    for line in lines:
        if "Number of atoms" in line:
            n_atoms = int(line.split()[-1])
            break

    for line in lines[::-1]:
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split()[-1]) * Hartree
            break

    forces = None
    for i, line in enumerate(lines[::-1]):
        if "CARTESIAN GRADIENT" in line:
            forces = np.asarray(
                [
                    [float(val) for val in grad_line.split()[3:]]
                    for grad_line in lines[nlines - i + 2 : nlines - i + 2 + n_atoms]
                ],
            ) * (-Hartree / Bohr)
            break

    frequencies = None
    for i, line in enumerate(lines[::-1]):
        if "VIBRATIONAL FREQUENCIES" in line:
            frequencies = [
                float(freq_line.split()[1])
                for freq_line in lines[nlines - i + 4 : nlines - i + 4 + 3 * n_atoms]
            ]
            break

    info = ResultInfo(path=filepath, completed=True, n_atoms=n_atoms)
    return Result(
        info=info,
        energy=energy,
        forces=forces,
        frequencies=frequencies,
    )
