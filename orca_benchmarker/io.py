from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import all_properties
from ase.units import Hartree, Bohr
from orca_benchmarker.result import Result, ResultInfo
from typing import Any

import numpy as np
from pathlib import Path
from textwrap import dedent

from orca_benchmarker.utils import check_type
from ase.calculators.orca import ORCA


def read_orca_generated_trajectory(trajectory_path: str | Path) -> list[Atoms]:
    """Read ORCA generated xyz trajectory.

    Similar to ase.io.read, except also reads the energy in the comment line.

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
    sbatch_kwargs: dict[str, Any] | None = None,
    orca_path: str | Path = "orca",
    module_info: str | list[str] | None = None,
) -> None:
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

        if module_info is not None:
            if isinstance(module_info, str):
                module_info = [module_info]
            for step in module_info:
                file.write(f"module {step}\n")

        file.write(
            dedent(f"""

                calculation_dirs_file={parameter_file}
                calculation_dir=$(sed "${{SLURM_ARRAY_TASK_ID}}q;d" $calculation_dirs_file)
                calculation_dir=$(realpath $calculation_dir)
            
                # Replace NPROCS_HERE by the amount of tasks per node.
                sed -i -e "s/NPROCS_HERE/${{SLURM_NTASKS_PER_NODE}}/g" ${{calculation_dir}}/orca.inp
                
                # Always run stuff in local scratch on node for job and not in ${{HOME}} to avoid NFS network slowness
                scratch_dir=$SCRATCH
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
                rm orca.densities*
                rm orca.gbw

                # Copy relevant things to calculation directory
                cp orca.* ${{calculation_dir}}/.

                wait""")
        )


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
            file.write(f"{calculator.directory}\n")


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

    base_directory = Path(calculator.directory)
    with open(parameter_file, "w") as file:
        for atoms_idx, atoms in enumerate(atoms_list):
            calculator.directory = base_directory / str(atoms_idx)
            calculator.write_inputfiles(atoms, all_properties)
            file.write(f"{calculator.directory}\n")


def remove_finished_calculators(calculators: list[ORCA]) -> list[ORCA]:
    to_keep = []
    for i, calculator in enumerate(calculators):
        output_file = calculator.directory / "orca.out"
        if not output_file.is_file():
            to_keep.append(i)
            continue
        result = read_orca_output(output_file)
        if not result.info.completed:
            to_keep.append(i)
    return [calculators[i] for i in to_keep]


def read_orca_output(filepath: str | Path) -> Result:
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
        info = ResultInfo(path=filepath, completed=False)
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
            ) * (Hartree / Bohr)
            break

    frequencies = None
    for i, line in enumerate(lines[::-1]):
        if "VIBRATIONAL FREQUENCIES" in line:
            frequencies = [
                float(freq_line.split()[1])
                for freq_line in lines[nlines - i + 4 : nlines - i + 4 + 3 * n_atoms]
            ]
            break

    info = ResultInfo(path=filepath, completed=True)
    return Result(
        info=info,
        n_atoms=n_atoms,
        energy=energy,
        forces=forces,
        frequencies=frequencies,
    )
