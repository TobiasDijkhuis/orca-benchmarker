from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import ParseError
from ase.io.orca import read_orca_output


@dataclass
class Result:
    filepath: str | Path
    completed: bool
    duration: float | None
    energy: float | None
    forces: np.ndarray | None

    @classmethod
    def from_orca_output_file(cls, filepath: str | Path) -> Result:
        try:
            output = read_orca_output(filepath, index=-1)
            energy = output.get_potential_energy()
            try:
                forces = output.get_forces()
            except PropertyNotImplementedError:
                forces = None

            with open(filepath, "r") as file:
                lines = file.readlines()
                for line in lines[::-1]:
                    if "Sum of individual times" in line:
                        duration = float(line.split()[-5])
                        break
            completed = True
        except ParseError as e:
            completed = False
            energy, forces, duration = None, None, None

        return cls(
            completed=completed,
            filepath=Path(filepath),
            duration=duration,
            energy=energy,
            forces=forces,
        )
