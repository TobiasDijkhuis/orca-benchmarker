from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import log
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.constants import Boltzmann, centi, hbar, pi, speed_of_light

from orca_benchmarker.constants import eV_per_reciprocal_cm
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms


@dataclass
class ResultInfo:
    path: str | Path
    completed: bool
    n_atoms: int | None

    @cached_property
    def duration(self) -> float | None:
        if not self.completed:
            return None

        with open(self.path) as file:
            lines = file.readlines()
        for line in lines[::-1]:
            if "Sum of individual times" in line:
                return float(line.split("...")[1].split()[0])

    @property
    def positions(self) -> np.ndarray:
        with open(self.path) as file:
            lines = file.readlines()

        nlines = len(lines)
        for i, line in enumerate(lines[::-1]):
            if not "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                continue
            return np.asarray(
                [
                    [float(val) for val in line.split()[1:]]
                    for line in lines[nlines - i + 1 : nlines - i + 1 + self.n_atoms]
                ]
            )

        raise ValueError(f"Could not infer positions from output file {self.path}")

    @property
    def symbols(self) -> list[str]:
        with open(self.path) as file:
            lines = file.readlines()

        nlines = len(lines)
        for i, line in enumerate(lines[::-1]):
            if not "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                continue
            return [
                line.split()[0]
                for line in lines[nlines - i + 1 : nlines - i + 1 + self.n_atoms]
            ]

        raise ValueError(f"Could not infer symbols from output file {self.path}")


@dataclass
class Result:
    info: ResultInfo
    energy: float | None = None
    forces: np.ndarray | None = None
    frequencies: list | None = None

    """A convenient way of keeping track of the output of an ORCA calculation.

    Attributes:
        info: ResultInfo
            information about calculation
        energy: float | None, default = None
            Single point energy in eV. None if not completed
        forces: np.ndarray | None, default = None
            Forces on each atom in eV/Angstrom. None if not completed or not
                calculated
        frequencies: list | None, default = None
            Vibrational frequencies in cm-1. None if not completed or not
                calculated.
    """

    @property
    def zero_point_corrected_energy(self) -> float:
        return self.energy + self.zero_point_energy

    @property
    def zero_point_energy(self) -> float:
        """The zero-point-energy in eV"""
        if self.frequencies is None:
            raise ValueError(f"No frequencies in Result of path {self.info.path}")
        zero_point_energy = 0.0
        for freq in self.frequencies:
            if freq <= 0:
                continue
            zero_point_energy += 0.5 * freq * eV_per_reciprocal_cm
        return zero_point_energy

    @property
    def imaginary_frequency(self) -> float | None:
        """The imaginary frequency, if it exists, in cm-1."""
        # TODO: Check if there is only one negative frequency
        # and perhaps return more if there are multiple.
        if self.frequencies is None:
            raise ValueError(f"No frequencies in Result of path {self.info.path}")
        for freq in self.frequencies:
            if freq < 0:
                return abs(freq)

    def get_crossover_temperature(
        self,
        method: Literal["standard", "fermann"] = "standard",
        barrier_height: float | None = None,
    ) -> float:
        try:
            imaginary_frequency = self.imaginary_frequency
        except ValueError as e:
            raise ValueError(
                "Cannot determine cross-over temperature if no frequencies were calculated for this Result"
            ) from e
        if imaginary_frequency is None:
            raise ValueError(
                f"No imaginary frequency in Result of path {self.info.path}"
            )

        # Convert imaginary frequency to units of radians s-1 (angular frequency)
        imaginary_frequency = (
            abs(imaginary_frequency) / centi * speed_of_light * 2.0 * pi
        )

        method = method.lower()
        if method == "standard":
            # https://iopscience.iop.org/article/10.1088/0022-3719/20/24/005
            return hbar * imaginary_frequency / (2.0 * pi * Boltzmann)
        elif method == "fermann":
            if barrier_height is None:
                raise ValueError("Barrier height necessary for Fermann method")
            # https://pubs.aip.org/aip/jcp/article/112/15/6787/183830
            return (hbar * imaginary_frequency * barrier_height / Boltzmann) / (
                2.0 * pi * barrier_height - hbar * imaginary_frequency * log(2.0)
            )
        else:
            raise NotImplementedError(
                f"Method of calculating crossover temperature '{method}' is not implemented. Options are: ['standard', 'fermann']"
            )

    def print_summary(self) -> None:
        print(f"\tCompleted: {self.info.completed}")
        if not self.completed:
            return
        print(f"\tEnergy: {self.energy:.2f} eV")

        if self.frequencies is None:
            return
        print(f"\tZero-point energy: {self.zero_point_energy:.2f} eV")
        print(f"\tZP-corrected energy: {self.zero_point_corrected_energy:.2f} eV")

        if self.imaginary_frequency is None:
            return
        print(f"\tImaginary frequency: {self.imaginary_frequency:.2f} cm-1")
        print(f"\tCross-over temperature: {self.get_crossover_temperature():.2f} K")

    def to_ase_atoms(self) -> Atoms:
        """Convert Result instance to ASE Atoms instance, including
        potential energy and forces in SinglePointCalculator.

        Returns:
            atoms (Atoms): ASE Atoms instance.
        """
        atoms = Atoms(symbols=self.info.symbols, positions=self.info.positions)
        if not self.info.completed:
            return atoms

        results = {"energy": self.energy}
        if self.forces is not None:
            results["forces"] = self.forces

        atoms.calc = SinglePointCalculator(atoms, **results)
        return atoms
