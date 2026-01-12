from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from functools import cached_property
import numpy as np
from orca_benchmarker.constants import eV_per_reciprocal_cm
from scipy.constants import speed_of_light, centi, pi, Boltzmann, hbar
from math import log

from typing import Literal


@dataclass
class ResultInfo:
    path: str | Path
    completed: bool

    @cached_property
    def duration(self) -> float | None:
        if not self.completed:
            return None

        with open(self.path) as file:
            lines = file.readlines()
        for line in lines[::-1]:
            if "Sum of individual times" in line:
                return float(line.split("...").split()[0])


@dataclass
class Result:
    info: ResultInfo
    n_atoms: int | None = None
    energy: float | None = None
    forces: np.ndarray | None = None
    frequencies: list | None = None

    """A convenient way of keeping track of the output of an ORCA calculation.

    Attributes:
        info: ResultInfo
            information about calculation
        n_atoms: int | None, default = None
            Number of atoms in structure
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
    def zero_point_corrected_energy(self) -> float | None:
        return self.energy + self.zero_point_energy

    @property
    def zero_point_energy(self) -> float | None:
        if self.frequencies is None:
            return None
        zero_point_energy = 0.0
        for freq in self.frequencies:
            if freq <= 0:
                continue
            zero_point_energy += 0.5 * freq * eV_per_reciprocal_cm
        return zero_point_energy

    @property
    def imaginary_frequency(self) -> float | None:
        # TODO: Check if there is only one negative frequency
        # and perhaps return more if there are multiple.
        for freq in self.frequencies:
            if freq < 0:
                return abs(freq)

    def get_crossover_temperature(
        self,
        method: Literal["standard", "fermann"] = "standard",
        barrier_height: float | None = None,
    ) -> float | None:
        imaginary_frequency = (
            abs(self.imaginary_frequency) / centi * speed_of_light * 2.0 * pi
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
