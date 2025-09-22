from __future__ import annotations
from typing import Union, Optional
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read
import ase.units as units
from ase.io.orca import read_orca_output
from ase.io import ParseError
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.orca import ORCA, OrcaProfile
from subprocess import CalledProcessError, run
import os
from copy import deepcopy
from time import perf_counter
import yaml
from shutil import copyfile
from time import sleep
import re

from .utils import (
    get_job_state,
    wait_until_job_done,
    safe_calculate,
    slurm_calculator,
    replace_brackets_parenthesis_by_lowercase,
)
from .result import Result

_implement_child_string = "To be implemented by child class"


class BenchmarkORCA:
    def __init__(
        self,
        benchmarks: list[str],
        config: dict,
        valid_configuration_keys: list[str],
    ):
        self.config = {k.lower(): v for k, v in config.items()}

        self.valid_configuration_keys = [
            key.lower() for key in valid_configuration_keys
        ]

        for key in self.config:
            self._check_configuration_key(key)

        self.benchmarks = benchmarks

        self.energies = []
        self.forces = []
        self.durations = []

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        valid_configuration_keys: list[str],
        comment: str = "#",
    ):
        benchmarks = []
        config = {}

        with open(filepath, "r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if line == "":
                # skip empty lines
                continue
            if line[0] == comment:
                # skip lines starting with comment
                continue
            line = line.split(comment)[0].strip()  # remove all comments from line
            if ":" in line:
                line_split = line.split(":")
                param = line_split[0].lower()
                config[param] = line_split[1].strip()
                continue
            benchmarks.append(line)

        return cls(benchmarks, config, valid_configuration_keys)

    @classmethod
    def from_yaml(cls, filepath: str | Path, valid_configuration_keys: list[str]):
        with open(filepath, "r") as file:
            lines = file.readlines()
        lines = "\n".join(lines)
        parsed_yaml = yaml.safe_load(lines)
        parsed_yaml = {k.lower(): v for k, v in parsed_yaml.items()}

        config = {k.lower(): v for k, v in parsed_yaml["config"].items()}

        benchmarks = parsed_yaml["benchmarks"]

        return cls(benchmarks, config, valid_configuration_keys)

    def _check_configuration_key(self, parameter: str) -> None:
        if not isinstance(parameter, str):
            raise TypeError()
        if parameter not in self.valid_configuration_keys:
            msg = f"{parameter} is not a valid configuration parameter. Valid configuration parameters are: {self.valid_configuration_keys}"
            raise ValueError(msg)

    def get_ase_simpleinputs(self, benchmarks: str | list[str]) -> list[str]:
        raise NotImplementedError(_implement_child_string)

    def get_reference(self) -> str:
        if self.reference_is_set():
            raise ValueError("Reference not specified in input")
        return self.config["reference"]

    def reference_is_set(self) -> bool:
        return "reference" in self.config

    def get_config(self) -> dict:
        return self.config

    def get_valid_configuration_keys(self) -> list[str]:
        return self.valid_configuration_keys

    def get_total_orcablocks(
        self, orcablocks: str | list[str] | None = None, use_slurm: bool = False
    ) -> str:
        if isinstance(orcablocks, str):
            orcablocks = [orcablocks]
        if orcablocks is None:
            orcablocks = []
        if "orcablocks" in self.config:
            if self.config["orcablocks"] == "":
                return ""
            if isinstance(self.config["orcablocks"], str):
                config_orcablocks = ["%" + self.config["orcablocks"]]
            else:
                config_orcablocks = [
                    "%" + config for config in self.config["orcablocks"]
                ]
            orcablocks += config_orcablocks
        orcablocks = "\n".join(orcablocks)

        if not use_slurm:
            return orcablocks

        # Substitute the amount of cores in the ORCA input file
        # by NPROCS_HERE. This is (at job submission time) replaced by the amount
        # of cores used by the Slurm job.
        orcablocks = re.sub("nprocs [0-9]+", "nprocs NPROCS_HERE", orcablocks)
        return orcablocks

    def get_benchmark_calculators(
        self,
        profile: OrcaProfile,
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str | list[str] | None = None,
        use_slurm: bool = False,
    ) -> list[ORCA]:
        calculators = []

        orcablocks = self.get_total_orcablocks(orcablocks, use_slurm=use_slurm)
        simpleinputs = self.get_ase_simpleinputs(self.benchmarks)
        for i, benchmark in enumerate(self.benchmarks):
            benchmark_dir = Path(directory) / benchmark.replace(" ", "_").replace(
                "/", ""
            )

            benchmark_dir = Path(
                replace_brackets_parenthesis_by_lowercase(str(benchmark_dir))
            )

            calculator = ORCA(
                orcasimpleinput=simpleinputs[i],
                orcablocks=orcablocks,
                profile=profile,
                directory=benchmark_dir,
                charge=charge,
                mult=mult,
            )
            calculators.append(calculator)
        return calculators

    def get_reference_calculator(
        self,
        profile: OrcaProfile,
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str = "",
        use_slurm: bool = False,
    ) -> ORCA:
        simpleinput = self.get_ase_simpleinputs(self.config["reference"])

        rel_path = "reference_" + self.config["reference"].replace(" ", "_").replace(
            "/", ""
        )

        rel_path = Path(replace_brackets_parenthesis_by_lowercase(rel_path))

        orcablocks = self.get_total_orcablocks(orcablocks, use_slurm=use_slurm)
        reference_dir = Path(directory) / rel_path
        calculator = ORCA(
            orcasimpleinput=simpleinput,
            orcablocks=orcablocks,
            profile=profile,
            directory=reference_dir,
            charge=charge,
            mult=mult,
        )
        return calculator

    def run(
        self,
        atoms: Atoms,
        profile: OrcaProfile,
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str | list[str] | None = None,
        force: bool = False,
        use_slurm: bool = False,
        slurm_script_path: str | Path | None = None,
    ):
        calculator_kwargs = {
            "charge": charge,
            "mult": mult,
            "directory": directory,
            "orcablocks": orcablocks,
            "use_slurm": use_slurm,
        }
        calculators = self.get_benchmark_calculators(
            profile,
            **calculator_kwargs,
        )

        calculate_kwargs = {
            "force": force,
            "use_slurm": use_slurm,
            "slurm_script_path": slurm_script_path,
        }
        for i, calculator in enumerate(calculators):
            result = safe_calculate(
                calculator,
                atoms,
                **calculate_kwargs,
            )
            self.energies.append(result.energy)
            self.forces.append(result.forces)
            self.durations.append(result.duration)

        if not self.reference_is_set():
            print("Reference was not set. Finishing.")

        reference = self.get_reference_calculator(
            profile,
            **calculator_kwargs,
        )
        result = safe_calculate(
            reference,
            atoms,
            **calculate_kwargs,
        )
        self.reference_energy = result.energy
        self.reference_forces = result.forces
        self.reference_duration = result.duration

    def get_energy_deviations(self) -> list[float]:
        if not self.energies:
            raise ValueError(
                "Energy list was empty. Execute BenchmarkORCA.run() first."
            )
        if not self.reference_is_set():
            raise ValueError(
                "Reference was not set in configuration. Cannot calculate relative energies."
            )
        if self.reference_energy is None:
            raise ValueError(
                "Reference energy is None. Something was wrong with its input."
            )
        return [
            energy - self.reference_energy if energy is not None else None
            for energy in self.energies
        ]


class BenchmarkFunctionals(BenchmarkORCA):
    valid_configuration_keys = [
        "basisdft",
        "basiscc",
        "reference",
        "orcablocks",
        "grid",
        "parameters",
    ]

    def __init__(self, benchmarks, config):
        super().__init__(
            benchmarks,
            config,
            BenchmarkFunctionals.valid_configuration_keys,
        )
        self._add_basis_sets()

    @classmethod
    def from_BenchmarkORCA(cls, benchmark_orca_instance: BenchmarkORCA):
        benchmarks = benchmark_orca_instance.benchmarks
        config = benchmark_orca_instance.config
        return cls(benchmarks, config)

    @classmethod
    def from_file(cls, filepath: Union[str, Path], comment: str = "#"):
        benchmark_orca_instance = BenchmarkORCA.from_file(
            filepath,
            BenchmarkFunctionals.valid_configuration_keys,
            comment=comment,
        )
        return cls.from_BenchmarkORCA(benchmark_orca_instance)

    @classmethod
    def from_yaml(cls, filepath: str | Path):
        benchmark_orca_instance = BenchmarkORCA.from_yaml(
            filepath, BenchmarkFunctionals.valid_configuration_keys
        )
        return cls.from_BenchmarkORCA(benchmark_orca_instance)

    def get_ase_simpleinputs(
        self,
        benchmarks: list[str] | str,
    ) -> list[str]:
        if isinstance(benchmarks, list):
            return [self.get_ase_simpleinputs(benchmark) for benchmark in benchmarks]
        elif isinstance(benchmarks, str):
            simpleinput = benchmarks
            if "grid" in self.config:
                simpleinput += " " + self.config["grid"]
            if (
                "forces" in self.config["parameters"]
                and "energy" in self.config["parameters"]
            ):
                simpleinput += " EnGrad"
                if "cc" in benchmarks.lower():
                    # No analytical gradients for CC calculations
                    simpleinput += " NumGrad"
            elif "energy" in self.config["parameters"]:
                simpleinput += " Energy"
            else:
                raise ValueError(self.config["parameters"])
            return simpleinput
        else:
            raise ValueError()

    def _add_basis_sets(self) -> None:
        for i, functional in enumerate(self.benchmarks):
            if "cc" in functional.lower():
                # coupled cluster
                self.benchmarks[i] += " " + self.config["basiscc"]
            else:
                self.benchmarks[i] += " " + self.config["basisdft"]

        if "cc" in self.config["reference"].lower():
            # coupled cluster
            self.config["reference"] += " " + self.config["basiscc"]
        else:
            self.config["reference"] += " " + self.config["basisdft"]
