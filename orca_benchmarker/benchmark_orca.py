from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from subprocess import CalledProcessError, run
from time import perf_counter, sleep
from typing import Optional, Union

import ase.units as units
import yaml
from ase.atoms import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import ParseError, read
from ase.io.orca import read_orca_output

from .result import Result
from .utils import (
    convert_benchmark_to_dir,
    get_job_state,
    replace_brackets_parenthesis_by_lowercase,
    safe_calculate,
    slurm_calculator,
    wait_until_job_done,
)

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
        self.benchmark_dirs = [
            convert_benchmark_to_dir(benchmark) for benchmark in self.benchmarks
        ]

        self.reference_dir = convert_benchmark_to_dir(
            self.config["reference"], reference=True
        )

        self.job_ids = []

        self.energies = []
        self.forces = []
        self.durations = []

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        valid_configuration_keys: list[str],
        comment: str = "#",
    ) -> BenchmarkORCA:
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
    def from_yaml(
        cls, filepath: str | Path, valid_configuration_keys: list[str]
    ) -> BenchmarkORCA:
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

    def get_total_orcablocks(self, orcablocks: str | list[str] | None = None) -> str:
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

        if not self.use_slurm:
            return orcablocks

        # Substitute the amount of cores in the ORCA input file
        # by NPROCS_HERE. This is (at job submission time) replaced by the amount
        # of cores used by the Slurm job.
        orcablocks = re.sub("nprocs [0-9]+", "nprocs NPROCS_HERE", orcablocks)
        return orcablocks

    def get_benchmark_calculators(
        self,
        profile: OrcaProfile | None = None,
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str | list[str] | None = None,
    ) -> list[ORCA]:
        calculators = []

        orcablocks = self.get_total_orcablocks(orcablocks)
        simpleinputs = self.get_ase_simpleinputs(self.benchmarks)
        for i, benchmark in enumerate(self.benchmarks):
            calculator = ORCA(
                orcasimpleinput=simpleinputs[i],
                orcablocks=orcablocks,
                profile=profile,
                directory=Path(directory) / self.benchmark_dirs[i],
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
    ) -> ORCA:
        simpleinput = self.get_ase_simpleinputs(self.config["reference"])

        orcablocks = self.get_total_orcablocks(orcablocks)
        calculator = ORCA(
            orcasimpleinput=simpleinput,
            orcablocks=orcablocks,
            profile=profile,
            directory=Path(directory) / self.reference_dir,
            charge=charge,
            mult=mult,
        )
        return calculator

    def run(
        self,
        atoms: Atoms,
        profile: OrcaProfile | None = None,
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str | list[str] | None = None,
        force: bool = False,
        use_slurm: bool = False,
        slurm_script_path: str | Path | None = None,
    ):
        self.use_slurm = use_slurm

        calculator_kwargs = {
            "charge": charge,
            "mult": mult,
            "directory": directory,
            "orcablocks": orcablocks,
            "profile": profile,
        }
        calculators = self.get_benchmark_calculators(
            **calculator_kwargs,
        )

        calculate_kwargs = {
            "force": force,
            "use_slurm": self.use_slurm,
            "slurm_script_path": slurm_script_path,
        }
        for i, calculator in enumerate(calculators):
            result = safe_calculate(
                calculator,
                atoms,
                **calculate_kwargs,
            )
            self.job_ids.append(result)

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
        self.job_ids.append(result)

    def wait(self) -> None:
        """Wait until all calculations are done. Does nothing if slurm is not being used."""
        if not self.use_slurm:
            return

        if all([job is None for job in self.job_ids]):
            return

        jobs = [job for job in self.job_ids if job is not None]

        try:
            from slurm_manager.job import SlurmJob
            from slurm_manager.manager import SlurmJobManager

            manager = SlurmJobManager()
            manager.add_jobs(jobs)
            manager.wait()
        except (ImportError, ModuleNotFoundError):
            [wait_until_job_done(job_id) for job_id in jobs]

    def read_results(
        self, directory: str | Path = "."
    ) -> tuple[list[Result], Result | None]:
        """Get all results"""
        benchmark_results = []
        for benchmark_dir in self.benchmark_dirs:
            benchmark_results.append(
                Result.from_orca_output_file(
                    Path(directory) / benchmark_dir / "orca.out"
                )
            )

        if not self.reference_is_set():
            return benchmark_results, None

        reference_result = Result.from_orca_output_file(
            Path(directory) / self.reference_dir / "orca.out"
        )
        return benchmark_results, reference_result

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
