from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ase.atoms import Atoms
from ase.calculators.calculator import all_changes, all_properties
from ase.calculators.orca import ORCA, OrcaProfile

from orca_benchmarker.io import (
    create_parameters_from_calculator_list,
    read_orca_output,
    remove_finished_calculators,
    write_orca_array_submission_script,
)
from orca_benchmarker.result import Result
from orca_benchmarker.utils import (
    convert_benchmark_to_dir,
    extract_sbatch_kwargs_from_orcablocks,
    get_orca_path,
    is_slurm_active,
    start_job,
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
        return "\n".join(orcablocks)

    def get_benchmark_calculators(
        self,
        profile: OrcaProfile | None = None,
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str | list[str] | None = None,
    ) -> list[ORCA]:
        calculators = []

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
        charge: int = 0,
        mult: int = 1,
        directory: Path | str = ".",
        orcablocks: str | list[str] | None = None,
        force: bool = False,
        orca_path: str | Path | None = None,
        sbatch_kwargs: dict[str, Any] | None = None,
        environment_commands: str | list[str] | None = None,
    ):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            self.directory.mkdir(parents=True)

        orcablocks = self.get_total_orcablocks(orcablocks)

        if is_slurm_active():
            sbatch_kwargs_from_blocks, orcablocks = (
                extract_sbatch_kwargs_from_orcablocks(orcablocks)
            )
            if sbatch_kwargs is None:
                sbatch_kwargs = sbatch_kwargs_from_blocks
            else:
                sbatch_kwargs.update(sbatch_kwargs_from_blocks)

        orca_path = get_orca_path(orca_path)
        calculator_kwargs = {
            "charge": charge,
            "mult": mult,
            "directory": self.directory,
            "orcablocks": orcablocks,
            "profile": OrcaProfile(orca_path.as_posix()),
        }
        calculators = self.get_benchmark_calculators(
            **calculator_kwargs,
        )

        if self.reference_is_set():
            # Insert at 0 position, to start the reference calculation first.
            # That will take the longest, so we have to wait the least.
            calculators.insert(0, self.get_reference_calculator(**calculator_kwargs))

        if not force:
            calculators = remove_finished_calculators(calculators)

        if not calculators:
            # All calculators were finished, none left to calculate
            return

        if is_slurm_active():
            parameter_file = self.directory / "directories.txt"
            submission_file = self.directory / "run.sh"
            create_parameters_from_calculator_list(
                calculators, atoms, parameter_file=parameter_file
            )
            write_orca_array_submission_script(
                parameter_file=parameter_file,
                output_file=submission_file,
                orca_path=orca_path,
                sbatch_kwargs=sbatch_kwargs,
                environment_commands=environment_commands,
            )
            start_job(submission_file, wait_until_done=True)
        else:
            for i, calculator in enumerate(calculators):
                calculator.calculate(atoms, all_properties, all_changes)

    def read_results(
        self, directory: str | Path | None
    ) -> tuple[list[Result], Result | None]:
        """Get all results"""

        if directory is None:
            if hasattr(self, "directory"):
                directory = self.directory
            else:
                raise ValueError()

        benchmark_results = []
        for benchmark_dir in self.benchmark_dirs:
            benchmark_results.append(
                read_orca_output(Path(directory) / benchmark_dir / "orca.out")
            )

        if not self.reference_is_set():
            return benchmark_results, None

        reference_result = read_orca_output(
            Path(directory) / self.reference_dir / "orca.out"
        )
        return benchmark_results, reference_result
