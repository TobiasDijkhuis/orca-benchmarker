from __future__ import annotations

from .benchmark_orca import BenchmarkORCA


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
    def from_BenchmarkORCA(
        cls, benchmark_orca_instance: BenchmarkORCA
    ) -> BenchmarkFunctionals:
        benchmarks = benchmark_orca_instance.benchmarks
        config = benchmark_orca_instance.config
        return cls(benchmarks, config)

    @classmethod
    def from_file(
        cls, filepath: Union[str, Path], comment: str = "#"
    ) -> BenchmarkFunctionals:
        benchmark_orca_instance = BenchmarkORCA.from_file(
            filepath,
            BenchmarkFunctionals.valid_configuration_keys,
            comment=comment,
        )
        return cls.from_BenchmarkORCA(benchmark_orca_instance)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> BenchmarkFunctionals:
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
