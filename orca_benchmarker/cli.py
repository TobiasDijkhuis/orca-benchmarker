import typer
from orca_benchmarker.io import read_orca_generated_trajectory, read_orca_output
from ase.visualize import view
from pathlib import Path

app = typer.Typer()


@app.command()
def traj(filepath: Path) -> None:
    """Visualize an ORCA-generated XYZ trajectory"""
    frames = read_orca_generated_trajectory(filepath)
    view(frames, block=True)


@app.command()
def result(filepath: Path) -> None:
    """Print some information about an ORCA calculation"""
    result = read_orca_output(filepath)
    print(f"\tCompleted: {result.info.completed}")
    print(f"\tEnergy: {result.energy:.2f} eV")

    if result.frequencies is None:
        return
    print(f"\tZero-point energy: {result.zero_point_energy:.2f} eV")
    print(f"\tZP-corrected energy: {result.zero_point_corrected_energy:.2f} eV")

    if result.imaginary_frequency is None:
        return
    print(f"\tImaginary frequency: {result.imaginary_frequency:.2f} cm-1")
    print(f"\tCross-over temperature: {result.get_crossover_temperature():.2f} K")
