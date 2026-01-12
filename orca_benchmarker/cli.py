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
    read_orca_output(filepath).print_summary()
