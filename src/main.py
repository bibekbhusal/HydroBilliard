import json
from enum import Enum
import numpy as np

import click.types
import typer
from pathlib import Path
from datetime import datetime

from configs.tesla_valve_simulation_config import TeslaValveSimulationConfig
from simulators.full_circle_simulator import FullCircleSimulator
from simulators.tesla_valve_simulator import TeslaValveSimulator
from utils.miscellaneous import highlighted_str

from configs.base_simulation_config import BaseSimulationConfig
from configs.full_circle_simulation_config import FullCircleSimulationConfig

main = typer.Typer()

TODAY = datetime.now()
CONFIG_DIR = Path.home() / ".HydroBilliards"
FULL_CIRCLE_CONFIG = CONFIG_DIR / "full_circle.config"


class SupportedSimulation(str, Enum):
    FullCircle = "full_circle"
    TeslaValve = "tesla_valve"


def _load_full_circle_config():
    config_path = Path(FULL_CIRCLE_CONFIG)
    if config_path.exists():
        with config_path.open('r') as file:
            return FullCircleSimulationConfig(**json.load(file))


def _load_tesla_valve_config():
    pass


# Defines a list of all supported simulations and their corresponding configuration file.
# These configuration files will be loaded into memory when the application starts.
SUPPORTED_CONFIG_CREATORS = {
    SupportedSimulation.FullCircle.value: lambda: _load_full_circle_config(),
    SupportedSimulation.TeslaValve.value: lambda: _load_tesla_valve_config()
}

_loaded_configs = None


def _load_configs(forced: bool = False):
    # To be able to directly retrieve and assign to _loaded_configs defined outside
    global _loaded_configs
    if not forced and _loaded_configs is not None:
        return _loaded_configs

    config_map = {}
    for config_key, config_creator_func in SUPPORTED_CONFIG_CREATORS.items():
        config_map[config_key] = config_creator_func()
    _loaded_configs = config_map


def base_init(config: BaseSimulationConfig):
    config.experimental_particle_density = typer.prompt(
        highlighted_str("Experimental Particle Density", prefix="Please enter"),
        type=float
    )
    config.simulation_particle_density = typer.prompt(
        highlighted_str("Simulation Particle Density", prefix="Please enter"),
        type=float
    )
    config.source_drain_ratio = typer.prompt(
        highlighted_str("Source Drain Ratio", prefix="Please enter"),
        type=float
    )
    config.scattering_probability = typer.prompt(
        highlighted_str("Scattering Probability", prefix="Please enter"),
        type=float
    )
    config.add_probe_tip = typer.confirm(
        highlighted_str("Probe Tips", prefix="Would you like to add")
    )
    if config.add_probe_tip:
        config.probe_center_x = typer.prompt(
            highlighted_str("Probe Tip Location X", prefix="Please enter"),
            type=float
        )
        config.probe_center_y = typer.prompt(
            highlighted_str("Probe Tip Location Y", prefix="Please enter"),
            type=float
        )
    config.re_inject_with_unit_energy = typer.confirm(
        highlighted_str("re-inject with unit energy", "Would you like to")
    )
    config.step_size = typer.prompt(
        highlighted_str("Step Size", prefix="Please enter"),
        type=float
    )
    config.collision_distance = typer.prompt(
        highlighted_str("Collision Distance", prefix="Please enter"),
        type=float
    )
    config.field_resolution = typer.prompt(
        highlighted_str("Field Resolution", prefix="Please enter"),
        type=float
    )
    config.save_interval = typer.prompt(
        highlighted_str("Save Interval", prefix="Please enter"),
        type=int
    )
    use_initial_condition_file = typer.confirm(
        highlighted_str("initial condition file", prefix="Would you like to use")
    )
    if use_initial_condition_file:
        config.initial_condition_file = typer.prompt(
            highlighted_str("Initial Condition File", prefix="Please provide location of", suffix=", if any"),
            type=click.types.Path(exists=True, dir_okay=False, readable=True, resolve_path=True)
        )
    config.diffusive_edges = typer.confirm(
        highlighted_str("make a movie", prefix="Would you like to")
    )
    config.re_injection_probabilities = typer.confirm(
        highlighted_str("generate re-injection probabilities", "Would you like to")
    )
    config.probe_method = typer.prompt(
        highlighted_str("Probe Method", "Please choose a"),
        default="L",
        type=click.types.Choice(["L", "S"])  # By length, By simulation
    )


def full_circle_init():
    if FULL_CIRCLE_CONFIG.exists():
        typer.echo("Base configuration file exists.")
        old_config = FULL_CIRCLE_CONFIG.with_suffix(".old-" + TODAY.strftime("%Y%m%d-%H%M%S"))
        typer.echo("Copying previous configuration to " + highlighted_str(str(old_config)))
        FULL_CIRCLE_CONFIG.replace(old_config)

    full_circle_config = FullCircleSimulationConfig()
    base_init(full_circle_config)

    full_circle_config.constriction_width = typer.prompt(
        highlighted_str("Constriction Width", prefix="Please enter"),
        type=float
    )
    full_circle_config.diameter = typer.prompt(
        highlighted_str("Diameter", prefix="Please enter"),
        type=float
    )
    full_circle_config.inject_height = typer.prompt(
        highlighted_str("Inject height", prefix="Please enter"),
        type=float
    )
    full_circle_config.inject_width = typer.prompt(
        highlighted_str("Inject width", prefix="Please enter"),
        type=float
    )
    typer.echo(f"Saving config {full_circle_config.to_json()} to {FULL_CIRCLE_CONFIG}")
    FULL_CIRCLE_CONFIG.write_text(full_circle_config.to_json())


@main.command()
def configure(sim_type: SupportedSimulation =
              typer.Option(...,
                           help="Enter simulation type, e.g., full_circle, etc.",
                           case_sensitive=False)):
    typer.confirm(highlighted_str("Starting (re)initialization.", suffix="Are you sure?"), abort=True)
    Path(CONFIG_DIR).mkdir(exist_ok=True)
    if sim_type == SupportedSimulation.FullCircle:
        full_circle_init()
    else:
        raise NotImplementedError(f"Configuration of given simulation type ({sim_type}) is not supported.")


# TODO: Allow dynamic update of config value
@main.command()
def simulate(sim_type: SupportedSimulation =
             typer.Option(...,
                          help="Enter simulation type, e.g., full_circle, etc.",
                          case_sensitive=False),
             temperature: float = typer.Option(
                 ...,
                 help="Please provide temperature (in K) to simulate.",
                 min=0.0,
                 max=1000.0
             ),
             time_steps: int = typer.Option(
                 ...,
                 min=1,
                 help="Please provide number of steps to run."
             ),
             output_path: str = typer.Option(
                 ...,
                 exists=True,
                 dir_okay=True,
                 file_okay=False,
                 writable=True,
                 resolve_path=True,
                 path_type=str,
                 help="Location to save results."
             )):
    typer.echo(f"Selected simulation type: {sim_type}. Temperature[{temperature}]")

    simulation_config = _loaded_configs[sim_type]
    if simulation_config is None:
        if sim_type == SupportedSimulation.FullCircle:
            simulation_config = FullCircleSimulationConfig()
        elif sim_type == SupportedSimulation.TeslaValve:
            simulation_config = TeslaValveSimulationConfig()

    simulator = None
    if sim_type == SupportedSimulation.FullCircle:
        simulator = FullCircleSimulator(temperature, simulation_config)
    elif sim_type == SupportedSimulation.TeslaValve:
        simulator = TeslaValveSimulator(temperature, simulation_config)

    simulator.calc_scatter_probability()
    simulator.update_body()
    simulator.calc_area()

    if simulation_config.scattering_probability == 0.0:
        simulator.update_scattering_probability(simulation_config.scattering_probability)

    if simulator.diffusive:
        simulator.set_diffusive_edges()
    else:
        simulator.set_mirror_edges()

    simulator.update_particles()

    print(f"Starting simulation with Temp: {temperature}")

    output_dir = Path(output_path) / f"{TODAY.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator.run_and_save(time_steps, output_dir.resolve())


if __name__ == "__main__":
    _load_configs(True)
    main()
