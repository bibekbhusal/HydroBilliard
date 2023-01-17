import json
from abc import ABC


class BaseSimulationConfig(ABC):
    """
    Class with base configurations for simulation.
    """
    def __init__(self, **kwargs):
        self._experimental_particle_density = kwargs.get("_experimental_particle_density", 2e12)
        self._simulation_particle_density = kwargs.get("_simulation_particle_density", 400)
        self._source_drain_ratio = kwargs.get("_source_drain_ratio", 1.2)
        self._scattering_probability = kwargs.get("_scattering_probability", 0.1)
        self._add_probe_tip = kwargs.get("_add_probe_tip", False)
        self._probe_center_x = kwargs.get("_probe_center_x", 0.4)
        self._probe_center_y = kwargs.get("_probe_center_y", 3.7)
        self._re_inject_with_unit_energy = kwargs.get("_re_inject_with_unit_energy", False)
        self._step_size = kwargs.get("_step_size", 0.01)
        self._collision_distance = kwargs.get("_collision_distance", 0.05)
        self. _field_resolution = kwargs.get("_field_resolution", 0.05)
        self._time_steps = kwargs.get("_time_steps", 10000)
        self. _save_internal = kwargs.get("_save_internal", 2000)
        self._initial_condition_file = kwargs.get("_initial_condition_file", None)
        self._diffusive_edges = kwargs.get("_diffusive_edges", True)
        self._re_injection_probabilities = kwargs.get("_re_injection_probabilities", False)
        self._probe_method = kwargs.get("_probe_method", "L")
        self._transmission_probability = kwargs.get("_transmission_prob", 0.5)
        self._initialization_count = 500_000
        self._probe_tip_diameter = 1.5
        self._make_movie = kwargs.get("_make_movie", False)

    @property
    def experimental_particle_density(self):
        return self._experimental_particle_density

    @experimental_particle_density.setter
    def experimental_particle_density(self, value):
        self._experimental_particle_density = value

    @property
    def simulation_particle_density(self):
        return self._simulation_particle_density

    @simulation_particle_density.setter
    def simulation_particle_density(self, value):
        self._simulation_particle_density = value

    @property
    def source_drain_ratio(self):
        return self._source_drain_ratio

    @source_drain_ratio.setter
    def source_drain_ratio(self, value):
        self._source_drain_ratio = value

    @property
    def scattering_probability(self):
        return self._scattering_probability

    @scattering_probability.setter
    def scattering_probability(self, value):
        self._scattering_probability = value

    @property
    def add_probe_tip(self):
        return self._add_probe_tip

    @add_probe_tip.setter
    def add_probe_tip(self, value):
        self._add_probe_tip = value

    @property
    def probe_center_x(self):
        return self._probe_center_x

    @probe_center_x.setter
    def probe_center_x(self, value):
        self._probe_center_x = value

    @property
    def probe_center_y(self):
        return self._probe_center_y

    @probe_center_y.setter
    def probe_center_y(self, value):
        self._probe_center_y = value

    @property
    def re_inject_with_unit_energy(self):
        return self._re_inject_with_unit_energy

    @re_inject_with_unit_energy.setter
    def re_inject_with_unit_energy(self, value):
        self._re_inject_with_unit_energy = value

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        self._step_size = value

    @property
    def collision_distance(self):
        return self._collision_distance

    @collision_distance.setter
    def collision_distance(self, value):
        self._collision_distance = value

    @property
    def field_resolution(self):
        return self._field_resolution

    @field_resolution.setter
    def field_resolution(self, value):
        self._field_resolution = value

    @property
    def save_interval(self):
        return self._save_internal

    @save_interval.setter
    def save_interval(self, value):
        self._save_internal = value

    @property
    def initial_condition_file(self):
        return self._initial_condition_file

    @initial_condition_file.setter
    def initial_condition_file(self, value):
        self._initial_condition_file = value

    @property
    def diffusive_edges(self):
        return self._diffusive_edges

    @diffusive_edges.setter
    def diffusive_edges(self, value):
        self._diffusive_edges = value

    @property
    def re_injection_probabilities(self):
        return self._re_injection_probabilities

    @re_injection_probabilities.setter
    def re_injection_probabilities(self, value):
        self._re_injection_probabilities = value

    @property
    def probe_method(self):
        return self._probe_method

    @probe_method.setter
    def probe_method(self, value):
        self._probe_method = value

    @property
    def make_movie(self):
        return self._make_movie

    @make_movie.setter
    def make_movie(self, value):
        self._make_movie = value

    @property
    def initialization_count(self):
        return self._initialization_count

    @initialization_count.setter
    def initialization_count(self, value):
        self._initialization_count = value

    @property
    def probe_tip_diameter(self):
        return self._probe_tip_diameter

    @probe_tip_diameter.setter
    def probe_tip_diameter(self, value):
        self._probe_tip_diameter = value

    def to_json(self):
        return json.dumps(self.__dict__)

    def __str__(self):
        return self.to_json()
