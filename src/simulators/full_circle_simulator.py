import json

import numpy as np

from configs.full_circle_simulation_config import FullCircleSimulationConfig
from simulators.simulator_base import SimulatorBase

_SOURCE_EDGE = 2
_DRAIN_EDGE = 1
_MIRROR_EDGE = 0
_ROUGH_EDGE = -1
_FALSE_EDGE = -2


class FullCircleSimulator(SimulatorBase):
    """
    Simulates hydrodynamic flow of dirac electrons through a constriction and into a circular device. The entire arc
    of the circle is the drain and a source injects through a constriction at the arc's radial center.
    -> Does this mean the source is surrounded by a circle of drains?
    """

    def __init__(self, temperature: float, config: FullCircleSimulationConfig):
        # self._source_drain_ratio = config.source_drain_ratio

        # TODO: Can we make update_body() only once at the end of setting this parameters
        self._set_diameter(config.diameter)
        self._set_constriction_width(config.constriction_width)
        self._set_injector_shape(config.inject_width, config.inject_height)

        super().__init__(temperature, config)

        self._roll_index = 17

        # TODO - Why 1.05?
        # TODO: Set-diameter above sets _box_l = diameter. But then this sets 1.05 * diameter
        # self._set_boundary_length(self._diameter * 1.05)

        self._counts_per_snap_shot = 500

    def _set_diameter(self, diameter):
        self._box_l = diameter
        self._diameter = diameter
        # self.update_body()

    def _set_constriction_width(self, constriction_width):
        self._constriction_width = constriction_width
        # self.update_body()

    def _set_injector_shape(self, injector_width, injector_height):
        self._injector_width = injector_width
        self._injector_height = injector_height
        # self.update_body()

    def update_body(self):
        print(f"Updating body")
        radius = self._diameter / 2.0
        x_c = self._constriction_width / 2.0
        x_i = self._injector_width / 2.0
        y_i = self._injector_height

        print(f"Radius: {radius}, X_C: {x_c} X_I: {x_i} Y_I: {y_i}")

        thetas = np.linspace(1.4 * np.pi, -0.4 * np.pi, 25)

        self.border_x = np.cos(thetas) * radius
        self.border_y = np.sin(thetas) * radius

        self.border_x = np.append(self.border_x, np.array([x_c, x_i, -x_i, -x_c, self.border_x[0]]))
        self.border_y = np.append(self.border_y, np.array([0, -y_i, -y_i, 0, self.border_y[0]]))

        self._edge_styles = [_DRAIN_EDGE] * 24 + [_MIRROR_EDGE] * 2 + [_SOURCE_EDGE] + [_MIRROR_EDGE] * 2
        self.device_edges_count = len(self._edge_styles)
        self._box_range = [[-radius, radius], [-radius, radius]]

        self._setup_edges()

        if self._tip:
            print("Adding tips")
            self._add_tip()
