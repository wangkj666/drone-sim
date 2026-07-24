"""物理闭环悬停测试：状态反馈 -> 级联 PID -> 混控 -> 四电机推力。"""
import os
import sys

import torch
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from omni.isaac.core import World

from drone_model import DroneModel
from flight_controller import FlightController
from mixer import QuadMixer


cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()
drone = DroneModel(config).build(world)
controller = FlightController(config, QuadMixer(config))
controller.set_target([0.0, 0.0, 2.0])
world.reset()

frame = 0
while simulation_app.is_running():
    state = drone.get_state()
    controller.update_state(
        state["position"], state["velocity"], state["attitude"], state["angular_velocity"]
    )
    drone.apply_rotor_forces(controller.compute_control())
    world.step(render=True)

    frame += 1
    if frame % 120 == 0:
        print(f"z={state['position'][2]:.2f}, velocity_z={state['velocity'][2]:.2f}")

simulation_app.close()
