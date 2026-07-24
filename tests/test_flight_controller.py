import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drone_project"))

from flight_controller import PID, PositionMPC
from mixer import QuadMixer


class PIDTest(unittest.TestCase):
    def test_measured_rate_avoids_setpoint_derivative_kick(self):
        pid = PID(kp=0.0, ki=0.0, kd=2.0, integral_max=1.0)

        pid.step(0.0, dt=0.01, measurement_rate=0.0)
        output = pid.step(10.0, dt=0.01, measurement_rate=0.0)

        self.assertEqual(output, 0.0)

    def test_integral_does_not_grow_when_output_is_saturated(self):
        pid = PID(kp=2.0, ki=1.0, kd=0.0, integral_max=10.0, output_limit=1.0)

        for _ in range(10):
            pid.step(2.0, dt=0.1)

        self.assertEqual(pid._integral, 0.0)


class MPCTest(unittest.TestCase):
    def test_mpc_accelerates_toward_a_higher_target(self):
        controller = PositionMPC(
            dt=0.04,
            horizon=12,
            position_weight=12.0,
            velocity_weight=3.0,
            control_weight=0.3,
            acceleration_limit=4.0,
        )

        acceleration = controller.step(position=0.0, velocity=0.0, target=2.0)

        self.assertGreater(acceleration, 0.0)
        self.assertLessEqual(acceleration, 4.0)


class MixerTest(unittest.TestCase):
    def test_positive_pitch_command_increases_left_rotors(self):
        config = {
            "drone": {"arm_length": 1.0},
            "control": {
                "thrust_to_torque": 1.0,
                "mixer_type": "x",
                "min_thrust": -100.0,
                "max_thrust": 100.0,
            },
        }
        thrusts = QuadMixer(config).mix(thrust=0.0, roll=0.0, pitch=4.0, yaw=0.0)

        self.assertGreater(thrusts[1], 0.0)  # M2: left-front
        self.assertGreater(thrusts[2], 0.0)  # M3: left-rear
        self.assertLess(thrusts[0], 0.0)     # M1: right-front
        self.assertLess(thrusts[3], 0.0)     # M4: right-rear


if __name__ == "__main__":
    unittest.main()
