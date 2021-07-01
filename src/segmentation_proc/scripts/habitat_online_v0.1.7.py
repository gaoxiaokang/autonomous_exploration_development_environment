#!/usr/bin/python3

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import argparse

import math
import multiprocessing
import os
import random
import time
from enum import Enum

import numpy as np
from PIL import Image

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    quat_from_coeffs,
)

from scipy.spatial.transform import Rotation as R

default_sim_settings = {
    "frame_rate": 30, # image frame rate
    "width": 640, # horizontal resolution
    "height": 360, # vertical resolution
    "hfov": "114.591560981", # horizontal FOV
    "camera_offset_z": 0, # camera z-offset
    "color_sensor": True,  # RGB sensor
    "depth_sensor": True,  # depth sensor
    "semantic_sensor": True,  # semantic sensor
    "scene": "../../vehicle_simulator/mesh/matterport/segmentations/matterport.glb",
}

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default=default_sim_settings["scene"])
args = parser.parse_args()

def make_settings():
    settings = default_sim_settings.copy()
    settings["scene"] = args.scene

    return settings

settings = make_settings()

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.frustum_culling = False
    sim_cfg.gpu_device_id = 0
    if not hasattr(sim_cfg, "scene_id"):
        raise RuntimeError(
            "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
        )
    sim_cfg.scene_id = settings["scene"]

    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["camera_offset_z"], 0.0],
            "sensor_subtype": habitat_sim.SensorSubType.PINHOLE,
            "hfov": settings["hfov"],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["camera_offset_z"], 0.0],
            "sensor_subtype": habitat_sim.SensorSubType.PINHOLE,
            "hfov": settings["hfov"],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["camera_offset_z"], 0.0],
            "sensor_subtype": habitat_sim.SensorSubType.PINHOLE,
            "hfov": settings["hfov"],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.sensor_subtype = sensor_params["sensor_subtype"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.gpu2gpu_transfer = False
            sensor_spec.parameters["hfov"] = sensor_params["hfov"]

            sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2
    AB_TEST = 3

class ABTestGroup(Enum):
    CONTROL = 1
    TEST = 2

class DemoRunner:
    camera_roll = 0
    camera_pitch = 0
    camera_yaw = 0
    camera_x = 0
    camera_y = 0
    camera_z = 0

    def __init__(self, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._demo_type = simulator_demo_type

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def publish_color_observation(self, obs):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        # publish image on ROS topic '/habitat_camera/color/image' in 'habitat_camera' frame

    def publish_semantic_observation(self, obs):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        # publish image on ROS topic '/habitat_camera/semantic/image' in 'habitat_camera' frame

    def publish_depth_observation(self, obs):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        # publish image on ROS topic '/habitat_camera/depth/image' in 'habitat_camera' frame

    def init_common(self):
        self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        self._sim = habitat_sim.Simulator(self._cfg)

        if not self._sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)

    def state_estimation_callback(self, msg):
        orientation = msg.pose.pose.orientation
        (self.camera_roll, self.camera_pitch, self.camera_yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.camera_x = msg.pose.pose.position.x
        self.camera_y = msg.pose.pose.position.y
        self.camera_z = msg.pose.pose.position.z

    def listener(self):
        rospy.init_node('habitatOnline')

        rospy.Subscriber("/state_estimation", Odometry, self.state_estimation_callback)

        start_state = self.init_common()

        r = rospy.Rate(default_sim_settings["frame_rate"])
        while not rospy.is_shutdown():
            roll = -self.camera_roll
            pitch = self.camera_pitch
            yaw = 1.5708 - self.camera_yaw

            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

            position = np.array([self.camera_x, self.camera_y, self.camera_z])
            position[1], position[2] = position[2], -position[1]
            
            agent_state = self._sim.get_agent(0).get_state()
            for sensor in agent_state.sensor_states:
                agent_state.sensor_states[sensor].position = position + np.array([0, default_sim_settings["camera_offset_z"], 0])
                agent_state.sensor_states[sensor].rotation = quat_from_coeffs(np.array([-qy, -qz, qx, qw]))

            self._sim.get_agent(0).set_state(agent_state, infer_sensor_states = False)                
            observations = self._sim.step("move_forward")

            if self._sim_settings["color_sensor"]:
                self.publish_color_observation(observations)
            if self._sim_settings["depth_sensor"]:
                self.publish_depth_observation(observations)
            if self._sim_settings["semantic_sensor"]:
                self.publish_semantic_observation(observations)

            state = self._sim.last_state()
            print(rospy.get_time(), position) # remove in the end
            r.sleep()

        self._sim.close()
        del self._sim

demo_runner = DemoRunner(settings, DemoRunnerType.EXAMPLE)
demo_runner.listener()
