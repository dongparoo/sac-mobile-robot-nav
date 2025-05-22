# end_to_end_nav.py

import gym
from gym import spaces
import numpy as np
import math
import rclpy
from ros_interface_cnn import ROSInterface
from geometry_msgs.msg import Pose2D
import time
import cv2


class EndToEndNavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        rclpy.init()
        self.ros_interface = ROSInterface()

        self.robot_reset_pub = self.ros_interface.create_publisher(Pose2D, '/reset_robot_pose', 10)
        self.goal_reset_pub = self.ros_interface.create_publisher(Pose2D, '/reset_goal_pose', 10)

        self.num_lidar_points = 12
        self.image_size = (3, 128, 128)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=self.image_size, dtype=np.uint8),
            "vector": spaces.Box(low=0.0, high=30.0, shape=(self.num_lidar_points + 3,), dtype=np.float32)
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, -5.0], dtype=np.float32),
            high=np.array([2.0, 5.0], dtype=np.float32),
            dtype=np.float32
            )
        
        self.goal_pos = (0.0, 0.0)
        self.robot_start_world = (-9.0, -9.0)
        self.prev_distance = None
        self.reached_near_goal = False
        self.step_count = 0
        self.episode_count = 1220
    
        self.ignore_collision_steps = 0

    def reset(self):
        self.step_count = 0
        self.prev_distance = None
        self.ignore_collision_steps = 10

        self.episode_count += 1
        level = self.episode_count // 300
        x_max = min(-3.0 + level * 1.0, 9.0)
        y_max = min(-3.0 + level * 1.0, 9.0)

        goal_x = np.random.uniform(-9.0, x_max)
        goal_y = np.random.uniform(-9.0, y_max)
        self.goal_pos = (goal_x, goal_y)

        for _ in range(5):
            self.ros_interface.send_velocity(0.0, 0.0)
            rclpy.spin_once(self.ros_interface, timeout_sec=0.1)

        time.sleep(0.5)

        self._reset_robot_pose(0.0, 0.0)
        self._move_goal_cube(goal_x, goal_y)

        time.sleep(2.0)

        for _ in range(5):
            rclpy.spin_once(self.ros_interface, timeout_sec=0.1)
        return self._get_observation()
    
    def step(self, action):
        # linear_x = 0.8
        linear_x = float(action[0])
        angular_z = float(action[1])
        self.ros_interface.send_velocity(linear_x, angular_z)

        rclpy.spin_once(self.ros_interface, timeout_sec=0.1)
        # time.sleep(0.1)

        obs = self._get_observation()
        reward, done, distance = self._compute_reward_done(action)
        self.step_count += 1

        if self.step_count > 500:
            done = True
            reward -= 100.0

        if done:
            self.ros_interface.send_velocity(0.0, 0.0)

        print(f"🔍 Min laser: {np.min(self.ros_interface.laser_ranges):.2f}")
        print(f"📌 Reward: {reward:.2f}, Done: {done}")
        print("📏  distance: %f" % distance)
        print(f"🧾 Episode: {self.episode_count}, Step: {self.step_count}")

        return obs, reward, done, {}
    
    def _get_observation(self):
        lidar = self.ros_interface.get_laser_scan(self.num_lidar_points)
        (x_odom, y_odom), _ = self.ros_interface.get_position_and_yaw()
        x_world = x_odom + self.robot_start_world[0]
        y_world = y_odom + self.robot_start_world[1]
        dx = self.goal_pos[0] - x_world
        dy = self.goal_pos[1] - y_world
        dist = np.sqrt(dx**2 + dy**2)

        vector = np.concatenate([lidar, np.array([dx, dy, dist], dtype=np.float32)])
        image = self.ros_interface.get_rgb_image()
        image = np.transpose(image, (2, 0, 1))  # (HWC) → (CHW)

        #return {
        #    "image": image,
        #    "vector": vector
        #}

        obs = {
            "image": image,
            "vector": vector
        }

        # NaN 체크 디버깅
        if np.isnan(image).any():
            print("⚠️ image contains NaN")
        if np.isnan(vector).any():
            print("⚠️ vector contains NaN")

        return obs

    
    def _compute_reward_done(self, action):
        (x_odom, y_odom), yaw = self.ros_interface.get_position_and_yaw()
        x_world = x_odom + self.robot_start_world[0]
        y_world = y_odom + self.robot_start_world[1]
        dx = self.goal_pos[0] - x_world
        dy = self.goal_pos[1] - y_world
        distance = math.sqrt(dx ** 2 + dy ** 2)

        step_max = 500
        time_scale = 1.0 - (self.step_count / step_max)

        p = 300.0       # 충돌 패널티
        r_l = 200.0    # 목표 도달 보상

        # laser
        min_lidar = np.min(self.ros_interface.laser_ranges)

        # 충돌 여부
        if min_lidar < 0.45:
            reward = -p
            print(f"📊 Reward: {reward:.2f} (충돌)")
            return reward, True, distance
        
        # 목표 도달 여부
        if distance < 1.0:
            reward = r_l * time_scale
            print(f"📊 Reward: {reward:.2f} (도달)")
            return reward, True, distance
        
        # 일반 주행
        reward = -distance * 0.6

        if min_lidar < 0.6:
            reward -= 3.0
            print(f"📉 Proximity penalty: -3.0")

        print(f"📊 Reward: {reward:.2f}, distance: {distance:.2f}, time scale: {time_scale:.2f}")
        return reward, False, distance
    
    
    def _reset_robot_pose(self, x, y):
        msg = Pose2D()
        msg.x = x
        msg.y = y
        msg.theta = 0.0
        self.robot_reset_pub.publish(msg)
        print(f"[ROS] 로벌 위치 초기화: ({x:.2f}, {y:.2f})")

    def _move_goal_cube(self, x, y):
        msg = Pose2D()
        msg.x = x
        msg.y = y
        msg.theta = 0.0
        self.goal_reset_pub.publish(msg)
        print(f"[ROS] 목표 위치 설정: ({x:.2f}, {y:.2f})")


    def close(self):
        self.ros_interface.destroy_node()
        rclpy.shutdown()

