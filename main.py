import numpy as np
import time
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple

from renderer import StateRenderer

cones_world = np.array([[-1., 1., 0.],
                        [ 1., 1., 0.],
                        [-1., 2., 0.],
                        [ 1., 2., 0.],])

path = np.array([[0., 0.],
                 [0., 1.],
                 [0., 2.],])

def kinematic_model(yaw, speed, delta_f):
    lr = 0.75
    wheel_base = 1.5
    beta = np.arctan(lr * np.tan(delta_f) / wheel_base)
    d_yaw = speed * (np.tan(delta_f) * np.cos(beta) / wheel_base)
    dx = speed * np.cos(beta + yaw)
    dy = speed * np.sin(beta + yaw)
    der = [dx, dy, d_yaw]
    return der

class FSEnv(gym.Env):
    def __init__(self, render=False):
        # self.action_space = Discrete(3) # forward, left, right
        self.action_space = Tuple((Discrete(2), Discrete(2), Discrete(2)))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3+2*3,), dtype=np.float32)

        self.reset()

        if render:
            self.renderer = StateRenderer()
            self.renderer.set_state({"cones_world": cones_world, "path": path, "start_pose": self.car_pose.copy()})

    def step(self, action):
        forw, left, right = action
        if forw:
            speed_action = 1.
        else:
            speed_action = 0.

        if left and not right:
            steer_action = np.deg2rad(10)
        elif right and not left:
            steer_action = -np.deg2rad(10)
        else:
            steer_action = 0.

        self.car_pose += kinematic_model(self.car_pose[2], speed_action, steer_action)

        return 0, 0, False, {}

    def reset(self):
        self.car_pose = np.array([0., 0., np.deg2rad(90)])
        self.goal_pose = np.array([0., 0., 0.])

    def render(self, mode="human"):
        self.renderer.render_state(self.car_pose)
        

if __name__ == "__main__":
    env = FSEnv(render=True)
    env.reset()

    i = 0
    while True:
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.1)
        if i % 10 == 0:
            env.reset()

        i += 1

