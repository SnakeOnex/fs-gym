import numpy as np
import argparse
from pathlib import Path
import time
import gymnasium as gym
import pickle
from gymnasium.spaces import Box, Discrete, Tuple, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from renderer import StateRenderer

cones_world = np.array([[-1., 1., 1.],
                        [ 1., 1., 0.],
                        [-1., 2., 1.],
                        [ 1., 2., 0.],])

path = np.array([[0., 0.],
                 [0., 1.],
                 [0., 2.],
                 [0., 3.],
                 [0., 4.],])

basic_map = {"cones_world": cones_world, "path": path, "start_pose": np.array([0., 0., np.deg2rad(90)])}

data = pickle.load(open("skidpad_single_pass.pkl", "rb")) 
cones_world = np.zeros((0,3)) 

def colored_to_world_cones(cones, color):
    cones = np.hstack((cones, np.ones((cones.shape[0], 1)) * color))
    return cones

yellow_cones = colored_to_world_cones(data["yellow_cones"], 0)
blue_cones = colored_to_world_cones(data["blue_cones"], 1)
orange_cones = colored_to_world_cones(data["orange_cones"], 2)
cones_world = np.vstack((yellow_cones, blue_cones, orange_cones))
path = data["center_line"]
path = path[40:, :]

path = path[::1, :]

skidpad_map = {"cones_world": cones_world, "path": path, "start_pose": np.array([*path[0], np.deg2rad(90)])}

# autox map
tracks_path = Path("/home/snake/fun/bros/data/e2e/tracks_dataset.pkl")
tracks = pickle.load(open(tracks_path, "rb"))
track = tracks[0]

cones_world = np.zeros((0,3))
yellow_cones = colored_to_world_cones(track["yellow_cones"], 0)
blue_cones = colored_to_world_cones(track["blue_cones"], 1)
cones_world = np.vstack((yellow_cones, blue_cones))
path = track["center_line"]

autox_map = {"cones_world": cones_world, "path": path, "start_pose": np.array([*path[0], np.deg2rad(90)])}


def kinematic_model(yaw, speed, delta_f):
    lr = 0.75
    wheel_base = 1.5
    beta = np.arctan(lr * np.tan(delta_f) / wheel_base)
    d_yaw = speed * (np.tan(delta_f) * np.cos(beta) / wheel_base)
    dx = speed * np.cos(beta + yaw)
    dy = speed * np.sin(beta + yaw)
    der = [dx, dy, d_yaw]
    return np.array(der)

class FSEnv(gym.Env):
    def __init__(self, map):
        self.map = map
        self.renderer = None

        self.car_pose = map["start_pose"]
        self.cones_world = map["cones_world"]
        self.path = map["path"]
        self.path_idx = 0
        self.path_sum = 0.

        self.action_space = MultiDiscrete([2, 2, 2])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3+self.cones_world.shape[0]*3,), dtype=np.float32)

        self.reset()

        self.goal_pose = self.path[-1]

        self.text = ""
        self.t = 0

    def step(self, action):
        
        # 1. perform action
        forw, left, right = action
        if forw:
            speed_action = 1.
        else:
            speed_action = 0.

        if left and not right:
            steer_action = np.deg2rad(60)
        elif right and not left:
            steer_action = -np.deg2rad(60)
        else:
            steer_action = 0.
        self.car_pose += kinematic_model(self.car_pose[2], speed_action, steer_action) * 0.2

        # update path
        if np.linalg.norm(self.car_pose[:2] - self.path[self.path_idx]) < 1.0:
            self.path_idx += 1
            self.path_sum += np.linalg.norm(self.path[self.path_idx] - self.path[self.path_idx-1])

        # 2. compute reward
        dist_to_road = np.min(np.linalg.norm(self.car_pose[:2] - self.path, axis=1))
        obs = np.concatenate([self.car_pose, self.cones_world.flatten()])
        start_dist = np.linalg.norm(self.goal_pose[:2] - self.path[0])
        curr_dist = np.linalg.norm(self.car_pose[:2] - self.goal_pose[:2])
        # reward = start_dist - curr_dist
        reward = self.path_sum - dist_to_road
        finished = curr_dist < 0.2
        if finished:
            reward = 100.

        truncated = dist_to_road > 2.0

        # Q: what are the return args? (observation, reward, done, info)
        self.text = f"t={self.t}\nreward: {reward:.2f}\n{self.path_idx=}, {self.path_sum=:.2f}\n"
        self.t += 1

        return obs, reward, finished, truncated, {}

    def reset(self, **kwargs):
        self.car_pose = np.array([*self.path[0], np.deg2rad(90)])
        self.t = 0
        self.path_idx = 0
        self.path_sum = 0.
        return np.concatenate([self.car_pose, self.cones_world.flatten()]), {}

    def render(self, mode="human"):
        if self.renderer is None:
            self.renderer = StateRenderer()
            self.renderer.set_state(self.map)
        self.renderer.render_state(self.car_pose, self.text)

gym.envs.register(id="FSBasic-v0",
                  entry_point=FSEnv,
                  kwargs={"map": basic_map})

gym.envs.register(id="FSSkidpad-v0",
                  entry_point=FSEnv,
                  kwargs={"map": skidpad_map})

gym.envs.register(id="FSAutox-v0",
                  entry_point=FSEnv,
                  kwargs={"map": autox_map})

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train", action="store_true")
    args = arg_parser.parse_args()

    # env_name = "CartPole-v1"
    # env_name = "FSBasic-v0"
    env_name = "FSSkidpad-v0"
    # env_name = "FSAutox-v0"

    if args.train:
        vec_env = make_vec_env(env_name, n_envs=8)
        model = PPO("MlpPolicy", vec_env, verbose=1)
        model.learn(total_timesteps=60000)
        model.save("ppo_cartpole")

    model = PPO.load("ppo_cartpole")

    env = gym.make(env_name)
    while True:
        obs, info = env.reset()
        for i in range(200):
            # action, _states = model.predict(obs)
            action = env.action_space.sample()
            obs, rewards, done, truncated, info = env.step(action)
            if done:
                print("done")
                exit(0)

            if truncated:
                print("truncated")
                # exit(0)

            env.render()
            # time.sleep(0.1)
