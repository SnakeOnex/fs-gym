import numpy as np
import time
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple, MultiDiscrete

from renderer import StateRenderer

cones_world = np.array([[-1., 1., 0.],
                        [ 1., 1., 0.],
                        [-1., 2., 0.],
                        [ 1., 2., 0.],])

path = np.array([[0., 0.],
                 [0., 1.],
                 [-2., 2.],])

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
    def __init__(self, render_mode="none"):
        # self.action_space = Discrete(3) # forward, left, right
        # self.action_space = Tuple((Discrete(2), Discrete(2), Discrete(2)))
        self.action_space = MultiDiscrete([2, 2, 2])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3+4*3,), dtype=np.float32)

        self.reset()

        if render_mode == "human":
            self.renderer = StateRenderer()
            self.renderer.set_state({"cones_world": cones_world, "path": path, "start_pose": self.car_pose.copy()})
        self.goal_pose = path[-1]

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
        self.car_pose += kinematic_model(self.car_pose[2], speed_action, steer_action) * 0.1

        # 2. compute reward
        obs = np.concatenate([self.car_pose, cones_world.flatten()])
        start_dist = np.linalg.norm(self.goal_pose[:2] - path[0])
        curr_dist = np.linalg.norm(self.car_pose[:2] - self.goal_pose[:2])
        reward = start_dist - curr_dist
        # reward = -np.linalg.norm(self.car_pose[:2] - self.goal_pose[:2])
        finished = curr_dist < 0.1
        if finished:
            reward = 10.

        truncated = np.linalg.norm(self.car_pose[:2] - self.goal_pose[:2]) > 5.

        # Q: what are the return args? (observation, reward, done, info)
        self.text = f"t={self.t}\nreward: {reward:.2f}"
        self.t += 1

        return obs, reward, finished, truncated, {}

    def reset(self, **kwargs):
        self.car_pose = np.array([0., 0., np.deg2rad(90)])
        self.t = 0
        return np.concatenate([self.car_pose, cones_world.flatten()]), {}

    def render(self, mode="human"):
        self.renderer.render_state(self.car_pose, self.text)

gym.envs.register(id="FSEnv-v0", entry_point=FSEnv)

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # env_name = "CartPole-v1"
    env_name = "FSEnv-v0"

    vec_env = make_vec_env(env_name, n_envs=16)
    model = PPO("MlpPolicy", vec_env, verbose=1)

    model.learn(total_timesteps=40000)
    model.save("ppo_cartpole")
    # del model # remove to demonstrate saving and loading

    # model = PPO.load("ppo_cartpole")
    # make cartpole render
    # vec_env = model.get_env()
    # vec_env = make_vec_env(env_name, n_envs=1)
    # # env = gym.make(env_name, render_mode="human")
    # obs = vec_env.reset()

    # for i in range(1000):
        # action, _states = model.predict(obs)
        # obs, rewards, dones, info = vec_env.step(action)
        # vec_env.render("human")

    env = gym.make(env_name, render_mode="human")
    while True:
        obs, info = env.reset()
        for i in range(50):
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, info = env.step(action)
            if done:
                print("done")
                break
            env.render()
            time.sleep(0.05)


    # env = gym.make("FSEnv-v0", render=False)
    # i = 0
    # while True:
        # # env.render()
        # env.step(env.action_space.sample())
        # # time.sleep(0.1)
        # if i % 10 == 0:
            # env.reset()

        # i += 1

