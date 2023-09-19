#!/usr/bin/env python3

from __future__ import annotations

import os
import copy
import datetime
import gymnasium as gym
import pygame
import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, DIR_TO_VEC
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class PartiallyObsWrapper(ObservationWrapper):
    """
    Partially masking FullyObsWrapper.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = env = PartiallyObsWrapper(FullyObsWrapper(env))
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space.spaces["grid_obs"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.sight_range = self.env.unwrapped.agent_view_size
        self.observation_mask = np.zeros((self.env.width, self.env.height), dtype=bool)

    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.observation_mask.fill(False)
        return self.observation(obs), info

    def observation(self, obs):
        env = self.env.unwrapped
        grid = env.grid.encode()
        grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )
        agent_positon = np.array(np.where(grid[:, :, 0] == OBJECT_TO_IDX["agent"])).reshape(2)
        agent_direction = grid[agent_positon[0], agent_positon[1], 2]

        # Compute the sight range in front of the agent
        direction_vec = DIR_TO_VEC[agent_direction]
        for i in range(self.sight_range):
            for j in range((self.sight_range + 1) // 2):
                pos1 = agent_positon + direction_vec * i + np.array([direction_vec[1], direction_vec[0]]) * j
                pos2 = agent_positon + direction_vec * i - np.array([direction_vec[1], direction_vec[0]]) * j
                if pos1[0] in range(env.width) and pos1[1] in range(env.height):
                    self.observation_mask[pos1[0], pos1[1]] = True
                if pos2[0] in range(env.width) and pos2[1] in range(env.height):
                    self.observation_mask[pos2[0], pos2[1]] = True
        
        grid[~self.observation_mask] = OBJECT_TO_IDX["unseen"]
        return {**obs, "grid_obs": grid}


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of the agent view.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = FullyObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "state": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "state": full_grid}
    

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
        save_path="data",
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.save_path = os.path.join(save_path, f'human-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.npy')
        self.data = []
        self.episode = {}

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def push_data(self, **kwargs):
        for key, value in kwargs.items():
            data_list = self.episode.get(key, list())
            data_list.append(value)
            self.episode[key] = data_list
    
    def save_episode_data(self):
        self.data.append(copy.deepcopy(self.episode))
        self.episode.clear()
        print(f"Save data to {self.save_path}")
        np.save(self.save_path, self.data)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.push_data(obs=obs, action=int(action), reward=reward, terminated=terminated, truncated=truncated)

        if terminated or truncated:
            self.save_episode_data()
            print(f"Reward: {reward}; Terminated: {terminated}; Truncated {truncated}")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.push_data(obs=obs)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    env = FullyObsWrapper(env)
    env = PartiallyObsWrapper(env)

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()
