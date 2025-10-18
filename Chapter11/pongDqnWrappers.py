# pongDqnWrappers.py
# Wrappers for the Pong DQN environment
# Description: This module contains custom wrappers for the Pong environment 
# to preprocess observations and stack frames for DQN training.
# - stacks the last 4 frames and starts the game automatically
# - uses AtariPreprocessing for standard preprocessing steps


import collections
from typing import Any, Tuple, Deque

import ale_py
import gymnasium as gym
import numpy as np


class StartGameWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env.reset()

    # Automatically start the game by taking the FIRE action on reset
    def reset(self, **kwargs: Any) -> Any:
        self.env.reset()
        observation, _, _, _, info = self.env.step(1)  # FIRE
        return observation


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_buffer_frames: int) -> None:
        super().__init__(env)
        # Number of frames to stack
        self.num_buffer_frames = num_buffer_frames
        # Deque to hold the last `num_buffer_frames` frames
        self.frames: collections.deque = collections.deque(
            maxlen=self.num_buffer_frames,
        )
        # Modify the observation space to reflect the frame stacking
        low = np.repeat(
            self.observation_space.low[np.newaxis, ...],
            repeats=self.num_buffer_frames,
            axis=0,
        )
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...],
            repeats=self.num_buffer_frames,
            axis=0,
        )
        # Update the observation space to have shape (84, 84, num_buffer_frames)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype,
        )

    # Stack the last `num_buffer_frames` frames along the last dimension
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, done, truncated, info = self.env.step(action)
        self.frames.append(observation)
        frame_stack = np.asarray(self.frames, dtype=np.float32)  # (4, 84, 84)
        frame_stack = np.moveaxis(
            frame_stack,
            source=0,
            destination=-1,
        )  # (84, 84, 4)
        frame_stack = np.expand_dims(frame_stack, axis=0)  # (1, 84, 84, 4)
        return frame_stack, reward, done, truncated, info

    # Update the frame stack on reset
    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        self.env.reset(**kwargs)
        self.frames = collections.deque(maxlen=self.num_buffer_frames)
        for _ in range(self.num_buffer_frames):
            self.frames.append(np.zeros(shape=(84, 84), dtype=np.float32))
        return np.zeros(shape=(1, 84, 84, 4), dtype=np.float32), {}


def make_env(env_name: str, num_buffer_frames: int) -> gym.Env:
    env = gym.make(env_name)
    env = gym.wrappers.AtariPreprocessing(
        env=env,
        noop_max=20,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=True,
    )
    env = FrameStackWrapper(env, num_buffer_frames)
    return StartGameWrapper(env)


if __name__ == "__main__":
    env_name = "PongNoFrameskip-v4"
    num_buffer_frames = 4
    env = make_env(env_name, num_buffer_frames)
