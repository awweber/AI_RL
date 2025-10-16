import ale_py
import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium import envs


env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env.reset()

play(env, zoom=4, fps=15)
env.close()

