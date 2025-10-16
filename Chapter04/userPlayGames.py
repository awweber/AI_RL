import gymnasium as gym
from gymnasium.utils.play import play
import ale_py

games = [
    "CartPole-v1",
    "MountainCar-v0",
    "ALE/Pong-v5",
    "ALE/Breakout-v5",
]

game = games[0]  # Wähle das gewünschte Spiel aus der Liste
env = gym.make(game, render_mode="rgb_array")
obs, info = env.reset()

# Key mapping: 
# (97,): 0,     # 'a' (Mac) = Left Arrow, 
# (100,): 1,    # 'd' (Mac) = Right Arrow
keys_to_action = {"a ": 0, "w ": 1}

play(env, fps=15, zoom=2)
env.close()


