from cartpoleA2CNN import Actor
from cartpoleA2CNN import Critic
from cartpoleA2CAgent import Agent

import os
import numpy as np

PROJECT_PATH = os.path.abspath("/Users/alex/Code/udemy/AI_RL")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
ACTOR_PATH = os.path.join(MODELS_PATH, "actor_cartpole.weights.h5")
CRITIC_PATH = os.path.join(MODELS_PATH, "critic_cartpole.weights.h5")

import gymnasium as gym


def _ensure_weights_exist(actor_path: str, critic_path: str) -> bool:
    """Return True if both weight files exist, otherwise False."""
    actor_exists = os.path.exists(actor_path)
    critic_exists = os.path.exists(critic_path)
    return actor_exists and critic_exists


if __name__ == "__main__":
    # Create environment with GUI render
    env = gym.make("CartPole-v1", render_mode="human")

    agent = Agent(env)

    # Try to load saved weights for actor and critic. Agent.play() also
    # attempts to load the weights, but we load them here explicitly so
    # we can provide a clear error message if they're missing.
    if _ensure_weights_exist(ACTOR_PATH, CRITIC_PATH):
        print(f"Loading actor weights from: {ACTOR_PATH}")
        agent.actor.load_model(ACTOR_PATH)
        print(f"Loading critic weights from: {CRITIC_PATH}")
        agent.critic.load_model(CRITIC_PATH)
    else:
        print("Saved model weights not found.")
        print(f"Expected actor weights at: {ACTOR_PATH}")
        print(f"Expected critic weights at: {CRITIC_PATH}")
        raise SystemExit(
            "No saved weights found. Run training first or update the paths."
        )

    # Play using the loaded weights (render=True keeps Agent.play behavior)
    agent.play(num_episodes=1, render=True)