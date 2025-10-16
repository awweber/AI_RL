from typing import Any # For type hinting (MyPy)
import gymnasium as gym

class Agent:
    # Initialize the agent with the environment
    def __init__(self, env: gym.Env) -> None:
        self.env = env
    # Select an action randomly from the action space
    def get_action(self) -> Any:
        return self.env.action_space.sample() # Random action
    # Play a specified number of episodes, optionally rendering the environment
    def play(self, episodes: int, render: bool = True) -> list:
        rewards = [0.0 for _ in range(episodes)] # List to store total rewards per episode
        # Loop through the specified number of episodes
        for episode in range(episodes):
            self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action()
                _, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break

            print(f"Episode: {episode} Total Reward: {total_reward}")
            rewards[episode] = total_reward
        self.env.close()
        return rewards


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.play(episodes=100, render=True)


if __name__ == "__main__":
    main()
