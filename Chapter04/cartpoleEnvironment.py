import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    episodes = 100

    for episode in range(episodes):
        env.reset() # reset the environment at the start of each episode
        total_reward = 0.0

        while True:
            # env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Episode: {episode} Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
