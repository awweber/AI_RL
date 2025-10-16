import gymnasium as gym

def main() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    for _ in range(50):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
