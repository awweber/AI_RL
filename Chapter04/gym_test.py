import ale_py
import gymnasium as gym

# Prüfen, ob die Pong-Umgebung registriert ist

print("ALE/Pong-v5" in gym.registry.keys())  # sollte jetzt True sein


env = gym.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()