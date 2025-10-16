# Kapitel 3: Grundlagen zu Reinforcement Learning (RL)
# Zustände, Aktionen und Belohnungen
# Markov-Entscheidungsprozess (MDP)
STATES = ["a", "b"]
ACTIONS = ["a", "b"]
REWARDS = {"a": {"a": 0, "b": 7}, "b": {"a": -5, "b": 0}}

# Markov-Entscheidungsprozess (MDP)
def main() -> None:
    # Initialisierung
    state = "a"
    reward = 0
    total_reward = 0

    # Ausgabe des Startzustands und der Startbelohnung
    print(f"Start-State: {state} Start-Reward: {reward}\n\n")

    # Simuliere 10 Iterationen
    for i in range(1, 11):
        print(f"State: {state} - Iteration: {i}")
        action = input("Action: ")
        if action in ACTIONS:
            reward = REWARDS[state][action]
            total_reward += reward
            state = action
            print(
                f"New State: {state} "
                f"Reward: {reward} "
                f"Total-Reward: {total_reward}",
            )

# Starte das Programm
if __name__ == "__main__":
    main()
