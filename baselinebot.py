import gym

num_episodes = 10
rewards = []

def main():
    env = gym.make('CartPole-v0')
    env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done:
            rewards.append(episode_reward)

    print(f"Avg reward: {sum(rewards)/len(rewards)}")


if __name__ == "__main__":
    main()