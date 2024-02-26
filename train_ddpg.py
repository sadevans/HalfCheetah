import gym
from ddpg.DDPG import *

N_EPOCHS = 200
TRAJECTORY_LEN = 200

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True)
    episode_n = N_EPOCHS
    trajectory_len = TRAJECTORY_LEN 
    # state_dim, action_dim, action_scale = 17, 6, 1


    agent = DDPG(state_dim=17, action_dim=6, action_scale=1, noise_decrease=1/(episode_n*trajectory_len))
    obs, _= env.reset()
    # total_reward = 0

    for episode in range(episode_n):
        total_reward = 0
        state,_ = env.reset()
        for _ in range(trajectory_len):
            action = agent.get_action(state)

            next_action, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.fit(state, action, reward, done, next_action)

            if done:
                break

            state = next_action

        print(f'episode={episode}, total_reward={total_reward}')