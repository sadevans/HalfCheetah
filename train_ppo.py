from IPython.display import clear_output
from tqdm import trange
import matplotlib.pyplot as plt
import gym
import torch

from ppo.Runner import *
from ppo.Policy import *
from ppo.PPO import *
from ppo.utils import *
from ppo.Sampler import *
from ppo.Network import *



def make_ppo_runner(env, policy, num_runner_steps=2048,
                    gamma=0.99, lambda_=0.95, 
                    num_epochs=16, num_minibatches=64):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                         GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps, 
                       transforms=runner_transforms)
    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs, 
                                num_minibatches=num_minibatches,
                                transforms=sampler_transforms)
    return sampler



def plot_tools(legend, position, data_y):
    plt.subplot(2,4,position)
    plt.plot(data_y, label=legend)
    plt.title(legend); plt.grid(); plt.legend() 
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


def evaluate(env, agent, n_games=1, render=False):
    """Plays an a game from start till done, returns per-game rewards """
    agent.train(False)
    game_rewards = []
    done_counter = 0
    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        while True:
            if render:
                env.render()
            state = torch.tensor(state, dtype=torch.float32)
            (mus, sigmas), _ = agent(state)
            dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
            action = dist.sample().cpu().detach().numpy()
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        game_rewards.append(total_reward)
    agent.train(True)
    return game_rewards



if __name__ == '__main__':
    env = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True)
    model = Network(shape_in=17, action_shape=6)
    policy = Policy(model)
    runner = make_ppo_runner(env, policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-05)

    ppo = PPO(policy, optimizer) 
    num_steps = []
    rewards = []
    value_losses = []
    policy_losses = []
    values = []
    grad_norms = []
    advantages = []
    ppo_losses = []
    for i in trange(100_000):
        trajectory = runner.get_next()
        ppo.step(trajectory)
        value_losses.append(ppo.values_np)
        policy_losses.append(ppo.policy_loss_np)
        values.append(ppo.values_np)
        grad_norms.append(ppo.total_norm)
        advantages.append(ppo.advantages_np)
        ppo_losses.append(ppo.ppo_loss_np)
        if i%100==0:
            clear_output(True)
            num_steps.append(runner.runner.step_var)
            
            
            rewards.append(np.mean(evaluate(env, model, n_games=1)))
            
            plt.figure(figsize=[20,10])
            
            plt.subplot(2,4,1)
            plt.plot(num_steps, rewards, label='Reward')
            plt.title("Rewards"); plt.grid(); plt.legend()
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

            plot_tools('Values', 2, values)
            plot_tools('Value loss', 3, value_losses)
            plot_tools('Policy loss', 4, policy_losses)
            plot_tools('PPO loss', 5, ppo_losses)
            plot_tools('Grad_norm_L2', 6, grad_norms) 
            plot_tools('Advantages', 7, advantages)

            plt.show()

    env.close()
