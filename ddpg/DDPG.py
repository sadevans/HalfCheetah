import random
from collections import deque
from copy import deepcopy

from Buffer import *
from Noise import *
from NeuralNetwork import *


class DDPG():
  def __init__(self, state_dim, action_dim, action_scale, noise_decrease, gamma=0.99, batch_size=64, actor_lr=1e-4, critic_lr=1e-3, tau = 1e-2, memory_size=1000000):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.action_scale = action_scale
    self.actor = ThreeLayersNeuralNetwork(self.state_dim, 400, 300, self.action_dim, output_tanh = True, _lr=actor_lr, name='actor')

    self.critic = ThreeLayersNeuralNetwork(self.state_dim + self.action_dim, 400, 300, 1, output_tanh = False, _lr=critic_lr, name='critic') # конкатенируем вектор состояний и вектор действий, на выходе всего 1 значение

    self.Noise = OUNoise(self.action_dim)

    self.noise_threshold = 1 # для уменьшения Noise
    self.noise_decrease = noise_decrease


    self.memory = deque(maxlen=memory_size)
    # self.memory = Buffer(memory_size, self.state_dim, self.action_dim)
    self.batch_size = batch_size
    self.gamma=  gamma
    self.tau = tau

    # помогают уменьшить колебания в процессе обучения, что способствует более стабильному и эффективному обучению
    self.target_actor = ThreeLayersNeuralNetwork(self.state_dim, 400, 300, self.action_dim, output_tanh = True, _lr=actor_lr, name='target_actor')
    self.target_critic = ThreeLayersNeuralNetwork(self.state_dim + self.action_dim, 400, 300, 1, output_tanh = False, _lr=critic_lr, name='target_critic')


  def get_action(self, state):
    pred_action = self.actor(torch.FloatTensor(state))
    action = self.action_scale * ( pred_action.detach().numpy() + self.noise_threshold * self.Noise.sample()) # лучше умножать Noise, тк тогда его будет больше
    return np.clip(action, -self.action_scale, self.action_scale)


  def update_target_model(self, target_model, model, optimizer, loss):
    optimizer.zero_grad()
    optimizer.step()

    for target_param, param in zip(target_model.parameters(), model.parameters()):
      target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data) # двигаем веса на маленькую величину



  def fit(self, state, action, reward, done, next_state):
    self.memory.append([state, action, reward, done, next_state])
    # q, pi = 0, 0
    if len(self.memory) > self.batch_size:
      batch = random.sample(self.memory, self.batch_size)
      states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch)) # сразу переводим все в тензоры
      rewards = rewards.reshape(self.batch_size, 1) # тк у rewards будет значение batch_size x 1
      dones = rewards.reshape(self.batch_size, 1)   # тк у rewards будет значение batch_size x 1

      pred_next_actions = self.action_scale * self.target_actor(next_states)
      next_states_and_pred_next_actions = torch.cat((next_states, pred_next_actions), dim=1)
      targets = rewards + self.gamma * self.target_critic(next_states_and_pred_next_actions)

      states_and_actions = torch.cat((states, actions), dim=1)
      q_loss = torch.mean((targets - self.critic(states_and_actions))**2)
      # q = q_loss
      self.update_target_model(self.target_critic, self.critic, self.critic.optimizer, q_loss)

      pred_actions = self.actor(states)
      states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
      pi_loss = -torch.mean(self.critic(states_and_pred_actions))
      # pi = pi_loss
      self.update_target_model(self.target_actor, self.actor, self.actor.optimizer, pi_loss)

      # self.writer.add_scalar('Loss/loss1', q_loss, i)
      # self.writer.add_scalar('Loss/loss2', pi_loss, i)

    if self.noise_threshold > 0:
      self.noise_threshold = max(0, self.noise_threshold - self.noise_decrease) # чтобы не было ситуации, когда noise отрицательный

    # return q, pi


  def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

  def load_models(self):
      self.actor.load_checkpoint()
      self.target_actor.load_checkpoint()
      self.critic.load_checkpoint()
      self.target_critic.load_checkpoint()


