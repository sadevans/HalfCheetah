class OUNoise:
  def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
    """
    theta - стохастический сдвиг относительно среднего
    sigma - variance в нормальном распределении

    """
    self.action_dimension = action_dimension
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.state = np.ones(self.action_dimension) * self.mu # все действия = mu
    self.reset()

  def reset(self):
    self.state = np.ones(self.action_dimension) * self.mu

  def sample(self):
    """
    решение стохастического диффура
    """
    x=  self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
    self.state = x + dx
    return self.state