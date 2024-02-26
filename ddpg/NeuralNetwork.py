import torch
import torch.nn
import torch.optim as optim
import os


class ThreeLayersNeuralNetwork(nn.Module):
  def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim, output_tanh, _lr, name, chkpt_dir="tmp/ddpg"):
    super(ThreeLayersNeuralNetwork, self).__init__()
    self.checkpoint_dir = chkpt_dir
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    # self.name  = name
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

    self.layer1 = nn.Linear(input_dim, layer1_dim)
    # self.bn1 = nn.BatchNorm1d(layer1_dim)
    self.bn1 = nn.LayerNorm(layer1_dim)

    self.layer2 = nn.Linear(layer1_dim, layer2_dim)
    # self.bn2 = nn.BatchNorm1d(layer2_dim)
    self.bn2 = nn.LayerNorm(layer2_dim)

    self.layer3 = nn.Linear(layer2_dim, output_dim)

    self.output_tanh = output_tanh
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

    self.optimizer = optim.Adam(self.parameters(), lr=_lr, weight_decay=0.01)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.to(self.device)


  def forward(self, input):
    hidden = self.layer1(input)
    # print(hidden.shape)
    hidden = self.relu(hidden)

    hidden = self.bn1(hidden)
    # hidden = self.relu(hidden)

    hidden = self.layer2(hidden)
    hidden = self.relu(hidden)

    hidden = self.bn2(hidden)
    # hidden = self.relu(hidden)

    hidden = self.layer3(hidden)
    # hidden = self.bn3(hidden)
    output = self.relu(hidden)

    if self.output_tanh:
      return self.tanh(output)
    else:
      return output

  def save_checkpoint(self):
      print('... saving checkpoint ...')
      torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
      print('... loading checkpoint ...')
      self.load_state_dict(torch.load(self.checkpoint_file))
