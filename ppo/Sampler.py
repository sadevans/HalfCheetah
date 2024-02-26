import numpy as np


class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = self.runner.get_next()
        for transform in self.transforms:
                transform(self.trajectory)
    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.
            Should be called at the beginning of each epoch.
        """
        pass
    
    def get_next(self):
        """ Returns next minibatch.  """
        if self.epoch_count==self.num_epochs:
            self.trajectory = self.runner.get_next()
            for transform in self.transforms:
                transform(self.trajectory)
            self.epoch_count = 0
        minibatch_dict = {}
        rand_inds = np.random.randint(0, self.trajectory['state']['env_steps'], self.num_minibatches)
        for key, value in self.trajectory.items():
            if key!='state':
                if len(value)==2:
                    minibatch_dict[key] = self.trajectory[key][rand_inds,:]
                else:
                    minibatch_dict[key] = self.trajectory[key][rand_inds]
        self.epoch_count += 1
        return minibatch_dict