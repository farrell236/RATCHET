import numpy as np


class CustomSchedule(object):
    def __init__(self, _d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = _d_model
        self.warmup_steps = warmup_steps

    def adjust_learning_rate(self, optim, step):
        arg1 = np.reciprocal(np.sqrt(step))
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = np.reciprocal(np.sqrt(self.d_model)) * np.minimum(arg1, arg2)
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        return lr
