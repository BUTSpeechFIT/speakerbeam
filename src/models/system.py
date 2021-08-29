# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from asteroid.engine import System

class SystemInformed(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls = batch
        est_targets = self(inputs, enrolls)
        loss = self.loss_func(est_targets, targets)
        return loss
