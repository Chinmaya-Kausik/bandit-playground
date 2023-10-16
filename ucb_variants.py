import numpy as np
from mab_setup import *
from math import log, sqrt

class UCB(OnlineMABAlgorithm):
    
    def __init__(self, n_arms:int=5):
        super().__init__(n_arms)
        self.mean_rewards = np.ones(n_arms)*np.inf
        self.arm_count = np.zeros(n_arms)

    def choose_arm(self, step_num):
        t = step_num
        if(t>1):
            boosts = np.sqrt(2*(1+log(t*(log(t))**2))/self.arm_count)
        else:
            boosts = np.zeros(self.n_arms)
        return np.argmax(self.mean_rewards + boosts)
    
    def update(self, arm, reward, step_num):
        self.arm_count[arm] += 1
        if(self.mean_rewards[arm] == np.inf):
            self.mean_rewards[arm] = reward
        else:
            n_curr = self.arm_count[arm]
            self.mean_rewards[arm] *= n_curr/(n_curr+1)
            self.mean_rewards[arm] += reward/(n_curr+1)
        self.arm_count[arm] += 1
        
            
        