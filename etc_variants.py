import numpy as np
from mab_setup import *

class ETC(OnlineMABAlgorithm):
    
    def __init__(self, n_arms:int=5, explore_ct:int = 30):
        super().__init__(n_arms)
        self.explore_ct = explore_ct
        self.etc_reward_list = np.zeros(n_arms)
        
    def choose_arm(self, step_num):
        if(step_num <= self.n_arms*self.explore_ct):
            arm = step_num%self.n_arms
        else:
            arm = np.argmax(self.etc_reward_list)
        return arm
        
    def update(self, arm, reward, step_num):
        if(step_num<=self.n_arms*self.explore_ct):
            self.etc_reward_list[arm] += (reward)/(self.explore_ct)
            
            