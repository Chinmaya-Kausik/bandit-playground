import numpy as np

class MABEnvironment:
    
    # Initializes the reward distribution, number of arms and list of parameters
    def __init__(self, reward_dist:str ="Bernoulli", 
                 mean_rewards:np.ndarray[int] = np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                 param_list:np.ndarray=np.array([])):
        self.reward_dist = reward_dist
        self.mean_rewards = mean_rewards
        self.n_arms = mean_rewards.size
        self.param_list = param_list
    
    # Takes one step by pulling the specified arm, returns the stochastic reward
    def step(self, arm):
        assert(arm in list(range(self.n_arms)), "Invalid arm {}, only {} arms".format(arm, self.n_arms))
        
        # Bernoulli rewards implemented
        if(self.reward_dist == "Bernoulli"):
            reward = np.random.binomial(1, self.mean_rewards[arm])
            
        return reward
    
    # Returns an array of regret values 
    def run_alg_regret(self, alg, steps = 500):
        
        if(self.reward_dist == 'Bernoulli'):
            best_arm = np.argmax(self.mean_rewards)
            best_mean_reward = self.mean_rewards[best_arm]
            
        reward_array = np.zeros(steps)
        arm_array = np.zeros(steps)
        regret_array = np.zeros(steps+1)
            
        for i in range(steps):
            next_arm = int(alg.choose_arm(i))
            arm_array[i] = next_arm
            reward = self.step(next_arm)
            reward_array[i] = reward
            additional_regret = (best_mean_reward - self.mean_rewards[next_arm])
            regret_array[i+1] = regret_array[i] + additional_regret
            alg.update(next_arm, reward, i)
        
        return regret_array, reward_array, arm_array
    
class OnlineMABAlgorithm:
    
    def __init__(self, n_arms:int = 5):
        self.n_arms = n_arms
        
    def choose_arm(self, step_num):
        raise NotImplementedError

    def update(self, arm, reward, step_num):
        raise NotImplementedError