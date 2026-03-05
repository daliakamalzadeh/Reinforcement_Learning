#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q_sa[s_next])

        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    # sample initial state
    s = env.reset()


    # while budget do
    for t in range(n_timesteps):

        # sample action
        a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

        # simulate environment
        step_out = env.step(a)
        if len(step_out) == 3:
            s_next, r, done = step_out
        else:
            s_next, r, done, _ = step_out

        # Q update
        agent.update(s, a, r, s_next, done)

        # reset environment if episode ended
        if done:
            s = env.reset()
        else:
            s = s_next

        # evaluate
        if (t + 1) % eval_interval == 0:
            eval_returns.append(agent.evaluate(eval_env))
            eval_timesteps.append(t + 1)
    if plot:
       env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution
       import matplotlib.pyplot as plt
       plt.show(block=True)

    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
