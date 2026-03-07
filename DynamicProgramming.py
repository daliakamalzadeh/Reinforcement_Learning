#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        return argmax(self.Q_sa[s,:])
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        old_q = self.Q_sa[s, a]
        v_next = np.max(self.Q_sa, axis=1)  # v(s') = max_a' Q(s',a') for all s'
        new_q = np.sum(p_sas * (r_sas + self.gamma * v_next))
        self.Q_sa[s, a] = new_q
        return abs(new_q - old_q)
            
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

    # Q-value iteration (Alg. 1): repeatedly sweep through all (s,a) and update
    i = 0
    max_error = np.inf
    while max_error > threshold:
        max_error = 0.0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s, a)
                err = QIagent.update(s, a, p_sas, r_sas)
                if err > max_error:
                    max_error = err
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(i, max_error))
        i += 1
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

    # Compute mean reward per timestep under the optimal policy

    n_episodes = 100
    total_reward = 0
    total_timesteps = 0

    for _ in range(n_episodes):
        s = env.reset()
        done = False
        
        while not done:
            a = QIagent.select_action(s)   # greedy action
            s_next, r, done = env.step(a)
            
            total_reward += r
            total_timesteps += 1
            
            s = s_next

    mean_reward_per_timestep = total_reward / total_timesteps

    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    S
    # For 1.4 part c.3
    start_state = 3 
    V_star_start = np.max(QIagent.Q_sa[start_state])
    print("Optimal value at start state (s=3):", V_star_start)

    # For 1.4 part c.4

    V = V_star_start # optimal value at the start state
    T_expected = 101 - V # the magnitude of the terminal reward, gives the expected number of steps to reach the goal under the optimal policy.
    G_expected = 100 - (T_expected - 1)  # the magnitude of the reward on every other step, gives the expected total reward until reaching the goal under the optimal policy.
    mean_reward_derived = G_expected / T_expected 

    print("E[T]:", T_expected)
    print("E[G]:", G_expected)
    print("Mean reward per timestep (derived):", mean_reward_derived)
    
if __name__ == '__main__':
    experiment()
