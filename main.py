# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle


def run(episodes,render=False):
        env=gym.make("FrozenLake-v1",map_name="8x8",is_slippery=False,render_mode='human'if render else None )

        q=np.zeros((env.observation_space.n,env.action_space.n))#init a 64X4 array

        learn_rate_a=0.9 # alpha or learn rate
        discount_factor_g=0.9 #gama or discount factor

        # Epsilon-greedy algorithm
        epsilon=1   #1=100% random action
        epsilon_decay_rate=0.0001 #epsilon decay rate 1/0.0001=10.000 times
        rng=np.random.default_rng() #random number generator
        #print('check')

        reward_per_episode = np.zeros(episodes)  # tracking the process

        for i in range (episodes):
            state = env.reset()[0]  # states: 0 to 63 ,0=top left ,63= bottom corner
            terminated = False
            truncated = False  # 截断

            while (not terminated and not truncated):
                print('Im inside the while')
                if rng.random()<epsilon:
                    action=env.action_space.sample() # take the random action
                else :
                    action=np.argmax(q[state,:]) # follow the q table

                # action = env.action_space.sample()  # actions: 0-left, 1 down , 2-right, 3 up
                new_state, reward, terminated, truncated, _ = env.step(action)
                q[state, action] = q[state, action] + learn_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
                state=new_state
            epsilon=max(epsilon-epsilon_decay_rate,0) # decrease epsilon all the way until it gets to zero

            if epsilon==0:
                  learn_rate_a=0.0001
                  print('Im inside the if')
            if reward==1:
                  reward_per_episode[i]= 1 # tracking
            env.close()

        sum_rewards=np.zeros(episodes) #plot a graph
        for t in range(episodes):
            sum_rewards[t]=np.sum(reward_per_episode[max(0,t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_xiao8x8.png')

        f=open('frozen_xiao8x8.pkl','wb')
        pickle.dump(q,f)
        f.close()

if __name__=='__main__':
        run(15000)