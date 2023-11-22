# %%
#config
resume = False
render = False
save_freq = 10000
resume_pkl = "Q.pkl"
max_episodes = 300000

# %%
import gym
# env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env = gym.make('PongDeterministic-v4', render_mode='rgb_array')

# %%
import cv2
import numpy as np
import random
import math
import time
import pickle

# %%
obv = env.reset()
obv[1]


# %%
observation = env.reset()
cumulated_reward = 0

# frames = []
# for t in range(1000):
# #     print(observation)
#     frames.append(env.render())
#     # very stupid agent, just makes a random action within the allowd action space
#     action = env.action_space.sample()
# #     print("Action: {}".format(t+1))
#     observation, reward, done, truncated, info = env.step(action)
# #     print(reward)
#     cumulated_reward += reward
#     if done:
#         print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
#         break
# print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))

# env.close()

# %%
# for frame in frames:
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray_frame)
#     cv2.waitKey(1)

# %%
# frames[200].shape

# %%
# greyframe = cv2.cvtColor(frames[300], cv2.COLOR_BGR2GRAY)

# %%
# greyframe

# %%
# np.unique(cv2.cvtColor(frames[200][34:], cv2.COLOR_BGR2GRAY))

# %%
# gray_frame = cv2.cvtColor(frames[-1][34:-16], cv2.COLOR_BGR2GRAY)
# #shape: (210-25=185 , 160)

# unique_values = [64, 123, 147, 236]
# coordinates = {}

# for value in unique_values:
#     coordinates[value] = np.where(gray_frame == value)


# # for value, (x, y) in coordinates.items():
# #     print(f"Value: {value}, Coordinates: {list(zip(x, y))}")
# #     print(f"Value: {value}, Coordinates: {len(list(zip(x, y)))}")
# #     print()
# # #bg
# print(f"Value: {64}, Coordinates: {(coordinates[64][0][0]+coordinates[64][0][-1])/2, (coordinates[64][1][0]+coordinates[64][1][-1])/2}")
# #left
# print(f"Value: {123}, Coordinates: {(coordinates[123][0][0]+coordinates[123][0][-1])/2, (coordinates[123][1][0]+coordinates[123][1][-1])/2}")
# #right
# print(f"Value: {147}, Coordinates: {(coordinates[147][0][0]+coordinates[147][0][-1])/2, (coordinates[147][1][0]+coordinates[147][1][-1])/2}")
# #ball
# print(f"Value: {236}, Coordinates: {(coordinates[236][0][0]+coordinates[236][0][-1])/2, (coordinates[236][1][0]+coordinates[236][1][-1])/2}")

# %%
env.action_space

# %%
# state = env.reset()
# done = False
# no_steps = 0
# while not done:
#     a = random.randint(0, 5)
#     state, reward, done,tun, info = env.step(a)
#     no_steps += 1

# print(no_steps)


# %%
# print(state.shape)

# %%
def discretizer(state):
    greyimg = cv2.cvtColor(state[34:-16], cv2.COLOR_BGR2GRAY)
    unique_values = [123, 147, 236]
    coordinates = {}
    for value in unique_values:
        coordinates[value] = np.where(greyimg == value)
    #left center
    #print(coordinates[123], coordinates[147], coordinates[236])
    if coordinates[123][0].any():
        lc = (coordinates[123][0][0]+coordinates[123][0][-1])/2
    else:
        lc = -1
    #right center
    if coordinates[147][0].any():
        rc = (coordinates[147][0][0]+coordinates[147][0][-1])/2
    else:
        rc = -1
    #ball center
    if coordinates[236][0].any():
        bc = ((coordinates[236][0][0]+coordinates[236][0][-1])/2, (coordinates[236][1][0]+coordinates[236][1][-1])/2)
    else:
        bc = (-1,-1)
    
    return int(lc)//2, int(rc)//2, int(bc[0])//2,int(bc[1])//2

# %%
# discretizer(state)

# %%
# frames[200].shape

# %%
# discretizer(frames[200])

# %%
Q = np.zeros([160//2, 160//2, 160//2, 160//2, 6],dtype=np.uint8)


# %%
Q.nbytes
#31457280000 default dtype 
#3932160000 unit8 
#245760000 with//2

# %%
def policy(state:tuple):
    return np.argmax(Q[state])
def new_Q_value(reward: float, state_new: tuple, discount_factor=1):
    future_optimal_value = np.max(Q[state_new])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

def learning_rate(n:int, min_rate=0.01):
    return max(min_rate, min(1.0, 1.0-math.log10((n+1)/25)))

def exploration_rate(n:int, min_rate=0.1):
    return max(min_rate, min(1, 1.0-math.log10((n+1)/25)))

# %%
print(env.reset()[0].shape)

# %%
from tqdm import tqdm

# %%
n_episodes = max_episodes

if not resume:
    for e in tqdm(range(n_episodes)):
        # print(*env.reset())
        # print(*env.reset[-1])
        current_state, done = discretizer(env.reset()[0]), False
        while done == False:
            action = policy(current_state)
            if np.random.random() < exploration_rate(e):
                action = env.action_space.sample()
            #print(env.step(action)[0].shape)
            obs, reward, done,tun, _ = env.step(action)
            # print("obs:",obs[0])
            new_state = discretizer(obs)
            
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward, new_state)
            old_value = Q[current_state][action]
            Q[current_state][action] = (1-lr)*old_value + lr*learnt_value
            
            current_state = new_state

        if e%save_freq == 0:
            with open("Q%d.pkl"%e,"wb") as f:
                pickle.dump(Q,f)

else:
    frames = []
    with open(resume_pkl,"rb") as f:
        Q = pickle.load(f)
    for e in tqdm(range(n_episodes)):
        # print(*env.reset())
        # print(*env.reset[-1])
        current_state, done = discretizer(env.reset()[0]), False
        while done == False:
            action = policy(current_state)
            if np.random.random() < exploration_rate(e):
                action = env.action_space.sample()
            #print(env.step(action)[0].shape)
            obs, reward, done,tun, _ = env.step(action)
            # print("obs:",obs[0])
            new_state = discretizer(obs)
            
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward, new_state)
            old_value = Q[current_state][action]
            Q[current_state][action] = (1-lr)*old_value + lr*learnt_value
            
            current_state = new_state
            frames.append(obs)
        if e%save_freq == 0:
            with open("Q%d.pkl"%e,"wb") as f:
                pickle.dump(Q,f)

env.close()

# %%
Q.nbytes

# %%

with open("Q.pkl", "wb") as f:
    pickle.dump(Q, f)


# %%
if render:
    for frame in frames:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


