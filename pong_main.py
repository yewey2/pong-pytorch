import datetime

print('Start time:', datetime.datetime.now())

import numpy as np
import os
import pygame

from Env import Env, RENDERING

env = Env()
state_size = 5
action_size = 3
batch_size = 64
n_episodes = 1000
output_dir = './data-pong/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from PongAgent import PongAgent
agent = PongAgent(state_size, action_size, learning_rate=0.0008)

count=0
e=0

while True:
    e+=1
    # Learning Loop
    state=env.reset() 
    state=np.reshape(state,[1, state_size])
    for t in range(5000):
        if RENDERING:
            env.render()
        action = agent.act(state)
        next_state, reward, done = env.runframe(action)
        next_state=np.reshape(next_state,[1, state_size])
        if e < 10_000: # only train on old values
            agent.remember(state,action,reward,next_state,done)
        state=next_state
        if done:
            if e%50 == 1:
                print(f'episode: {e}/{n_episodes},\ttime: {t},\tscore: {env.count},\tepsilon: {agent.epsilon:.3} \t LR: {agent.learning_rate:.3}')
            break
        
    
    if e%5 == 0:
        agent.update_target_model()

    # Learning Algorithm in Replay, updates agent.model
    if len(agent.memory)>batch_size:
        agent.replay(batch_size)

    if e%200 == 0: # Save file every 200 episodes
        print('Saving', e,'epsilon:',round(agent.epsilon,3))
        agent.save('{}agent_{:07d}.hdf5'.format(output_dir,e))


    # Check if agent is winning, i.e. bounced the ball more than 20 times
    count+= 1 if env.count>=20 else -1 if count>10 else -count
    if count>5:
        print('won', count, 'consecutive games')
    if count>30: # If agent "won" the game > 30 times, save the agent and stop training
        agent.save(f'{output_dir}/pong_best{e}.hdf5')
        break

if RENDERING:
    pygame.quit()
