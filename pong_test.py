import numpy as np
import os
import Env

env = Env.Env()
state_size = 5
action_size = 3
batch_size = 64
n_episodes = 1000
output_dir = './data-pong/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from PongAgent import PongAgent

agent = PongAgent(state_size, action_size, learning_rate=0.0008)
agent.load_reset(f'{output_dir}/pong_best.hdf5')

count=0
e=0

while True:
    e+=1
    # Learning Loop
    state=env.reset() 
    state=np.reshape(state,[1, state_size])
    for t in range(5000):
        if Env.RENDERING:
            env.render()
        action = agent.act(state)
        next_state, reward, done = env.runframe(action)
        next_state=np.reshape(next_state,[1, state_size])
        agent.remember(state,action,reward,next_state,done)
        state=next_state
        if done:
            if e%50 == 1:
                print('episode: {}/{},\ttime: {},\tscore: {},\tepsilon: {:.3}'.format(e,n_episodes,t,env.count,agent.epsilon))
            break
