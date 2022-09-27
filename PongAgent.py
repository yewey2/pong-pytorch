import random, math, copy
import numpy as np
from collections import deque
import shelve

import torch
import torch.optim as optim
from PongModel import Model

output_dir = './data-pong/'

GAMMA = 0.95

class PongAgent:
    def __init__(self,state_size,action_size,learning_rate=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=100000)
        
        self.gamma = GAMMA
        self.epsilon = 0.999 # exploitation=0 vs exploration=1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001 # 0.1% exploration
        self.learning_rate = learning_rate

        # Adjusting learning rate for cyclic learning rate
        self.learning_rate_max = 0.0008
        self.learning_rate_min = 0.0001
        self.learning_rate_decay = 0.999

        self.model = self._build_model()
        self.model_tar = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        self.batch_size = batch_size
        self.O = np.ones((batch_size,self.state_size))
        self.Next_O = np.ones((batch_size,self.state_size))

    def _build_model(self, hidden_dims = 64):
        model = Model(self.state_size,hidden_dims,self.action_size)
        model = model.cuda()
        return model
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return(random.randint(0, self.action_size -1))
        act_values = self.model(torch.Tensor(state).cuda())
        return np.argmax(act_values.cpu().data[0])
        
    def replay(self,batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            q_values = self.model(torch.Tensor(state).cuda())
            target_q_values = self.model_tar(torch.Tensor(next_state).cuda()).max(dim = 1)[0]
            target_q_values = np.array(target_q_values.cpu().data)
            
            expected_q = np.array(q_values.cpu().data)
            expected_q[0][action] = reward + (1-done)*self.gamma*target_q_values
            
            loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() # Mean square error loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

        # Learning rate decay
        # if self.learning_rate>self.learning_rate_min:        
        #     self.learning_rate*=self.learning_rate_decay

        #Cyclic learning rate (more efficient)
        if self.learning_rate_decay>1 and self.learning_rate > self.learning_rate_max:
            self.learning_rate_decay = 1/self.learning_rate_decay
        if self.learning_rate_decay<1 and self.learning_rate < self.learning_rate_min:
            self.learning_rate_decay = 1/self.learning_rate_decay
        self.learning_rate*=self.learning_rate_decay
 
    def update_target_model(self):
        self.model_tar.load_state_dict(self.model.state_dict())

    def save(self,name):
        torch.save(self.model,name)
 
    def load(self,name):
        print(f'Loading {name}')
        self.model = torch.load(name) 

    def load_reset(self, filename):
        self.load(filename)
        self.epsilon = 0.0
 
    def load_memory(self, episode_num):
        with shelve.open(f'{output_dir}memory.pickle') as file:
            self.memory = file.get(f'{episode_num}') or self.memory

    def load_all(self, episode_num):
        self.load_reset('{}agent_{:05d}.hdf5'.format(output_dir,episode_num))
        self.load_memory(episode_num)
 