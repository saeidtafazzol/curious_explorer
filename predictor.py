import copy
import numpy as np
import torch
from torch import nn, cat
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Network, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim,1024)
        self.l2 = nn.Linear(1024,512)
        self.l3 = nn.Linear(512,256)
        self.l4 = nn.Linear(256,128)
        self.l5 = nn.Linear(128,state_dim+1)

        nn.init.normal_(self.l1.weight.data,std=0.01)
        nn.init.normal_(self.l2.weight.data,std=0.01)
        nn.init.normal_(self.l3.weight.data,std=0.01)
        nn.init.normal_(self.l4.weight.data,std=0.01)
        nn.init.normal_(self.l5.weight.data,std=0.01)

        nn.init.zeros_(self.l1.bias.data)
        nn.init.zeros_(self.l2.bias.data)
        nn.init.zeros_(self.l3.bias.data)
        nn.init.zeros_(self.l4.bias.data)
        nn.init.zeros_(self.l5.bias.data)

    def forward(self, state, action):
        inp = cat((state,action),1)
        out = F.leaky_relu(self.l1(inp))
        out = F.leaky_relu(self.l2(out))
        out = F.leaky_relu(self.l3(out))
        out = F.leaky_relu(self.l4(out))
        return self.l5(out)

class Predictor(object):
    def __init__(self,state_dim,action_dim):
        self.net = Network(state_dim,action_dim).to(device)
        self.net_optimizer = torch.optim.Adam(self.net.parameters())

    def predict(self,state,action):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action = torch.FloatTensor(action.reshape(1,-1)).to(device)

        return self.net(state,action).cpu().data.numpy().flatten()

    def train(self,replay_buffer, batch_size=64):
        state,action, next_state, reward, ex_reward, n_step, ex_n_step, not_done = replay_buffer.sample(batch_size)
        loss = F.mse_loss(self.net(state,action), torch.cat((next_state, reward), 1))

        self.net_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net_optimizer.step()

        return loss

    def save(self, filename):
        torch.save(self.net.state_dict(), filename + "_predictor")
        torch.save(self.net_optimizer.state_dict(), filename + "_predictor_optimizer")

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename + "_predictor"))
        self.net_optimizer.load_state_dict(torch.load(filename + "_predictor_optimizer"))

