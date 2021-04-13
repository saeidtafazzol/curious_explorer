import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Network, self).__init__()
        self.p1 = nn.Linear(state_dim+action_dim,1024)
        self.p2 = nn.Linear(1024,512)
        self.p3 = nn.Linear(512,256)
        self.p4 = nn.Linear(256,128)
        self.p5 = nn.Linear(128,state_dim+1)
        torch.nn.init.normal_(self.p1.weight.data,std=0.01)
        torch.nn.init.normal_(self.p2.weight.data,std=0.01)
        torch.nn.init.normal_(self.p3.weight.data,std=0.01)
        torch.nn.init.normal_(self.p4.weight.data,std=0.01)
        torch.nn.init.normal_(self.p5.weight.data,std=0.01)
        torch.nn.init.zeros_(self.p1.bias.data)
        torch.nn.init.zeros_(self.p2.bias.data)
        torch.nn.init.zeros_(self.p3.bias.data)
        torch.nn.init.zeros_(self.p4.bias.data)
        torch.nn.init.zeros_(self.p5.bias.data)

    def forward(self, state, action):
        inp = torch.cat((state,action),1)
        c = F.leaky_relu(self.p1(inp))
        c = F.leaky_relu(self.p2(c))
        c = F.leaky_relu(self.p3(c))
        c = F.leaky_relu(self.p4(c))

        return self.p5(c)
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
        loss = F.mse_loss(self.net(state,action),torch.cat((next_state,reward),1))


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

