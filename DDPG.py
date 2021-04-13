import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actors(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actors, self).__init__()

		self.p1_1 = nn.Linear(state_dim,1024)
		self.p1_2 = nn.Linear(1024,512)
		self.p1_3 = nn.Linear(512,256)
		self.p1_4 = nn.Linear(256,128)
		self.p1_5 = nn.Linear(128,action_dim)

		torch.nn.init.normal_(self.p1_1.weight.data,std=0.01)
		torch.nn.init.normal_(self.p1_2.weight.data,std=0.01)
		torch.nn.init.normal_(self.p1_3.weight.data,std=0.01)
		torch.nn.init.normal_(self.p1_4.weight.data,std=0.01)
		torch.nn.init.normal_(self.p1_5.weight.data,std=0.01)
	
		torch.nn.init.zeros_(self.p1_1.bias.data)
		torch.nn.init.zeros_(self.p1_2.bias.data)
		torch.nn.init.zeros_(self.p1_3.bias.data)
		torch.nn.init.zeros_(self.p1_4.bias.data)
		torch.nn.init.zeros_(self.p1_5.bias.data)

	def forward(self, state):
		p1 = F.leaky_relu(self.p1_1(state))
		p1 = F.leaky_relu(self.p1_2(p1))
		p1 = F.leaky_relu(self.p1_3(p1))
		p1 = F.leaky_relu(self.p1_4(p1))
		return	self.p1_5(p1)


class Critics(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critics, self).__init__()

		self.c1_1 = nn.Linear(state_dim+action_dim,1024)
		self.c1_2 = nn.Linear(1024,512)
		self.c1_3 = nn.Linear(512,256)
		self.c1_4 = nn.Linear(256,128)
		self.c1_5 = nn.Linear(128,1)

		torch.nn.init.normal_(self.c1_1.weight.data,std=0.01)
		torch.nn.init.normal_(self.c1_2.weight.data,std=0.01)
		torch.nn.init.normal_(self.c1_3.weight.data,std=0.01)
		torch.nn.init.normal_(self.c1_4.weight.data,std=0.01)
		torch.nn.init.normal_(self.c1_5.weight.data,std=0.01)

		torch.nn.init.zeros_(self.c1_1.bias.data)
		torch.nn.init.zeros_(self.c1_2.bias.data)
		torch.nn.init.zeros_(self.c1_3.bias.data)
		torch.nn.init.zeros_(self.c1_4.bias.data)
		torch.nn.init.zeros_(self.c1_5.bias.data)

	def forward(self, state, action):
		inp1 = torch.cat((state,action),1)
		c1 = F.leaky_relu(self.c1_1(inp1))
		c1 = F.leaky_relu(self.c1_2(c1))
		c1 = F.leaky_relu(self.c1_3(c1))
		c1 = F.leaky_relu(self.c1_4(c1))

		return self.c1_5(c1)


class DDPG(object):
	def __init__(self,state_dim,action_dim,max_action,min_action,discount=0.99,tau=1e-4):
		
		self.actors = Actors(state_dim, action_dim).to(device)
		self.actors_target = copy.deepcopy(self.actors)
		self.actors_optimizer = torch.optim.Adam(self.actors.parameters(),lr=1e-3)

		self.critics = Critics(state_dim, action_dim).to(device)
		self.critics_target = copy.deepcopy(self.critics)
		self.critics_optimizer = torch.optim.Adam(self.critics.parameters(),lr=1e-3)
		
		self.discount = discount
		self.tau = tau

		self.max_p = torch.FloatTensor(max_action).to(device)
		self.min_p = torch.FloatTensor(min_action).to(device)
		self.rng = (self.max_p - self.min_p).detach()
	
	def invert_gradient(self,delta_a,current_a):
		index = delta_a>0
		delta_a[index] *=  (index.float() * (self.max_p - current_a)/self.rng)[index]
		delta_a[~index] *= ((~index).float() * (current_a- self.min_p)/self.rng)[~index]

		return delta_a	

	def select_action(self,state):
		state = torch.FloatTensor(state.reshape(1,-1)).to(device)
		p = self.actors(state)
		np_max = self.max_p.cpu().data.numpy()
		np_min = self.min_p.cpu().data.numpy()

		return np.clip(p.cpu().data.numpy().flatten(),np_min,np_max)


	def train(self,replay_buffer, batch_size=64):
		state,action, next_state, reward, n_step, not_done = replay_buffer.sample(batch_size)
		
		target_Q = self.critics_target(next_state,self.actors_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()
		current_Q = self.critics(state, action)
		beta = 0.2
		mixed_q = beta*n_step + (1-beta)*target_Q
		critic_loss = F.mse_loss(current_Q, mixed_q)

		self.critics_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10)
		self.critics_optimizer.step()

		current_a = Variable(self.actors(state))
		current_a.requires_grad = True

		actor_loss = self.critics(state, current_a).mean()

		self.critics.zero_grad()
		actor_loss.backward()
		delta_a = copy.deepcopy(current_a.grad.data)
		delta_a = self.invert_gradient(delta_a,current_a)
		current_a = self.actors(state)
		out = -torch.mul(delta_a,current_a)
		self.actors.zero_grad()
		out.backward(torch.ones(out.shape).to(device))
		torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 10)
		self.actors_optimizer.step()

		for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actors.parameters(), self.actors_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critics.state_dict(), filename + "_critic")
		torch.save(self.critics_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actors.state_dict(), filename + "_actor")
		torch.save(self.actors_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critics.load_state_dict(torch.load(filename + "_critic"))
		self.critics_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critics_target = copy.deepcopy(self.critics)

		self.actors.load_state_dict(torch.load(filename + "_actor"))
		self.actors_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actors_target = copy.deepcopy(self.actors)
