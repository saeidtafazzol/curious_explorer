import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from AC import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG_EXPLORER(object):
	def __init__(self,state_dim,action_dim,max_action,min_action,discount=0.99,tau=1e-4):
		
		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=1e-3)
		
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
		p = self.actor(state)
		np_max = self.max_p.cpu().data.numpy()
		np_min = self.min_p.cpu().data.numpy()
		return np.clip(p.cpu().data.numpy().flatten(),np_min,np_max)


	def train(self,replay_buffer, batch_size=64):
		state,action, next_state, reward, ex_reward, n_step, ex_n_step, not_done = replay_buffer.sample(batch_size)
		
		target_Q = self.critic_target(next_state,self.actor_target(next_state))
		target_Q = ex_reward + (not_done * self.discount * target_Q).detach()
		current_Q = self.critic(state, action)
		beta = 0.2
		mixed_q = beta*ex_n_step + (1-beta)*target_Q
		critic_loss = F.mse_loss(current_Q, mixed_q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
		self.critic_optimizer.step()

		current_a = Variable(self.actor(state))
		current_a.requires_grad = True

		actor_loss = self.critic(state, current_a).mean()

		self.critic.zero_grad()
		actor_loss.backward()
		delta_a = copy.deepcopy(current_a.grad.data)
		delta_a = self.invert_gradient(delta_a,current_a)
		current_a = self.actor(state)
		out = -torch.mul(delta_a,current_a)
		self.actor.zero_grad()
		out.backward(torch.ones(out.shape).to(device))
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
		self.actor_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)