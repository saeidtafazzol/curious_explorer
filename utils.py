import numpy as np
import torch

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.exp_reward = np.zeros((max_size, 1))
		self.n_step = np.zeros((max_size, 1))
		self.exp_n_step = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self,state,action,next_state,reward,ex_rew,n_step,ex_n_step,done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.exp_reward[self.ptr] = ex_rew
		self.n_step[self.ptr] = n_step
		self.exp_n_step[self.ptr] = ex_n_step
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.exp_reward[ind]).to(self.device),
			torch.FloatTensor(self.n_step[ind]).to(self.device),
			torch.FloatTensor(self.exp_n_step[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def save(self,folder):
		f = open(folder+'/params','w')
		f.write(str(self.max_size)+','+str(self.ptr)+','+str(self.size))
		f.close()
		np.save(folder+'/state',self.state)
		np.save(folder+'/action',self.action)
		np.save(folder+'/next_state',self.next_state)
		np.save(folder+'/reward',self.reward)
		np.save(folder+'/not_done',self.not_done)
	
	def load(self,folder):
		f = open(folder+'/params','r')
		a = f.read()
		a = a.split(',')
		self.max_size,self.ptr,self.size = int(a[0]),int(a[1]),int(a[2])
		f.close()
		self.state = np.load(folder+'/state.npy')
		self.action = np.load(folder+'/action.npy')
		self.next_state = np.load(folder+'/next_state.npy')
		self.reward = np.load(folder+'/reward.npy')
		self.not_done = np.load(folder+'/not_done.npy')