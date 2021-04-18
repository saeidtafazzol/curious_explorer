from torch import nn, cat
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim,1024)
		self.l2 = nn.Linear(1024,512)
		self.l3 = nn.Linear(512,256)
		self.l4 = nn.Linear(256,128)
		self.l5 = nn.Linear(128,action_dim)

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

	def forward(self, state):
		out = F.leaky_relu(self.l1(state))
		out = F.leaky_relu(self.l2(out))
		out = F.leaky_relu(self.l3(out))
		out = F.leaky_relu(self.l4(out))
		return	self.l5(out)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim+action_dim,1024)
		self.l2 = nn.Linear(1024,512)
		self.l3 = nn.Linear(512,256)
		self.l4 = nn.Linear(256,128)
		self.l5 = nn.Linear(128,1)

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