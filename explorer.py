import policy_explorer
import predictor
import utils
import random
import numpy as np

class explorer(object):
	def __init__(self,state_dim,action_dim,max_action,min_action,discount=0.99,tau=1e-4):

		self.min_action = min_action
		self.max_action = max_action

		self.ddpg = policy_explorer.DDPG_EXPLORER(state_dim, action_dim, max_action, min_action)
		self.predictor = predictor.Predictor(state_dim,action_dim)
        
		self.counter = 0
    
	def train(self, replay_buffer, batch_size=64):

		return (self.ddpg.train(replay_buffer, batch_size), 
				self.predictor.train(replay_buffer, batch_size))

	def select_action(self, state):

		self.counter += 1
		eps_rnd = random.random()
		dec = min(max(0.1,1.0 - float(self.counter)*0.00003),1)
		
		if eps_rnd<dec:
			action = np.random.uniform(self.min_action, self.max_action)
		else:
			action = self.ddpg.select_action(state)
		return action

	def predict(self, state, action):
		return self.predictor.predict(state, action)
