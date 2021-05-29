import numpy as np

class Irl:

	def __init__(self,weights,size,learning_rate):
		# weights -> (1,reward_size)
		self.weights = weights
		self.normalized_weights = weights
		self.size = size
		self.learning_rate = learning_rate

	def update(self,cand_rewards,ref_rewards,cand_probs):
		cand_rewards = np.array(cand_rewards).transpose()
		ref_rewards = np.array(ref_rewards).transpose()
		# cand_rewards -> (reward_size,batch_size)
		cand_total_reward = np.dot(self.weights,cand_rewards)
		
		# cand_total_reward -> (batch_size)
		importance_weights = np.exp(cand_total_reward)/cand_probs
		check_inf = np.isinf(importance_weights)
		importance_weights[check_inf]=0

		relax_weights = [False]*len(self.weights)

		for i in range(len(self.weights)):
			ref_part = np.mean(ref_rewards[i])
			cand_part = np.sum(np.multiply(importance_weights,cand_rewards[i]))/np.sum(importance_weights)
			delta_weight = self.learning_rate*(ref_part-cand_part)
			if delta_weight < 0.00001:
				relax_weights[i] = True
#			print(ref_part,cand_part,np.mean(cand_rewards[i]))
			self.weights[i] += delta_weight

		weights_sum = np.sum(self.weights)
		for i in range(len(self.weights)):
			self.normalized_weights[i] = self.weights[i]/weights_sum

		return relax_weights



