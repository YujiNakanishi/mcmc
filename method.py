import numpy as np

def Metropolis(num, prob_func, x, c):
	"""
	MCMC sampling by Metropolis

	input:
		num -> <int> sampling number
		prob_func -> <function>
		x -> <np:float:(D, )> initial point
		c -> <float> step size
	output:
		sample -> <np:float:(num, D)> 
	"""
	dim = len(x)
	x_new = x - c + 2.*c*np.random.rand(dim)
	if (prob_func(x_new)/prob_func(x)) > np.random.rand():
		sample = np.stack((x, x_new), axis = 0)
	else:
		sample = np.stack((x, x), axis = 0)

	for itr in range(num-2):
		x_new = sample[-1] - c + 2.*c*np.random.rand(dim)
		if (prob_func(x_new)/prob_func(sample[-1])) > np.random.rand():
			sample = np.concatenate((sample, np.expand_dims(x_new, axis = 0)))
		else:
			sample = np.concatenate((sample, np.expand_dims(sample[-1], axis = 0)))

	return sample