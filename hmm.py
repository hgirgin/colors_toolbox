#!/usr/bin/env python
import numpy as np
import numpy.testing as npt

from sklearn.cluster import KMeans
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
np.set_printoptions(precision=3)

class hmm:

	def __init__(self,number_of_hidden_states, training_data):

		self.n          = number_of_hidden_states
		self.data       = training_data
		self.dataLength = self.data.shape[0]
		self.dataDimens = self.data.shape[1]


	def initPrior(self):
		# Uniform Initialization of Priors
		prior = np.ones(self.n)*(1.0/self.n)
		return np.log(prior)

	def initTransition(self):

		# Uniform Initialization of Transition Matrix
		a = np.ones((self.n,self.n))*(1.0/self.n)

		return np.log(a)

	def initMeansCovs(self):
		
		kmeans = KMeans(n_clusters=self.n, random_state=0).fit(self.data)
		mu = kmeans.cluster_centers_
		labels = kmeans.labels_
		mu = np.array(sorted(mu.tolist()))
		labels = np.array(sorted(labels.tolist()))


		covariances = np.array([np.eye(self.dataDimens) for i in range(self.n)])

		return mu,covariances



	def observationProb(self,mean,covars):


		b = np.zeros((self.n,self.dataLength))
		for i in range(self.n):
			b[i,:] = multivariate_normal.pdf(self.data,mean[i],covars[i], allow_singular=True)

		return np.log(b)



	def forward(self, prior, a, b):

		alpha = np.ones((self.n,self.dataLength))*np.log(0)

		# Initialization Step
		alpha[:,0] = prior+b[:,0]

		# Recursion Step
		for t in range(1,self.dataLength):
			for j in range(0,self.n):
				alpha[j,t] = b[j,t] + logsumexp(alpha[:,t-1]+a[:,j])


		return alpha

	def forward_backward(self, prior, a, b):
		# Initialization Step

		beta = np.ones((self.n,self.dataLength))*np.log(0)
		alpha = np.ones((self.n,self.dataLength))*np.log(0)
		alpha[:,0] = prior+b[:,0]
		beta[:,-1] = np.ones(self.n)*np.log(1)

		for t in range(1,self.dataLength):
			tbeta = self.dataLength-1-t
			for j in range(self.n):
				beta[:,tbeta] = np.logaddexp( beta[:,tbeta] , a[:,j]  + b[j,tbeta+1]+   beta[j,tbeta+1])
				tmp = 0
				for i in range(self.n):
					tmp =  np.logaddexp(tmp, alpha[i,t-1]+a[i,j])
				alpha[j,t]  = b[j,t] + tmp

		return alpha, beta


	def computeLikelihood(self, alpha):
		likelihood = np.log(0)
		for i in range(self.n):
			likelihood = np.logaddexp(likelihood, alpha[i,-1])
		return likelihood


	def backward(self, a, b):

		beta = np.ones((self.n,self.dataLength))*np.log(0)

		# Initialization Step
		list2reverse = range(0,self.dataLength-1)
		list2reverse.reverse(); reversedList = list2reverse


		beta[:,-1] = np.ones(self.n)*np.log(1)
		for t in reversedList: # for each timestep except initialtime
			for j in range(self.n):
				beta[:,t] = np.logaddexp( beta[:,t] , a[:,j]  + b[j,t+1]+   beta[j,t+1])



		return beta

	def Gamma(self,alpha,beta):


		normalizer = logsumexp(alpha +  beta, axis = 0)

		gamma = alpha + beta - normalizer

		npt.assert_almost_equal(np.sum(np.exp(gamma), axis = 0), 1.0) # up to default decimal 7

		return gamma

	def Zeta(self,alpha, a, beta, b):


		normalizer = np.ones(self.dataLength-1)*np.log(0)
		for i in range(self.n):
			for j in range(self.n):
				normalizer[:] = np.logaddexp(normalizer[:], \
						alpha[i,:-1]+a[i,j]+b[j,1:]+beta[j,1:])


		zeta_tensor = np.zeros((self.dataLength-1, self.n, self.n))
		for i in range(self.n):
			for j in range(self.n):
				zeta_tensor[:,i,j] = alpha[i,:-1]+a[i,j]+b[j,1:]+beta[j,1:] -normalizer[:]
		# print "zeta_inside",zeta_tensor
		return zeta_tensor

	def updateTransition(self,zeta_tensor,gamma):



		sum_zeta  = logsumexp(zeta_tensor, axis=0)
		sum_gamma = logsumexp(gamma[:,:-1], axis = 1)


		a = np.ones((self.n,self.n))*np.log(0)
		for i in range(self.n):
			a[i,:] = sum_zeta[i,:] - sum_gamma[i]

		# Check if sums to 1.
		# npt.assert_almost_equal(np.sum(np.exp(a), axis = 1), 1.0)
		return a

	def updatePrior(self,gamma):

		prior = gamma[:,0]
		npt.assert_almost_equal(np.sum(np.exp(prior)), 1.0)

		return prior

	def updateMean(self,gamma):

		gamma = np.exp(gamma)

		mu = np.zeros((self.n, self.dataDimens))
		for i in range(self.n):

			sum_nom = 0
			sum_denom = 0
			for t in range(self.dataLength):
				sum_nom  +=  gamma[i,t]*self.data[t,:]
				sum_denom +=  gamma[i,t]

			mu[i] =sum_nom/sum_denom


		return mu


	def updateCovars(self,gamma,mean):
		gamma = np.exp(gamma)
		covariances = []
		for i in range(self.n):
			sum_nom = np.zeros(self.dataDimens)
			sum_denom = 0
			mu = mean[i].reshape(1,self.dataDimens)
			for t in range(self.dataLength):
				o  = self.data[t,:].reshape(1,self.dataDimens)
				sum_nom   = sum_nom   + gamma[i,t]*np.outer(o-mu,o-mu)
				sum_denom = sum_denom + gamma[i,t]

			covariances.append(sum_nom/sum_denom)

		return np.array(covariances)

	def Baum_Welch(self,number_of_iterations = 1000,minError = 0.01):
		# initialize all
		prior     = self.initPrior()
		a         = self.initTransition()
		mu, sigma = self.initMeansCovs()
	

		b         = self.observationProb(mu, sigma)
		likelihood    = np.log(20)
		oldLikelihood = np.log(10)
		counter = 0



		while np.abs(likelihood-oldLikelihood)/np.abs(oldLikelihood) > minError:


			# Expectation Step
			start = time()
			alpha, beta = self.forward_backward(prior, a, b)
			gamma = self.Gamma(alpha,beta)
			zeta  = self.Zeta(alpha, a, beta, b)

			# Maximization Step

			prior = self.updatePrior(gamma)
			a     = self.updateTransition(zeta, gamma)
			mu    = self.updateMean(gamma)
			sigma = self.updateCovars(gamma,mu)
			sigma = sigma + np.eye(self.dataDimens)*(10**-10)
			b     = self.observationProb(mu,sigma)


			oldLikelihood = likelihood
			likelihood    = self.computeLikelihood(alpha)

			#print likelihood,counter

			counter += 1
			# if counter == 1:
			#     break
			# print "counter is :\n",counter, "\n"

		return prior,a,mu,sigma, likelihood
