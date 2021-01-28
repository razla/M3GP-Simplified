from .Node import Node
from .Constants import *
from .Util import *
from .MahalanobisDistanceClassifier import MahalanobisDistanceClassifier
from .EuclideanDistanceClassifier import EuclideanDistanceClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from scipy.stats import entropy

from statistics import median

from pandas.core.common import flatten

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019 J. E. Batista
#

class Individual:
	trainingPredictions = None
	testPredictions = None

	size = None
	depth = None

	dimensions = None


	fitness = None

	model_name = ["MahalanobisDistanceClassifier", "EuclideanDistanceClassifier"][1]
	model = None


	fitnessType = None


	def __init__(self, dim = None, fitnessType="Accuracy"):
		self.fitnessType = fitnessType
		if dim == None:
			self.dimensions = [Node(full=True)]
		else:
			self.dimensions = dim

	def __gt__(self, other):
		sf = self.getFitness()
		sd = len(self.getDimensions())
		ss = self.getSize()

		of = other.getFitness()
		od = len(other.getDimensions())
		os = other.getSize()

		return (sf > of) or \
				(sf == of and sd < od) or \
				(sf == of and sd == od and ss < os)

	def __ge__(self, other):
		return self.getFitness() >= other.getFitness()

	def __str__(self):
		return ",".join([str(d) for d in self.dimensions])


	def trainModel(self):
		'''
		Trains the classifier which will be used in the fitness function
		'''
		if self.model == None:
			if self.model_name == "MahalanobisDistanceClassifier":
				self.model = MahalanobisDistanceClassifier()
			if self.model_name == "EuclideanDistanceClassifier":
				self.model = EuclideanDistanceClassifier()
			

			ds = getTrainingSet()
			X = [s[:-1] for s in ds]
			hyper_X = self.convert(X)
			Y = [s[-1] for s in ds]
			self.model.fit(hyper_X,Y)




	def getSize(self):
		'''
		Returns the total number of nodes within an individual.
		'''
		if self.size == None:
			self.size = sum(n.getSize() for n in self.dimensions)
		return self.size

	def getDepth(self):
		'''
		Returns the depth of individual.
		'''
		if self.depth == None:
			self.depth = max([dimension.getDepth() for dimension in self.dimensions])
		return self.depth 

	def getDimensions(self):
		'''
		Returns a deep clone of the individual's list of dimensions.
		'''
		ret = []
		for dim in self.dimensions:
			ret.append(dim.clone())
		return ret


	def getNumberOfDimensions(self):
		'''
		Returns the total number of dimensions within the individual.
		'''
		return len(self.dimensions)

	def getFitnessType(self):
		return self.fitnessType

	def getFitness(self):
		'''
		Returns the individual's fitness.
		'''
		if self.fitness == None:
			if self.fitnessType == "Accuracy":
				acc = self.getTrainingAccuracy()
				self.fitness = acc 

			if self.fitnessType == "WAF":
				waf = self.getTrainingWaF()
				self.fitness = waf
			
			if self.fitnessType == "Balanced Accuracy":
				bala = self.getTrainingBalancedAccuracy()
				self.fitness = bala
			
			if self.fitnessType == "Reward Entropy":
				rew_ent = self.getTrainingRewardEntropy()
				self.fitness = rew_ent

			if self.fitnessType == "Penalty Entropy":
				pen_ent = self.getTrainingPenaltyEntropy()
				self.fitness = pen_ent

			if self.fitnessType == "Entropy":
				min_tot_ent = self.getTrainingEntropy()
				self.fitness = min_tot_ent

			if self.fitnessType == "Entropy With Diagonal":
				max_tot_ent = self.getTrainingEntropyWithDiagonal()
				self.fitness = max_tot_ent

		return self.fitness



	def getTrainingPredictions(self):
		'''
		Returns the individual's training predictions.
		'''
		self.trainModel()
		if self.trainingPredictions == None:
			X = [sample[:-1] for sample in getTrainingSet() ]
			self.trainingPredictions = self.predict(X)
	
		return self.trainingPredictions

	def getTestPredictions(self):

		'''
		Returns the individual's test predictions.
		'''
		self.trainModel()
		if self.testPredictions == None:
			X = [sample[:-1] for sample in getTestSet() ]
			self.testPredictions = self.predict(X)

		return self.testPredictions


	def getTrainingAccuracy(self):
		'''
		Returns the individual's training accuracy.
		'''
		self.getTrainingPredictions()

		ds = getTrainingSet()
		y = [ str(s[-1]) for s in ds]
		return accuracy_score(self.trainingPredictions, y)
	
	def getTestAccuracy(self):
		'''
		Returns the individual's test accuracy.
		'''
		self.getTestPredictions()

		ds = getTestSet()
		y = [ str(s[-1]) for s in ds]
		return accuracy_score(self.testPredictions, y)

	def getTrainingBalancedAccuracy(self):
		'''
		Returns the individual's training balanced accuracy.
		'''	
		self.getTrainingPredictions()

		ds = getTrainingSet()
		y = [ str(s[-1]) for s in ds]
		return balanced_accuracy_score(self.trainingPredictions, y)

	def getTestBalancedAccuracy(self):
		'''
		Returns the individual's test balanced accuracy.
		'''
		self.getTestPredictions()

		ds = getTestSet()
		y = [ str(s[-1]) for s in ds]
		return balanced_accuracy_score(self.testPredictions, y)

	def getTrainingPenaltyEntropy(self):
		'''
		Returns the individual's training cm penalty entropy.
		'''
		self.getTrainingPredictions()

		ds = getTrainingSet()
		y = [ str(s[-1]) for s in ds]
		balanced_acc = balanced_accuracy_score(y, self.trainingPredictions)
		current_cm = confusion_matrix(y, self.trainingPredictions)
		entropy_list = []
		if balanced_acc != 1:
			for i in range(len(current_cm[0])):
				if sum(current_cm[i]) - current_cm[i][i] != 0:
					normalized_entropy = entropy(np.concatenate((current_cm[i][:i], current_cm[i][i+1:]), axis = 0)) / entropy([1]*(len(current_cm[0])-1)) 
					entropy_list.append(normalized_entropy)
		if entropy_list != []:
			entropy_median = median(entropy_list)
		else:
			entropy_median = 0
		return balanced_acc - entropy_median

	def getTrainingRewardEntropy(self):
		'''
		Returns the individual's training cm reward entropy.
		'''
		self.getTrainingPredictions()

		ds = getTrainingSet()
		y = [ str(s[-1]) for s in ds]
		balanced_acc = balanced_accuracy_score(y, self.trainingPredictions)
		current_cm = confusion_matrix(y, self.trainingPredictions)
		entropy_list = []
		if balanced_acc != 1:
			for i in range(len(current_cm[0])):
				if sum(current_cm[i]) - current_cm[i][i] != 0:
					normalized_entropy = entropy(np.concatenate((current_cm[i][:i], current_cm[i][i+1:]), axis = 0)) / entropy([1]*(len(current_cm[0])-1)) 
					entropy_list.append(normalized_entropy)
		if entropy_list != []:
			entropy_median = median(entropy_list)
		else:
			entropy_median = 0
		return balanced_acc + entropy_median

	def getTrainingEntropy(self):
		'''
		Returns the individual's cm's entropy.
		'''
		self.getTrainingPredictions()

		ds = getTrainingSet()
		y = [ str(s[-1]) for s in ds]
		cm = confusion_matrix(y, self.trainingPredictions)
		normalization = entropy([1] * len(list(flatten(cm))))
		cm_entropy = 1 / (1 + entropy(list(flatten(cm)))/normalization)
		return cm_entropy
	
	def getTrainingEntropyWithDiagonal(self):
		'''
		Returns the individual's 1/(1+cm's entropy) + normalized sum of diagonal
		'''
		self.getTrainingPredictions()

		ds = getTrainingSet()
		y = [ str(s[-1]) for s in ds]
		cm = confusion_matrix(y, self.trainingPredictions)
		normalization = entropy([1] * len(list(flatten(cm))))
		diagonal_normalization = sum(list(flatten(cm)))
		diagonal_sum = sum(cm.diagonal()) / diagonal_normalization
		cm_entropy = 1 / (1 + entropy(list(flatten(cm)))/normalization) + diagonal_sum
		return cm_entropy


	def calculate(self, sample):
		'''
		Return the position of a sample in the output space.
		'''
		return [self.dimensions[i].calculate(sample) for i in range(len(self.dimensions))]

	def convert(self, X):
		'''
		Returns the converted input space.
		'''
		return [self.calculate(sample) for sample in X]


	def predict(self, X):
		'''
		Returns the class prediction of a sample.
		'''
		if self.model == None:
			self.trainModel()
			
		hyper_X = self.convert(X)
		predictions = self.model.predict(hyper_X)

		return predictions



	def prun(self,simp=True):
		'''
		Remove the dimensions that degrade the fitness.
		If simp==True, also simplifies each dimension.
		'''
		dup = self.dimensions[:]
		i = 0
		ind = Individual(dup)

		while i < len(dup) and len(dup) > 1:
			dup2 = dup[:]
			dup2.pop(i)
			ind2 = Individual(dup2)

			if ind2 >= ind:
				ind = ind2
				dup = dup2
				i-=1
			i+=1
	
		self.dimensions = dup
		self.trainingAccuracy = None
		self.testAccuracy = None
		self.size = None
		self.depth = None
		self.model = None

		if simp:
			# Simplify dimensions
			for d in self.dimensions:
				done = False
				while not done:
					state = str(d)
					d.prun()
					done = state == str(d)

