from .Individual import Individual
from .Constants import *
from .GeneticOperators import getElite, getOffspring
import multiprocessing as mp
import time
import datetime

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019 J. E. Batista
#

class Population:
	population = None
	bestIndividual = None
	currentGeneration = 0
	fitnessType = None



	def __init__(self, fitnessType):

		self.population = []
		self.fitnessType = fitnessType
		while len(self.population) < POPULATION_SIZE:
			self.population.append(Individual())
		self.bestIndividual = self.population[0]



	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= MAX_GENERATION
		if self.bestIndividual.getFitnessType() == "Reward Entropy" or self.bestIndividual.getFitnessType() == "Max Reversed Total Entropy":
			perfectTraining = self.bestIndividual.getFitness() == 2
		else:
			perfectTraining = self.bestIndividual.getFitness() == 1
		
		return genLimit  or perfectTraining



	def train(self):
		'''
		Training loop for the algorithm.
		'''
		if VERBOSE:
			print("> Running log:")

		while not self.stoppingCriteria():
			self.nextGeneration()
			self.currentGeneration += 1


		if VERBOSE:
			print()



	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = datetime.datetime.now()
		
		begin = str(begin.hour)+"h"+str(begin.minute)+"m"+str(begin.second)

		# Calculates the accuracy of the population using multiprocessing
		if THREADS > 1:
			with mp.Pool(processes= THREADS) as pool:
				fitArray = pool.map(getTrainingPredictions, [ind for ind in self.population] )
				for i in range(len(self.population)):
					self.population[i].trainingPredictions = fitArray[i][0]
					self.population[i].model = fitArray[i][1]
	        


		# Sort the population from best to worse
		self.population.sort(reverse=True)

		# Update best individual
		if self.population[0] > self.bestIndividual:
			self.bestIndividual = self.population[0]
			self.bestIndividual.prun(simp=False)

		# Generating Next Generation
		newPopulation = []
		newPopulation.extend(getElite(self.population))
		while len(newPopulation) < POPULATION_SIZE:
			newPopulation.extend(getOffspring(self.population, self.fitnessType))
		self.population = newPopulation[:POPULATION_SIZE]

		end = datetime.datetime.now()
		end = str(end.hour)+"h"+str(end.minute)+"m"+str(end.second)

		# Debug
		if VERBOSE and self.currentGeneration%5==0:
			if self.fitnessType == "Accuracy":
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Acc: "+ "%.6f" %self.bestIndividual.getTrainingAccuracy())
			elif self.fitnessType == "Balanced Accuracy":
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Bala: "+ "%.6f" %self.bestIndividual.getTrainingBalancedAccuracy())
			elif self.fitnessType == "Reward Entropy":
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Rew-Ent: "+ "%.6f" %self.bestIndividual.getTrainingRewardEntropy())
			elif self.fitnessType == "Penalty Entropy":
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Pen-Ent: "+ "%.6f" %self.bestIndividual.getTrainingPenaltyEntropy())
			elif self.fitnessType == "Entropy":
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Entropy: "+ "%.6f" %self.bestIndividual.getTrainingEntropy())
			elif self.fitnessType == "Entropy With Diagonal":
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Entropy-With-Diagonal: "+ "%.6f" %self.bestIndividual.getTrainingEntropyWithDiagonal())



	def predict(self, sample):
		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)

	def getBestIndividual(self):
		return self.bestIndividual

	def getCurrentGeneration(self):
		return self.currentGeneration

	def getTrainingAccuracyOverTime(self):
		return self.trainingAccuracyOverTime

	def getTestAccuracyOverTime(self):
		return self.testAccuracyOverTime

	def getTrainingBalancedAccuracyOverTime(self):
		return self.trainingBalancedAccuracyOverTime

	def getTestBalancedAccuracyOverTime(self):
		return self.testBalancedAccuracyOverTime

	def getTrainingRewardEntropyOverTime(self):
		return self.trainingRewardEntropyOverTime

	def getTrainingPenaltyEntropyOverTime(self):
		return self.trainingPenaltyEntropyOverTime

	def getTrainingEntropyOverTime(self):
		return self.trainingEntropyOverTime

	def getTrainingEntropyWithDiagonalOverTime(self):
		return self.trainingEntropyWithDiagonalOverTime

	def getTrainingWaFOverTime(self):
		return self.trainingWaFOverTime

	def getTestWaFOverTime(self):
		return self.testWaFOverTime

	def getTrainingKappaOverTime(self):
		return self.trainingKappaOverTime

	def getTestKappaOverTime(self):
		return self.testKappaOverTime

	def getSizeOverTime(self):
		return self.sizeOverTime

	def getNumberOfDimensionsOverTime(self):
		return self.dimensionsOverTime

	def getGenerationTimes(self):
		return self.generationTimes


def calculateIndividualAccuracy_MultiProcessing(ind, fitArray, indIndex):
	fitArray[indIndex] = ind.getTrainingAccuracy()

def getTrainingPredictions(ind):
	return [ind.getTrainingPredictions(), ind.model]
