# Evolutionary Strategies trainer

# lib
import numpy as np
from keras.models import model_from_json


class ESTrainer :
	def __init__( self, model, env ) :
		self.env = env
		self.model = model_from_json( model.to_json( ) )  # deep copy
		self.model.set_weights( model.get_weights( ) )
		self.weights = []
		self.noise = []
		self.rewards = []

		# get original weights
		for l in self.model.layers :
			self.weights.append( l.get_weights( )[:] )

	def Train( self, iterations=100, episodes=1, population=100, sigma=0.2, lr=0.0003, maxSteps=250,
			   render=False, verbose=False, logFile=None ) :
		totalReward = 0.0
		for i in range( iterations ) :
			self.GeneratePopulation( population, sigma )
			self.TestPopulation( episodes, sigma, maxSteps )
			self.ConsolidateModels( lr, sigma )

			if verbose :
				print( 'Iteration: {}, average reward: {}, std reward: {}'.format(
					i, np.mean( self.rewards ), np.std( self.rewards ) ) )

			if logFile :
				logFile.write( 'es,{},{}\n'.format( i, np.mean( self.rewards ) ) )

			totalReward += np.mean( self.rewards )

			if render :  # show debug episode of updated model
				obs = self.env.reset( )
				done = False

				while not done :
					self.env.render( )
					action = self.model.predict_classes( np.array( [obs] ), verbose=0 )  # highest prob action from nn
					obs, r, done, info = self.env.step( action[0] )

		return totalReward / iterations

	def TestPopulation( self, episodes, sigma, maxSteps ) :
		# test each variant over e episodes
		for p in range( self.population ) :
			# update model with noise appropriately

			for i, l in enumerate( self.model.layers ) :
				l.set_weights( [(self.weights[i][j] + sigma * self.noise[i][j][p, :])
								for j in range( len( self.weights[i] ) )] )

			for e in range( episodes ) :
				# setup/run episode
				obs = self.env.reset( )
				done = False
				self.rewards[p] = 0

				k = 0
				while not done :
					# self.env.render( )
					action = [0]
					if k < maxSteps :
						action = self.model.predict_classes( np.array( [obs] ),
															 verbose=0 )  # highest prob action from nn
					obs, reward, done, info = self.env.step( action[0] )
					self.rewards[p] += reward
					k += 1

	def GeneratePopulation( self, population, std ) :
		self.population = population

		# get base model shape and weights
		layerShapes = []
		for l in self.model.layers :
			layerShapes.append( [w.shape for w in l.get_weights( )] )

		# generate random population
		del self.noise[:]
		# set weights for each layer by adding noise to base model
		for l in layerShapes :
			layer = []
			for sh in l :
				layer.append( np.random.randn( population, *sh ) )
			self.noise.append( layer )

		self.rewards = np.zeros( population )

	def WeightedSum( self, values, weights ) :
		return np.dot( values, weights )

	def ConsolidateModels( self, lr, std ) :
		# normed = (self.rewards - np.mean( self.rewards )) / np.std( self.rewards )
		newWeights = self.weights[:]

		# weighted updates
		for i, l in enumerate( newWeights ) :
			for j, w in enumerate( l ) :
				update = np.apply_along_axis( self.WeightedSum, 0, self.noise[i][j], self.rewards )
				w += lr / (self.population * std) * update

		self.weights = newWeights
