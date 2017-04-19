# Evolutionary Strategies trainer

# lib
from keras.models import model_from_json
import numpy as np

# TODO --- find some bug with copying arrays??? SHIT SHOULDN"T BE CGHANGING .....

class ESTrainer :
	def __init__( self, model, env ) :
		self.env = env
		self.model = model
		self.weights = []
		self.noise = []
		self.rewards = []

	def Train( self, iterations=100, episodes=10, population=100, std=0.1, lr=0.00025, render=False ) :
		for i in range( iterations ) :
			print( i )
			self.GeneratePopulation( population, std )
			self.TestPopulation( episodes )
			self.ConsolidateModels( lr, std )

	def TestPopulation( self, episodes ) :
		# test each variant over e episodes
		for p in range( self.population ) :
			# update model with noise appropriately

			for i, l in enumerate( self.model.layers ) :
				l.set_weights( [self.weights[i][j] + self.noise[i][j][p, :]
								for j in range( len( self.weights[i] ) )] )

			for e in range( episodes ) :
				# setup/run episode
				obs = self.env.reset( )
				done = False

				while not done :
					# self.env.render( )
					action = self.model.predict_classes( np.array( [obs] ), verbose=0 )  # highest prob action from nn
					obs, reward, done, info = self.env.step( action[0] )
					self.rewards[p] += reward

	def GeneratePopulation( self, population, std ) :
		self.population = population

		# get base model shape and weights
		del self.weights[:]
		layerShapes = []
		for i, l in enumerate( self.model.layers ) :
			layerShapes.append( [w.shape for w in l.get_weights( )] )
			self.weights.append( l.get_weights( ) )

		print ( layerShapes )
		# generate random population
		del self.noise[:]
		# set weights for each layer by adding noise to base model
		for l in layerShapes :
			layer = []
			for sh in l :
				layer.append( np.random.normal( 0.0, std, (population, *sh) ) )
			self.noise.append( layer )

		self.rewards = np.zeros( population )

	def WeightedSum( self, values, weights ):
		return np.dot( values, weights )

	def ConsolidateModels( self, lr, std ) :
		weights = (self.rewards - np.mean( self.rewards )) / np.std( self.rewards )

		# weighted updates
		for i, l in enumerate( self.weights ) :
			for j, w in enumerate( l ) :
				update = np.apply_along_axis( self.WeightedSum, 0, self.noise[i][j], weights )
				print( 'update shape: {}'.format( update.shape ) )
				print( 'weight.shape: {}'. format ( w.shape ) )
				w += lr / (len( self.noise ) * std) * update

		for i, l in enumerate( self.model.layers ) :
			l.set_weights( self.weights[i] )

		print( self.weights )

		print( 'Average reward: {}, std reward: {}'.format(
			np.mean( self.rewards ), np.std( self.rewards ) ) )

