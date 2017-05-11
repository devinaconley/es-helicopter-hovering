# Evolution Strategies Species Handler for learning structure

# lib
from keras.models import model_from_json, Sequential
from keras import layers
from random import randint, choice

# src
from .ESTrainer import ESTrainer

# species struct
class Species :
	def __init__( self, model ) :
		self.model = model
		self.reward = 0.0

	def __str__( self ) :
		return str( self.model ) + ' - reward: ' + str( self.reward )

	def __repr__( self ) :
		return self.__str__( )

class SpeciesHandler :
	def __init__( self, model, env ) :
		self.env = env
		self.species = []
		m = model_from_json( model.to_json( ) )  # deep copy
		m.set_weights( model.get_weights( ) )
		self.species.append( Species( m ) )

	def Train( self, iterations=400, extinctionInterval=10, numSpecies=5 ) :
		for i in range( int( iterations / extinctionInterval ) ) :
			# do mutations
			while len( self.species ) < numSpecies :
				tempModel = self.Mutate( self.species[0].model )
				# mutate...
				self.species.append( Species( tempModel ) )

			# run each training
			for j, s in enumerate( self.species ) :
				s.model.summary( )
				trainer = ESTrainer( s.model, self.env )
				r = trainer.Train( iterations=extinctionInterval, render=(j == 0) )
				s.reward = r
				s.model = trainer.model

			# eliminate lowest performer
			self.species.sort( key=lambda s : s.reward, reverse=True )
			print( self.species )
			self.species.pop( )
			print( self.species )
			print( 'Top current species, reward: {}'.format( self.species[0].reward ) )
			self.species[0].model.summary( )

	def Mutate( self, origModel ) :
		n = randint( 0, len( origModel.layers ) - 1 )  # do mutate at nth layer
		insert = choice( [True, False] )  # insert/remove

		if (n == len( origModel.layers ) - 1) and (not insert) :  # consider last layer immutable
			return self.Mutate( origModel )  # try again

		# get general current dimensions of model
		minNodes = -1  # TODO: extend this to any generic layer shape (for conv, etc.)
		maxNodes = -1
		for l in origModel.layers :
			sz = l.output_shape[1]
			if sz < minNodes or minNodes == -1 :
				minNodes = sz
			if sz > maxNodes or maxNodes == 1 :
				maxNodes = sz

		model = Sequential( )
		for i, l in enumerate( origModel.layers ) :
			lCopy = layers.deserialize( {'class_name' : l.__class__.__name__, 'config' : l.get_config( )} )  # deep copy

			if i == n and insert :  # insert rand layer
				model.add( self.RandLayer( minNodes=minNodes, maxNodes=maxNodes,
										   inputShape=(l.input_shape[1 :] if i == 0 else None) ) )

			if i == n and not insert :  # remove
				continue

			model.add( lCopy )

		return model

	def RandLayer( self, minNodes=4, maxNodes=10, inputShape=None ) :
		# setup
		minNodes = minNodes if minNodes > 1 else 2
		sz = randint( minNodes - 1, maxNodes + 1 )
		i = randint( 0, 1 )  # case switch for type of layer to return

		if i == 0 :  # dense
			return layers.Dense( sz, input_shape=inputShape ) if inputShape else layers.Dense( sz )
		elif i == 1 :  # activation
			fx = choice( ['elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid'] )
			return layers.Activation( fx, input_shape=inputShape ) if inputShape else layers.Activation( fx )
