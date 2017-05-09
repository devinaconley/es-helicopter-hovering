# Evolution Strategies Species Handler for learning structure

# lib
from keras.models import model_from_json

# src
from .ESTrainer import ESTrainer

# species struct
class Species :
	def __init__( self, trainer ) :
		self.trainer = trainer
		self.reward = 0.0

class SpeciesHandler :
	def __init__( self, model, env ) :
		self.env = env
		self.species = []
		self.species.append( Species( ESTrainer( model, env ) ) )

	def Train( self, iterations=400, extinctionInterval=10, numSpecies=5 ) :
		for i in range( int( iterations / extinctionInterval ) ) :
			# do mutations
			while len( self.species ) < numSpecies :
				print( self.species )
				tempModel = model_from_json( self.species[0].trainer.model.to_json() )
				# mutate...
				self.species.append( Species( ESTrainer( tempModel, self.env ) ) )

			# run each training
			for j, s in enumerate( self.species ) :
				r = s.trainer.Train( iterations=extinctionInterval, render=(j == 0) )
				s.reward = r


			# eliminate lowest performer...

		print( self.species )
