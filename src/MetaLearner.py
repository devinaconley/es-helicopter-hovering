# Evolution Strategies for meta-learning

# lib
from keras.models import model_from_json

# src
from .ESTrainer import ESTrainer

class MetaLearner :
	def __init__( self, model, env ) :
		self.env = env
		self.model = model_from_json( model.to_json( ) )  # deep copy

	def Train( self, iterations=100, pop=100, lrStart=0.0003, sigmaStart=0.2,
			   popMeta=5, lrMeta=0.0003, sigmaMeta_lr=0.00001, sigmaMeta_sigma=0.01 ) :
		lr = lrStart
		sigma = sigmaStart

		for i in range( iterations ) :
			metaParams = []
			while len( metaParams ) < popMeta :
				trainer = ESTrainer( model_from_json( self.model.to_json( ) ), self.env )
				lrTemp = lr # + noise
				sigmaTemp = sigma # + noise

				# test, append to metaParams

			# set self.model as best model, update lr and sigma with reward weighting