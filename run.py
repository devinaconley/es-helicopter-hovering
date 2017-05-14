# lib
import argparse
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import serialize_keras_object
# src
from src.ESTrainer import ESTrainer
from src.SpeciesHandler import SpeciesHandler
from src.MetaLearner import MetaLearner


# main
def main( ) :
	# pull command line args
	args = ParseArguments( )

	# setup environment
	env = gym.make( args['environment'] )
	obs = env.reset( )

	# setup original model
	model = Sequential( )
	model.add( Dense( env.action_space.n, input_shape=obs.shape,
					  kernel_initializer='random_normal', use_bias=False ) )

	# evolutionary-strategies
	# es = ESTrainer( model, env )
	# es.Train( iterations=200, render=True )

	# structural learning
	# sh = SpeciesHandler( model, env )
	# sh.Train( extinctionInterval=10, numSpecies=5 )

	# Meta-Learning
	metalearner = MetaLearner( model, env )
	with open( 'metalearner.log', 'w' ) as logFile :
		metalearner.Train( logFile=logFile )

	# Compare with grid search
	# es = ESTrainer( model, env )
	# lr = [ 0.0002, 0.0003, 0.0004, 0.0005 ]
	# sigma = [0.1, 0.2, 0.25, 0.3]
	# with open( 'gridsearch.log', 'w' ) as logFile :
		# for l in lr :
			# for s in sigma :
				# es.Train( iterations=100, logFile=logFile )

	return


# Command Line Arguments
def ParseArguments( ) :
	# Define arguments
	parser = argparse.ArgumentParser( )
	parser.add_argument( '-m', '--model', help='Path to existing model to start training from',
						 default=None )
	parser.add_argument( '-n', '--episodes', help='Number of episodes per training iteration',
						 default=1000 )
	parser.add_argument( '-i', '--iterations', help='Max number of training iterations',
						 default=1000 )
	parser.add_argument( '-e', '--environment', help='OpenAI environment',
						 default='LunarLander-v2' )

	# Parse arguments and return
	args = vars( parser.parse_args( ) )

	return args


if __name__ == '__main__' :
	main( )
