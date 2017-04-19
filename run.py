# es-keras
# Driver script

# lib
import argparse
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import serialize_keras_object
# src
from src.ESTrainer import ESTrainer

# main
def main( ) :
	# pull command line args
	args = ParseArguments( )

	# setup environment
	env = gym.make( args['environment'] )
	obs = env.reset( )

	print( env.action_space.n )
	print( obs.shape )

	# setup original model
	model = Sequential( )
	model.add( Dense( env.action_space.n, input_shape=obs.shape ) )
	# model.add( Activation( 'relu' ) )

	# evolutionary-strategies
	es = ESTrainer( model, env )
	es.Train( population=100, episodes=1 )

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
