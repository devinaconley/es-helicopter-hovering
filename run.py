# lib
import argparse
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import serialize_keras_object, np_utils
import itertools
from sklearn.preprocessing import LabelEncoder

# src
from src.SpeciesHandler import SpeciesHandler
from src.MetaLearner import MetaLearner
from src.ESTrainer import ESTrainer
from src.trainer.KerasTrainer import KerasTrainer

# main
def main() :
	# pull command line args
	args = ParseArguments()

	# setup the appropriate trainer and task
	trainer = None

	if args['trainer'].lower() == 'KerasTrainer'.lower() :
		# load dataset
		x = np.loadtxt( args['dataset'], delimiter=',', usecols=[0, 1, 2, 3] )
		yRaw = np.loadtxt( args['dataset'], delimiter=',', usecols=[4], dtype=np.str )
		# encode labels
		enc = LabelEncoder()
		enc.fit( yRaw )
		y = enc.transform( yRaw )
		y = np_utils.to_categorical( y )

		# setup model
		model = Sequential()
		model.add( Dense( 8, input_dim=x.shape[1], activation='tanh' ) )
		model.add( Dense( y.shape[1], activation='sigmoid' ) )

		trainer = KerasTrainer( model, x, y )


	elif args['trainer'].lower() == 'ESTrainer'.lower() :
		# setup environment
		env = gym.make( args['environment'] )
		obs = env.reset()

		# setup original model
		model = Sequential()
		model.add( Dense( env.action_space.n, input_shape=obs.shape,
						  kernel_initializer='random_normal', use_bias=False ) )
		# evolutionary-strategies
		trainer = ESTrainer( model, env )
		trainer.Configure( population=100, maxSteps=None, maxStepsAction=0 )

	else :
		print( 'Invalid trainer specified: {}. Exiting.'.format( args['trainer'] ) )
		return

	# do training
	if args['esMeta'] :
		# Meta-Learning
		metalearner = MetaLearner( trainer )
		with open( '{}_metalearner.log'.format( args['trainer'] ), 'w' ) as logFile :
			metalearner.Train( logFile=logFile, paramsOrig=args['params'], sigmas=args['sigmas'],
							   population=args['population'], iterations=args['iterations'],
							   iterationsMeta=args['iterMeta'],verbose=True )

	elif args['gridMeta'] :
		# Compare with grid search
		perms = itertools.product( *args['gridValues'] )
		model = trainer.GetModel()
		with open( '{}_gridsearch.log'.format( args['trainer'] ), 'w' ) as logFile :
			for p in perms :
				trainer.SetModel( model )
				trainer.Train( params=p, iterations=args['iterations'], logFile=logFile, verbose=True )

	else :
		# train with no metalearning
		trainer.Train( iterations=args['iterations'], params=args['params'], verbose=True )

	# structural learning
	# sh = SpeciesHandler( model, env )
	# sh.Train( extinctionInterval=10, numSpecies=5 )

	return

# Command Line Arguments
def ParseArguments() :
	# Define arguments
	parser = argparse.ArgumentParser()
	parser.add_argument( '-m', '--model', help='Path to existing model to start training from',
						 default=None )
	parser.add_argument( '-i', '--iterations', help='Max number of training iterations',
						 type=int, default=100 )
	parser.add_argument( '-j', '--iterations-meta', help='Training iterations per metalearning generation',
						 type=int, dest='iterMeta', default=10 )
	parser.add_argument( '-n', '--population', help='Population for metalearning',
						 type=int, default=10 )
	parser.add_argument( '-p', '--params', help='Learning params. Used as original values for metalearning',
						 default=[], type=float, nargs='+' )
	parser.add_argument( '-s', '--sigmas', help='Std of noise for each param in ES metalearning',
						 default=[], type=float, nargs='+' )
	parser.add_argument( '-g', '--grid-values', help='Array of values for each param, to be used in grid search.'
													 ' (ex: -g [0.1,0.2] [0.0002,0.0003] )',
						 dest='gridValues', default=[], nargs='+' )
	parser.add_argument( '-e', '--environment', help='OpenAI environment (only for RL)',
						 default='LunarLander-v2' )
	parser.add_argument( '-d', '--dataset', help='Path to dataset (only for supervised)',
						 default='etc/datasets/iris.data' )
	parser.add_argument( '-t', '--trainer', help='RL options: ESTrainer; Supervised options: KerasTrainer, ...',
						 default='ESTrainer' )
	parser.add_argument( '-l', '--logfile', help='Filename to log learning metrics' )
	parser.add_argument( '--es-meta', help='Apply ES for metalearning.',
						 dest='esMeta', action='store_true', default=False )
	parser.add_argument( '--grid-meta', help='Apply grid search for metalearning.',
						 dest='gridMeta', action='store_true', default=False )

	# Parse arguments
	args = vars( parser.parse_args() )

	# verify proper args have been provided
	if args['gridMeta'] and not args['gridValues'] :
		parser.error( 'Grid search requires the grid-values param to be provided.' )

	if args['esMeta'] and (not args['sigmas'] or not args['params']) :
		parser.error( 'ES metalearning requires both the sigmas and params arguments to be provided.' )

	if not (args['esMeta'] or args['gridMeta']) and not args['params'] :
		parser.error( 'Must specify params argument for standard training.' )

	# convert grid search values to float
	args['gridValues'] = [eval( x ) for x in args['gridValues']]

	return args

if __name__ == '__main__' :
	main()
