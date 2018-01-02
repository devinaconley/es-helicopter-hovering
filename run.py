# lib
import argparse
import gym
import numpy as np
import itertools
import json
import time
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import serialize_keras_object, np_utils

# src
from src.SpeciesHandler import SpeciesHandler
from src.MetaLearner import MetaLearner
from src.trainer.ESTrainer import ESTrainer
from src.trainer.KerasTrainer import KerasTrainer


# main
def main():
    # pull command line args
    args = parseArguments()

    # load config file
    config = {}
    with open( args['config'] ) as f:
        config = json.load( f )

    # setup the appropriate trainer and task
    trainerType = config['trainer'].lower()
    trainer = None

    if trainerType == 'KerasTrainer'.lower():
        # get labels
        yRaw = np.loadtxt( config['dataset'], delimiter=',', dtype=np.str,
                           usecols=[config['datasetLabelIndex']] )

        # encode labels
        enc = LabelEncoder()
        enc.fit( yRaw )
        y = enc.transform( yRaw )
        y = np_utils.to_categorical( y )

        # aggregate all features
        x = np.zeros( (yRaw.shape[0], 0) )

        # get continuous features
        if config['datasetContinuousIndices']:
            xContinuous = np.loadtxt( config['dataset'], delimiter=',',
                                      usecols=config['datasetContinuousIndices'] )
            x = np.concatenate( (x, xContinuous), axis=1 )

        # get / encode categorical features
        if config['datasetCategoricalIndices']:
            for index in config['datasetCategoricalIndices']:
                # load as raw string labels
                xCategorical = np.loadtxt( config['dataset'], delimiter=',', dtype=np.str,
                                           usecols=[index] )
                # one-hot encoding
                enc = LabelEncoder()
                enc.fit( xCategorical )
                xCategorical = enc.transform( xCategorical )  # int labels
                xCategorical = np_utils.to_categorical( xCategorical )  # one-hot

                # concat entire encoding
                x = np.concatenate( (x, xCategorical), axis=1 )

        # setup model
        model = Sequential()
        model.add( Dense( 64, input_dim=x.shape[1], activation='tanh' ) )
        model.add( Dense( y.shape[1], activation='sigmoid' ) )

        trainer = KerasTrainer( model, x, y )

        trainer.configure( batchSize=config['batchSize'] )


    elif trainerType == 'ESTrainer'.lower():
        # setup environment
        env = gym.make( config['environment'] )
        obs = env.reset()

        # setup original model
        model = Sequential()
        model.add( Dense( env.action_space.n, input_shape=obs.shape,
                          kernel_initializer='random_normal', use_bias=False ) )
        # evolutionary-strategies
        trainer = ESTrainer( model, env )
        trainer.configure( population=100, maxSteps=None, maxStepsAction=0 )

    else:
        print( 'Invalid trainer specified: {}. Exiting.'.format( args['trainer'] ) )
        return

    ts = str( int( time.time() ) )

    # do training
    if config['runESMeta']:
        # Meta-Learning
        metalearner = MetaLearner( trainer )
        with open( 'etc/results/{}_{}_metalearner.log'.format( ts, trainerType ), 'w' ) as logFile:
            metalearner.train( logFile=logFile,
                               paramsOrig=config['paramInitials'],
                               sigmas=config['paramSigmas'],
                               population=config['population'],
                               iterations=config['iterations'],
                               iterationsMeta=config['iterationsMeta'],
                               verbose=True )

    elif config['runGridMeta']:
        # Compare with grid search
        perms = itertools.product( *config['gridValues'] )
        model = trainer.getModel()
        with open( 'etc/results/{}_{}_gridsearch.log'.format( ts, trainerType ), 'w' ) as logFile:
            for p in perms:
                trainer.setModel( model )
                trainer.train( params=p,
                               iterations=config['iterations'],
                               logFile=logFile,
                               verbose=True )

    else:
        # train with no metalearning
        trainer.train( iterations=config['iterations'], params=config['paramInitials'], verbose=True )

    # structural learning
    # sh = SpeciesHandler( model, env )
    # sh.train( extinctionInterval=10, numSpecies=5 )

    return


# command line arguments
def parseArguments():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--config', help='Path to config file',
                         default=None )

    # parse arguments and validate
    args = vars( parser.parse_args() )

    if args['config'] is None:
        parser.error( 'config file must be provided.' )

    return args


if __name__ == '__main__':
    main()
