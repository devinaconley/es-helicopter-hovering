# Evolutionary Strategies trainer
#
# this trainer expects a keras model, but instead uses ES techniques for training

# lib
import numpy as np
from keras.models import model_from_json


class ESTrainer:
    def __init__( self, model, env ):
        self.env = env
        self.noise = []
        self.rewards = []
        self.setModel( model )

        # default config
        self.population = 100
        self.maxSteps = None
        self.maxStepsAction = 0
        self.episodes = 1
        self.render = False

    def getModel( self ):
        # return deep copy
        m = model_from_json( self.model.to_json() )
        m.set_weights( self.model.get_weights() )
        return m

    def setModel( self, model ):
        # deep copy of model
        self.model = model_from_json( model.to_json() )
        self.model.set_weights( model.get_weights() )
        # get weights
        self.weights = []
        for l in self.model.layers:
            self.weights.append( l.get_weights()[:] )

    # Configure ES specific params
    def configure( self, population=None, maxSteps=None, maxStepsAction=None, episodes=None, render=None ):
        if population:
            self.population = population
        if maxSteps:
            self.maxSteps = maxSteps
        if maxStepsAction:
            self.maxStepsAction = maxStepsAction
        if episodes:
            self.episodes = episodes
        if render:
            self.render = render

    # Generic training interface
    def train( self, iterations=100, params=[0.2, 0.0003], verbose=False, logFile=None ):
        # parse training parameters
        if len( params ) != 2:
            print( 'Invalid number of parameters' )
            return
        sigma = params[0]
        lr = params[1]

        # do training
        totalReward = 0.0
        for i in range( iterations ):
            self.generatePopulation( sigma )
            self.testPopulation( sigma )
            self.consolidateModels( lr, sigma )

            if verbose:
                print( 'Iteration: {}, average reward: {}, std reward: {}'.format(
                    i, np.mean( self.rewards ), np.std( self.rewards ) ) )

            if logFile:
                logFile.write( 'es,{},{}\n'.format( i, np.mean( self.rewards ) ) )

            totalReward += np.mean( self.rewards )

            # show debug episode of updated model
            if self.render:
                obs = self.env.reset()
                done = False

                while not done:
                    self.env.render()
                    action = self.model.predict_classes( np.array( [obs] ), verbose=0 )  # highest prob action from nn
                    obs, r, done, info = self.env.step( action[0] )

        return totalReward / iterations

    def testPopulation( self, sigma ):
        # test each variant over e episodes
        for p in range( self.population ):
            # update model with noise appropriately

            for i, l in enumerate( self.model.layers ):
                l.set_weights( [(self.weights[i][j] + sigma * self.noise[i][j][p, :])
                                for j in range( len( self.weights[i] ) )] )

            self.rewards[p] = 0
            for e in range( self.episodes ):
                # setup/run episode
                obs = self.env.reset()
                done = False

                k = 0
                while not done:
                    action = [self.maxStepsAction]
                    if not self.maxSteps or k < self.maxSteps:
                        # highest prob action from nn
                        action = self.model.predict_classes( np.array( [obs] ), verbose=0 )
                    obs, reward, done, info = self.env.step( action[0] )
                    self.rewards[p] += reward
                    k += 1

    def generatePopulation( self, std ):

        # get base model shape and weights
        layerShapes = []
        for l in self.model.layers:
            layerShapes.append( [w.shape for w in l.get_weights()] )

        # generate random population
        del self.noise[:]
        # set weights for each layer by adding noise to base model
        for l in layerShapes:
            layer = []
            for sh in l:
                layer.append( np.random.randn( self.population, *sh ) )
            self.noise.append( layer )

        self.rewards = np.zeros( self.population )

    def weightedSum( self, values, weights ):
        return np.dot( values, weights )

    def consolidateModels( self, lr, std ):
        # normed = (self.rewards - np.mean( self.rewards )) / np.std( self.rewards )
        newWeights = self.weights[:]

        # weighted updates
        for i, l in enumerate( newWeights ):
            for j, w in enumerate( l ):
                update = np.apply_along_axis( self.weightedSum, 0, self.noise[i][j], self.rewards )
                w += lr / (self.population * std) * update

        self.weights = newWeights
