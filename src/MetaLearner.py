# Evolution Strategies for meta-learning

# lib
import numpy as np
from keras.models import model_from_json


class MetaLearner:
    def __init__( self, trainer ):
        self.trainer = trainer
        self.model = trainer.getModel()

    def train( self, iterations=100, population=10, paramsOrig=[], sigmas=[],
               iterationsMeta=10, lr=0.001, logFile=None, verbose=False ):
        # sanitize
        if len( paramsOrig ) != len( sigmas ):
            print( 'Length of original parameter values and parameters noise must match.' )
            return
        if not paramsOrig:
            print( 'Original parameter values must not be empty.' )
            return

        params = paramsOrig[:]

        for i in range( int( iterations / iterationsMeta ) ):
            cands = []

            while len( cands ) < population:
                self.trainer.setModel( self.model )

                noise = []
                for s in sigmas:
                    noise.append( np.random.normal( 0.0, s ) )

                # test, append to candidates
                paramsTemp = [params[i] + noise[i] for i in range( len( params ) )]
                reward = self.trainer.train( iterations=iterationsMeta, params=paramsTemp,
                                             logFile=logFile, verbose=verbose )

                m = self.trainer.getModel()
                cands.append( {
                    'reward': reward,
                    'params': paramsTemp,
                    'noise': noise,
                    'model': m
                } )

                # log and print as appropriate
                strParams = ','.join( str( p ) for p in paramsTemp )
                if logFile:
                    logFile.write( 'meta-cand,{},{},{}\n'.format(
                        (i + 1) * iterationsMeta, reward, strParams ) )
                if verbose:
                    print( 'candidate {} -- reward: {}, params: {}'.format(
                        len( cands ), reward, strParams ) )

            # do updates
            meanReward = np.mean( [x['reward'] for x in cands] )  # to normalize
            stdReward = np.std( [x['reward'] for x in cands] )
            bestReward = cands[0]['reward']
            bestModel = cands[0]['model']
            for c in cands:
                if c['reward'] > bestReward:
                    bestReward = c['reward']
                    bestModel = c['model']
                # weighted update of all params
                for j in range( len( params ) ):
                    params[j] += ((lr / (sigmas[j] * population))
                                  * ((c['reward'] - meanReward) / stdReward)
                                  * c['noise'][j])

            self.model = bestModel

            # log and print as appropriate
            strParams = ','.join( str( p ) for p in params )
            if logFile:
                logFile.write( 'meta-main,{},{},{}\n'.format(
                    (i + 1) * iterationsMeta, bestReward, strParams ) )
            if verbose:
                print( 'generations: {}, max reward: {}, params: {}'.format(
                    (i + 1) * iterationsMeta, bestReward, strParams ) )
