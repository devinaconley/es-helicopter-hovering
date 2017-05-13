# Evolution Strategies for meta-learning

# lib
import numpy as np
from keras.models import model_from_json

# src
from .ESTrainer import ESTrainer

class MetaLearner :
    def __init__( self, model, env ) :
        self.env = env
        self.model = model_from_json( model.to_json( ) )  # deep copy
        self.model.set_weights( model.get_weights( ) )

    def Train( self, iterations=100, pop=100, lrStart=0.0003, sigmaStart=0.2,
               popMeta=5, lrMeta=0.0003, sigmaMeta_lr=0.00001, sigmaMeta_sigma=0.01 ) :
        lr = lrStart
        sigma = sigmaStart

        for i in range( iterations ) :
            metaParams = []

            while len( metaParams ) < popMeta :
                m = model_from_json( self.model.to_json( ) )  # deep copy
                m.set_weights( self.model.get_weights( ) )
                trainer = ESTrainer( m, self.env )
                lrTemp = lr + np.random.normal(0, sigmaMeta_lr) # + noise
                sigmaTemp = sigma + np.random.normal(0, sigmaMeta_sigma) # + noise

                # test, append to metaParams
                reward = trainer.Train(iterations=10, population=pop, sigma=sigmaTemp, lr=lrTemp, render=True)
                w = trainer.model.get_weights()
                metaParams.append([reward, lrTemp, sigmaTemp, w])

                print("metaparams is {}".format(metaParams[-1]))

            # update lr and sigma
            max = metaParams[0][0]
            w_new = metaParams[0][3]
            for meta in metaParams:
                if max < meta[0]:
                    max = meta[0]
                    lr = meta[1] # + np.random.normal(0, sigmaMeta_lr) * lrMeta
                    sigma = meta[2] # + np.random.normal(0, sigmaMeta_sigma) * lrMeta
                    w_new = meta[3]
            self.model.set_weights(w_new)

            print ("max reward is {}, lr is {}, sigma is {}".format(max, lr, sigma))