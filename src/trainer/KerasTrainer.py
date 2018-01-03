# Standard Keras trainer
#
# this trainer wraps a keras model, and uses the standard fit function for learning

# lib
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.optimizers import SGD, Adam


class KerasTrainer:
    def __init__( self, model, x, y ):
        self.x = x
        self.y = y
        self.setModel( model )

        # default config
        self.batchSize = 1
        self.validationSplit = 0.0  # if nonzero, will do ES based on validation accuracy

    def getModel( self ):
        # return deep copy
        m = model_from_json( self.model.to_json() )
        m.set_weights( self.model.get_weights() )
        return m

    def setModel( self, model ):
        # deep copy of model
        self.model = model_from_json( model.to_json() )
        self.model.set_weights( model.get_weights() )

    def configure( self, validationSplit=None, batchSize=None ):
        if validationSplit:
            self.validationSplit = validationSplit
        if batchSize:
            self.batchSize = batchSize

    def train( self, iterations=100, params=[0.001], verbose=False ):
        # parse training parameters
        if len( params ) != 3:
            print( 'Invalid number of parameters' )
            return
        lr = params[0]
        b1 = params[1]
        b2 = params[2]

        # setup training
        adam = Adam( lr=lr, beta_1=b1, beta_2=b2, decay=0.0 )
        self.model.compile( loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'] )

        # do training
        res = self.model.fit( self.x, self.y, epochs=iterations, batch_size=self.batchSize,
                              validation_split=self.validationSplit, verbose=verbose )

        return res.history['acc']
