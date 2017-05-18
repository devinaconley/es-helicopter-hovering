# es-meta

Evolutionary Strategies with Meta-Learning and Network Learning, for training an agent to play LunarLander-v2 using keras model.

To run the code, please just use "python run.py" with appropriate parameters, could easily comment out or uncomment the related part to run ES, network learning or Meta-Learning like the following, more parameter info. please refer to the specific codes.

## run the code
python run.py 

## evolutionary-strategies
es = ESTrainer( model, env ) <br>
es.Train( iterations=200, render=True )

## structural learning
sh = SpeciesHandler( model, env )<br>
sh.Train( extinctionInterval=10, numSpecies=5 )

## Meta-Learning
metalearner = MetaLearner( model, env )<br>
with open( 'metalearner.log', 'w' ) as logFile :<br>
	metalearner.Train( logFile=logFile )
