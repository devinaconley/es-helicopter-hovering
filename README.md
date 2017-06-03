# es-meta

Evolution Strategies applied to Meta-Learning and Network Learning

## Meta-learning

ES is applied to the optimization of meta-learning for a "black-box" learning process. These learning processes are wrapped in a *trainer* class. Current examples are ES (for training a neural network on a RL problem) or backpropagation in Keras (for supervised learning).


```
usage: run.py [-h] [-m MODEL] [-i ITERATIONS] [-j ITERMETA] [-n POPULATION]
              [-p PARAMS [PARAMS ...]] [-s SIGMAS [SIGMAS ...]]
              [-g GRIDVALUES [GRIDVALUES ...]] [-e ENVIRONMENT] [-d DATASET]
              [-t TRAINER] [-l LOGFILE] [--es-meta] [--grid-meta]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to existing model to start training from
  -i ITERATIONS, --iterations ITERATIONS
                        Max number of training iterations
  -j ITERMETA, --iterations-meta ITERMETA
                        Training iterations per metalearning generation
  -n POPULATION, --population POPULATION
                        Population for metalearning
  -p PARAMS [PARAMS ...], --params PARAMS [PARAMS ...]
                        Learning params. Used as original values for
                        metalearning
  -s SIGMAS [SIGMAS ...], --sigmas SIGMAS [SIGMAS ...]
                        Std of noise for each param in ES metalearning
  -g GRIDVALUES [GRIDVALUES ...], --grid-values GRIDVALUES [GRIDVALUES ...]
                        Array of values for each param, to be used in grid
                        search. (ex: -g [0.1,0.2] [0.0002,0.0003] )
  -e ENVIRONMENT, --environment ENVIRONMENT
                        OpenAI environment (only for RL)
  -d DATASET, --dataset DATASET
                        Path to dataset (only for supervised)
  -t TRAINER, --trainer TRAINER
                        RL options: ESTrainer; Supervised options:
                        KerasTrainer, ...
  -l LOGFILE, --logfile LOGFILE
                        Filename to log learning metrics
  --es-meta             Apply ES for metalearning.
  --grid-meta           Apply grid search for metalearning.

```

### Examples:

es-meta over supervised learning for iris data set:
```
python run.py -t KerasTrainer --es-meta -d etc/datasets/iris.data -p 0.0003 0.9 0.999 -s 0.00002 0.02 0.0002 -i 50
```

es-meta over ES for training on OpenAI RL task:
```
python run.py -t ESTrainer --es-meta -e LunarLander-v2 -p 0.2 0.0003 -s 0.02 0.00003
``` 

*compare with*

grid-search for meta-learning over supervised learning for iris:
```
python run.py -t KerasTrainer --grid-meta -d etc/datasets/iris.data -g [0.0002,0.0003,0.0004] [0.85,0.9,0.95] [0.9985,0.999,0.9995] -i 50
```

grid search for meta-learning over ES on OpenAI RL task:
```
python run.py -t ESTrainer --grid-meta -e LunarLander-v2 -g [0.15,0.2,0.25,0.3] [0.0002,0.0003,0.0035,0.0004]
```

## Network structure learning
Still a work in progress, but a current functional implementation is defined in *SpeciesHandler.py*