# es-helicopter-hovering
# Driver script

# lib
import argparse
import gym

# src
from src.ESTrainer import ESTrainer

# main
def main( ) :
	# pull command line args
	args = ParseArguments( )

	# setup environment
	env = gym.make( args['environment'] )
	obs = env.reset( )
	done = False

	while not done :
		env.render()
		action = env.action_space.sample()
		obs, reward, done, info = env.step( action )

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
