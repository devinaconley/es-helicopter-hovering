# es-helicopter-hovering
# Driver script

# lib
import argparse

# src
from src.ESTrainer import ESTrainer

# main
def main() :
	# pull command line args
	args = ParseArguments()



# Command Line Arguments
def ParseArguments( ) :
	# Define arguments
	parser = argparse.ArgumentParser( )
	parser.add_argument( '-m', '--model', help='Path to existing model to start training from',
						 default=None )
	parser.add_argument( '-e', '--episodes', help='Number of episodes per training iterations',
						 default=1000 )
	parser.add_argument( '-i', '--iterations', help='Max number of training iterations',
						 default=1000 )

	# Parse arguments and return
	args = vars( parser.parse_args( ) )

	return args

if __name__ == '__main__' :
	main( )