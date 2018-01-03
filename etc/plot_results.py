# helper script to generate plots

# lib
import argparse
import csv
import matplotlib

def main():
    args = parseArguments()

    if args['es_results'] :
        with open( args['es_results']) as file :
            reader = csv.reader(file, delimiter=',' )
            for line in reader :
                print( line )


# command line arguments
def parseArguments():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '-r', '--results', help='Path to log file of metalearning results',
                         default=None )

    # parse arguments and validate
    args = vars( parser.parse_args() )

    if args['es_results'] is None and args['grid_results'] is None:
        parser.error( 'Some results must be provided.' )

    return args


if __name__ == '__main__':
    main()
