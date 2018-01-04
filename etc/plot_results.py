# helper script to generate plots

# lib
import argparse
import csv
import matplotlib.pyplot as plt


def main():
    args = parseArguments()

    with open( args['results'] ) as file:
        reader = csv.DictReader( file, delimiter=',' )
        groups = {}
        for line in reader:
            g = line['group']
            if g in groups:
                groups[g]['x'].append( float( line['epoch'] ) )
                groups[g]['y'].append( float( line['accuracy'] ) )
            else:
                groups[g] = {
                    'x': [float( line['epoch'] )],
                    'y': [float( line['accuracy'] )]
                }

        for id, group in groups.items():
            plt.plot( group['x'], group['y'] )

        plt.show()


# command line arguments
def parseArguments():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '-r', '--results', help='Path to log file of metalearning results',
                         default=None )

    # parse arguments and validate
    args = vars( parser.parse_args() )

    if args['results'] is None:
        parser.error( 'Results file must be provided.' )

    return args


if __name__ == '__main__':
    main()
