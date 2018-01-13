# utility script to convert arff file to csv

# lib
import argparse
import csv
import arff
import os


def main():
    args = parseArguments()

    with open( args['arff'] ) as file:
        dataset = arff.load( file )

    csvPath = args['csv'] if args['csv'] else os.path.splitext( args['arff'] )[0] + '.csv'
    with open( csvPath, 'w' ) as file:
        writer = csv.writer( file )
        if args['header']:
            writer.writerow( [a[0] for a in dataset['attributes']] )
        writer.writerows( dataset['data'] )


# command line arguments
def parseArguments():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '-a', '--arff', help='Path to arff data file',
                         default=None )
    parser.add_argument( '-c', '--csv', help='Path to converted csv output file',
                         default=None )
    parser.add_argument( '-i', '--header', help='Include header with attribute names',
                         default=False, action='store_true' )

    # parse arguments and validate
    args = vars( parser.parse_args() )

    if args['arff'] is None:
        parser.error( 'Data file must be provided.' )

    return args


if __name__ == '__main__':
    main()
