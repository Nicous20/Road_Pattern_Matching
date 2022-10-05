import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')

    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    parser.add_argument("-o", "--output", metavar="DIR", help="write weights to DIR")

    args = parser.parse_args()

    print(args)  # Namespace(output='nihao')
    print(type(args))  # <class 'argparse.Namespace'>

    print(args.accumulate(args.integers))