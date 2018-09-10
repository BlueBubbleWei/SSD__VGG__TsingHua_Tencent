import argparse
parser = argparse.ArgumentParser(description='I print fibonacci sequence')
parser.add_argument('-s', '--start', type=int, dest='start',
                    help='Start of the sequence', required=True)
parser.add_argument('-e', '--end', type=int, dest='end',
                    help='End of the sequence', required=True)

def infinite_fib():
    a, b = 0, 1
    yield a
    yield b
    while True:
        #print 'Before caculation: a, b = %s, %s' % (a, b)
        a, b = b, a + b
        #print 'After caculation: a, b = %s, %s' % (a, b)
        yield b


def fib(start, end):
    for cur in infinite_fib():
        #print 'cur: %s, start: %s, end: %s' % (cur, start, end)
        if cur > end:
            return
        if cur >= start:
            #print 'Returning result %s' % cur
            yield cur

def main():
    args = parser.parse_args()
    args.start=3
    args.end=5
    for n in fib(args.start, args.end):
        print(n)

if __name__ == '__main__':
    main()