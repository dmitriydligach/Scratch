#!/usr/bin/env python3

import argparse

# help is always an option
# parseargs.py -h

def main():
  """Some function"""

  parser = argparse.ArgumentParser()

  # positional arguments
  # parseargs.py 11 12
  parser.add_argument('arg1', help='required argument 1')
  parser.add_argument('arg2', help='required argument 2')

  # optional arguments
  # parseargs.py --arg3 111 --arg4 222 500 600
  # parseargs.py -a3 111 -a4 222 one two
  parser.add_argument('-a3', '--arg3', help='optional argument 3')
  parser.add_argument('-a4', '--arg4', help='optional argument 4')

  # args is a argparse.Namespace object
  args = parser.parse_args()

  print(args)
  print(args.arg1, args.arg2, args.arg3, args.arg4)

if __name__ == "__main__":

  main()

