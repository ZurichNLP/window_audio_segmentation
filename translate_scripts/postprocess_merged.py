#!/usr/bin/env python3

import sys
import re
import argparse


def get_arg_parser() -> argparse.ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = argparse.ArgumentParser(description='Command for creating time stamp annotation for the merged-window SLTev evaluation format.')

    parser.add_argument('-i', '--input',
                        type=argparse.FileType('r'),
                        required=True,
                        help='File with merged window log.')
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w'),
                        required=True,
                        help='Output file for SLTev evaluation format.')
    parser.add_argument('-s', '--shift',
                        type=int,
                        default=2,
                        help='Shift for windows.')
    return parser


def main() -> None:
    '''
    Create the time stamps for the merged window output in a file following the SLTev format.
    '''
    cfg = get_arg_parser().parse_args()

    talks = cfg.input.read().split('<EOT>\n')[:-1]

    for talk in talks:
        lines = talk.split('\n')[:-1]
        window_size = float(lines[0].split()[-1])
        progress = 0.0
        for line in lines[1:-1]:
            _, *line = line.split()
            line = ' '.join(line)
            if line.startswith('Output Stream:'):

                line = re.sub('Output Stream:', '', line).lstrip()
                cfg.output.write(f'P {window_size + progress} 1.0 {window_size + progress} {line}\n')
                progress += cfg.shift

        _, *line =  lines[-1].split()
        line = ' '.join(line)
        line = re.sub('Output Stream:', '', line).lstrip()
        cfg.output.write(f'C {window_size + progress} 1.0 {window_size + progress} {line}\n')


if __name__ == '__main__':
        main()
