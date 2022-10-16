#!/usr/bin/env python3

import sys
import argparse


def get_arg_parser() -> argparse.ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = argparse.ArgumentParser(description='Command for merging time stamp annotation and SLT output to SLTev evaluation format.')

    parser.add_argument('-a', '--annotations',
                        type=argparse.FileType('r'),
                        required=True,
                        help='File with annotated time stamps.')
    parser.add_argument('-s', '--slt',
                        type=argparse.FileType('r'),
                        required=True,
                        help='File with SLT output.')
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w'),
                        required=True,
                        help='Output file for SLTev evaluation format.')
    return parser


def main() -> None:
    '''
    Combine time stamps and SLT output in a file following the SLTev format.
    '''
    cfg = get_arg_parser().parse_args()

    slt_outputs = cfg.slt.read().split('<EOS>\n')

    buffer = []
    slt_index = 0

    for line in cfg.annotations:
        line = line.strip()
        buffer.append(line)

        if line.startswith('C'):

            slt_lines = slt_outputs[slt_index].rstrip().split('\n')
            diff = len(slt_lines) - len(buffer)
            if diff < 0:
                print(slt_lines)
                print(buffer)
            if not len(buffer) == len(slt_lines):
                slt_lines = slt_lines[:2*diff:2] + slt_lines[2*diff:]

            for a, s in zip(buffer, slt_lines):
                cfg.output.write(f'{a} {s}\n')

            buffer = []
            slt_index += 1


if __name__ == '__main__':
        main()
