import argparse

import numpy as np
import yaml

np.random.seed(2022)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-y', '--yaml_file',
                    type=str, required=True,
                    help='path to yaml file with segmentations')
    ap.add_argument('-s', '--src_file',
                    type=str, required=True,
                    help='path to src file')
    ap.add_argument('-t', '--trg_file',
                    type=str, required=True,
                    help='path to trg file')
    return ap.parse_args()


def main(args):

    with open(args.yaml_file) as yml_file, \
         open(args.src_file) as src_file, \
         open(args.trg_file) as trg_file, \
         open(args.src_file+'.prefix', 'w') as src_out, \
         open(args.trg_file+'.prefix', 'w') as trg_out:

        yml = yaml.safe_load(yml_file)

        modified = []
        for seg, src, trg in zip(yml, src_file, trg_file):
            src = src.strip().split()
            trg = trg.strip().split()

            # compute audio-to-token ratios
            len_audio = seg['duration']
            len_src = len(src)
            len_trg = len(trg)
            yml_src_ratios = len_src / len_audio
            yml_trg_ratios = len_trg / len_audio

            # sample a new audio duration
            new_duration = min(len_audio, (np.random.random_sample() * len_audio) + 1)
            partial_seg = seg.copy()
            partial_seg['duration'] = new_duration

            # add original and prefixed segments to yaml file
            modified.append(partial_seg)
            modified.append(seg)

            # find corresponding text prefixes via audio-to-token ratio
            new_src_index = int(round(new_duration*yml_src_ratios,0))
            new_trg_index = int(round(new_duration*yml_trg_ratios,0))
            if new_src_index == 0:
                new_src_index = 1
            if new_trg_index == 0:
                new_trg_index = 1

            # write original and prefixed segments to src and trg files
            src_out.write(' '.join(src[:new_src_index])+'\n')
            src_out.write(' '.join(src)+'\n')
            trg_out.write(' '.join(trg[:new_trg_index])+'\n')
            trg_out.write(' '.join(trg)+'\n')

    with open(args.yaml_file+'.prefix', 'w') as outfile:
        yaml.dump(modified, outfile, default_flow_style=None)


if __name__ == '__main__':

    args = parse_args()
    main(args)
