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
         open(args.trg_file) as trg_file,  \
         open(args.src_file+'.context', 'w') as src_out, \
         open(args.trg_file+'.context', 'w') as trg_out:

        yml = yaml.safe_load(yml_file)

        modified = []
        prev_src = ''
        prev_trg = ''
        prev_seg = ''
        prev_prev_src = ''
        prev_prev_trg = ''
        prev_prev_seg = ''
        for i, (seg, src, trg) in enumerate(zip(yml, src_file, trg_file)):
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

            # find corresponding text prefixes via audio-to-token ratio
            new_src_index = int(round(new_duration*yml_src_ratios,0))
            new_trg_index = int(round(new_duration*yml_trg_ratios,0))
            if new_src_index == 0:
                new_src_index = 1
            if new_trg_index == 0:
                new_trg_index = 1

            # in 50% of cases add original segment and prefixed version of segment
            if i % 4 in [0,1] or prev_seg == '' or \
                prev_prev_seg == '' or prev_seg['wav'] != seg['wav'] or \
                prev_seg['wav'] != prev_prev_seg['wav']:

                modified.append(partial_seg)
                modified.append(seg)

                src_out.write(' '.join(src[:new_src_index])+'\n')
                src_out.write(' '.join(src)+'\n')

                trg_out.write(' '.join(trg[:new_trg_index])+'\n')
                trg_out.write(' '.join(trg)+'\n')

            # in 25% of cases add original segment and one previous segment plus
            # the prefixed version of the current segment
            elif (i % 4 == 3 and prev_seg != '' and prev_prev_seg != ''):
                partial_seg['duration'] += prev_seg['duration'] + \
                                           partial_seg['offset'] - \
                                           (prev_seg['offset'] + prev_seg['duration'])
                partial_seg['offset'] = prev_seg['offset']
                context_seg = seg.copy()
                context_seg['duration'] += prev_seg['duration'] + \
                                           context_seg['offset'] - \
                                           (prev_seg['offset'] + prev_seg['duration'])
                context_seg['offset'] = prev_seg['offset']
                modified.append(partial_seg)
                modified.append(context_seg)

                src_out.write(' '.join(prev_src + src[:new_src_index])+'\n')
                src_out.write(' '.join(prev_src + src)+'\n')

                trg_out.write(' '.join(prev_trg + trg[:new_trg_index])+'\n')
                trg_out.write(' '.join(prev_trg + trg)+'\n')

            # in 25% of cases add original segment and the two previous segments
            # plus the prefixed version of the current segment
            else:
                partial_seg['duration'] += prev_seg['duration'] + \
                                           partial_seg['offset'] - \
                                           (prev_seg['offset'] + prev_seg['duration']) + \
                                           prev_prev_seg['duration'] + \
                                           prev_seg['offset'] - \
                                           (prev_prev_seg['offset'] + prev_prev_seg['duration'])
                partial_seg['offset'] = prev_prev_seg['offset']
                context_seg = seg.copy()
                context_seg['duration'] += prev_seg['duration'] + \
                                           context_seg['offset'] - \
                                           (prev_seg['offset'] + prev_seg['duration']) + \
                                           prev_prev_seg['duration'] + \
                                           prev_seg['offset'] - \
                                           (prev_prev_seg['offset'] + prev_prev_seg['duration'])
                context_seg['offset'] = prev_prev_seg['offset']

                modified.append(partial_seg)
                modified.append(context_seg)

                src_out.write(' '.join(prev_prev_src + prev_src + src[:new_src_index])+'\n')
                src_out.write(' '.join(prev_prev_src + prev_src + src)+'\n')

                trg_out.write(' '.join(prev_prev_trg + prev_trg + trg[:new_trg_index])+'\n')
                trg_out.write(' '.join(prev_prev_trg + prev_trg + trg)+'\n')

            prev_prev_seg = prev_seg
            prev_prev_src = prev_src
            prev_prev_trg = prev_trg
            prev_seg = seg
            prev_src = src
            prev_trg = trg

    with open(args.yaml_file+'.context', 'w') as outfile:
        yaml.dump(modified, outfile, default_flow_style=None)


if __name__ == '__main__':

    args = parse_args()
    main(args)
