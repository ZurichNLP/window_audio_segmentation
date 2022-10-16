import argparse
import sys
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


def run_loop(args, modified, src_out, trg_out, src_file, trg_file, yml):

    s_i = 0
    t_i = 0
    i = 0
    seg = yml[i]
    src = src_file[s_i]
    trg = trg_file[t_i]
    pointer = seg['offset']
    prev_file = seg['wav']

    # iterate over all original segments
    while i < len(yml) -1:
        srcline = src.strip().split()
        trgline = trg.strip().split()

        # write original segment to file if not seen
        if seg not in modified:
            modified.append(seg)
            src_out.write(src)
            trg_out.write(trg)

        # generate new window starting from previous window's end
        duration = np.random.randint(10,30)
        end = pointer + duration

        segs = [seg]
        srcs = [srcline]
        trgs = [trgline]

        # move forward until the combined duration of the segments is longer
        # than the sampled duration
        while end > seg['offset'] + seg['duration'] and i < len(yml) -1 and seg['wav'] == segs[0]['wav']:
            i += 1
            s_i += 1
            t_i += 1
            seg = yml[i]
            src = src_file[s_i]
            srcline = src.strip().split()
            trg = trg_file[t_i]
            trgline = trg.strip().split()

            # write original segment to file if not seen
            if seg not in modified:
                modified.append(seg)
                src_out.write(src)
                trg_out.write(trg)

            segs.append(seg)
            srcs.append(srcline)
            trgs.append(trgline)

        # set pointer in first segment in window to end of
        # last segment in previous window
        partial_seg = segs[0].copy()
        partial_seg['offset'] = pointer
        partial_seg['duration'] = duration

        # ensure we do not process same windows again
        if partial_seg not in modified:

            modified.append(partial_seg)

            # trim the first segment in the window to start where the
            # last segment in the previous window ended
            start_seg = segs[0]
            start_ratio =  (pointer - start_seg['offset']) / start_seg['duration']
            start_src_id = int(round(len(srcs[0]) * start_ratio, 0))
            start_src = ' '.join(srcs[0][start_src_id:])
            start_trg_id = int(round(len(trgs[0]) * start_ratio, 0))
            start_trg = ' '.join(trgs[0][start_trg_id:])

            # segments that are in the middle of the window should be added
            # with full text
            mid_src = ' '.join([' '.join(j) for j in srcs[1:-1]])
            mid_trg = ' '.join([' '.join(j) for j in trgs[1:-1]])

            # trim the last segment in the window to end with the sampled
            # duration
            end_seg = segs[-1]
            end_ratio =  (end - end_seg['offset']) / end_seg['duration']
            end_src_id = max(int(round(len(srcs[-1]) * end_ratio, 0)) + 1, len(srcs[-1]))
            end_src = ' '.join(srcs[-1][:end_src_id])
            end_trg_id = max(int(round(len(trgs[-1]) * end_ratio, 0)) + 1, len(srcs[-1]))
            end_trg = ' '.join(trgs[-1][:end_trg_id])

            # if the current segment belongs to an new TED talk,
            # skip last segment
            if seg['wav'] != prev_file:
                pointer = seg['offset']
                end = seg['offset']
                prev_file = seg['wav']

                end_src = ''
                end_trg = ''

            # only add segment once if the window only covers one segment
            elif len(srcs) == 1:

                start_seg = segs[0]
                start_ratio =  (pointer - start_seg['offset']) / start_seg['duration']
                start_src_id = int(round(len(srcs[0]) * start_ratio, 0))
                start_trg_id = int(round(len(trgs[0]) * start_ratio, 0))

                end_seg = segs[0]
                end_ratio =  (end - end_seg['offset']) / end_seg['duration']
                end_src_id = max(int(round(len(srcs[0]) * end_ratio, 0)) + 1, len(srcs[0]))
                end_trg_id = max(int(round(len(trgs[0]) * end_ratio, 0)) + 1, len(trgs[0]))

                start_src = ' '.join(srcs[0][start_src_id:end_src_id])
                start_trg = ' '.join(trgs[0][start_trg_id:end_trg_id])

                mid_src = ''
                mid_trg = ''

                end_src = ''
                end_trg = ''

            # write windowed segments to src and trg files
            comb_src = start_src+' '+mid_src+' '+end_src
            src_out.write(f'{comb_src.strip()}\n')
            comb_trg = start_trg+' '+mid_trg+' '+end_trg
            trg_out.write(f'{comb_trg.strip()}\n')

        pointer = end
        
    return modified


def main(args):

    modified = []

    with open(args.yaml_file) as yml_file, \
         open(args.src_file) as src_file, \
         open(args.trg_file) as trg_file, \
         open(args.src_file+'.window', 'w') as src_out, \
         open(args.trg_file+'.window', 'w') as trg_out:

        yml = yaml.safe_load(yml_file)
        src_file = src_file.readlines()
        trg_file = trg_file.readlines()

        # we want to train on same amount of segments as with the other models
        # so we need to create several window sequences because they are longer
        print('Start 1st loop')
        modified.append(run_loop(args, modified, src_out, trg_out, src_file, trg_file, yml))
        print('Start 2nd loop')
        modified.append(run_loop(args, modified, src_out, trg_out, src_file, trg_file, yml))
        print('Start 3rd loop')
        modified.append(run_loop(args, modified, src_out, trg_out, src_file, trg_file, yml))

    with open(args.yaml_file+'.window', 'w') as outfile:
        yaml.dump(modified, outfile, default_flow_style=None)


if __name__ == '__main__':

    args = parse_args()
    main(args)
