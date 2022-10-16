
set -e -o pipefail

SRCLANG=$1
TRGLANG=$2
MODELTYPE=$3

DATA_DIR=# path to director where data lies
FAIRSEQ_DIR=# path to fairseq directory
MODEL_DIR=# path to directory where model checkpoints lie
WORKING_DIR=# path to directory where scripts lie

wav_files=$DATA_DIR/data/tst-COMMON/wav
yaml_file=$DATA_DIR/data/tst-COMMON/txt/tst-COMMON.yaml
shift=2

model=$MODEL_DIR/checkpoint_ave_10.pt
output_folder=$DATA_DIR/${MODELTYPE}_$SRCLANG-$TRGLANG
window_len=15
outfile=$output_folder/${MODELTYPE}_no_biased_beam_merged-15.$SRCLANG-$TRGLANG

# Generate a file that stores the paths to each ted talk with start and end configurations
for file in $(grep -Po 'wav: .+wav' $yaml_file | uniq | awk -F ' ' '{print $2}');
  do start=0.0 ;
  line=$(grep $file $yaml_file | tail -n 1);
  offset=$(echo $line | grep -Po "offset: \d+\.\d+" | awk -F' ' '{print $2}');
  duration=$(echo $line | grep -Po "duration: \d+\.\d+" | awk -F' ' '{print $2}');
  echo $wav_files/$file $start $offset $duration >> $outfile.temp;
done

# Translate window-by-window
cat $outfile.temp | $WORKING_DIR/onlinization.py \
  -lang de -window_len $window_len -type extended -shift $shift \
  -config config.yml -ratio 0.4 -merge_last_overlap -log $outfile -slt -max_history 0.2 \
  STOP $DATA_DIR/data/tst-COMMON/txt \ # start of model parameter config
  --task speech_text_joint_to_text --path $model \
  --config-yaml $MODEL_DIR/config.yaml --max-tokens 100000 \
  --beam 5 --user-dir $FAIRSEQ_DIR/examples/speech_text_joint_to_text \
  --load-speech-only --max-source-positions 12000 --nbest 1 --batch-size 1 \
  --scoring sacrebleu --lenpen 1.0 --quiet > $outfile.stdout 2> $outfile.stderr
