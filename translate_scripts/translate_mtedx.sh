SRCLANG=$1
TRGLANG=$2
MODELTYPE=$3
SEGTYPE=$4
PORT=$5

DATA_DIR=# path to director where data lies
FAIRSEQ_DIR=# path to fairseq directory
MODEL_DIR=# path to directory where model checkpoints lie
WORKING_DIR=# path to directory where scripts lie

wav_files=$DATA_DIR/data/iwslt2021/wav
yaml_file=$DATA_DIR/iwslt2021.$SEGTYPE.yaml

output_folder=$DATA_DIR/${MODELTYPE}_${SEGTYPE}_$SRCLANG-$TRGLANG
rate=2.0
output_file=${MODELTYPE}_no_biased_beam_${SEGTYPE}_$rate.$SRCLANG-$TRGLANG
model=$MODEL_DIR/checkpoint17.pt
temp_wav_files=$DATA_DIR/temp_wav_${MODELTYPE}_no_beam_${SEGTYPE}_$rate

rm $output_folder/$output_file.temp.json
rm $output_folder/$output_file.$TRGLANG
rm $output_folder/$output_file.annotation

mkdir -p $temp_wav_files
mkdir -p $output_folder

# For biased beam search set variables below and add parameters to config
beam_beta=0.25
beam_mask=5
# --prefix-bias-beta $beam_beta
# --prefix-bias-mask $beam_mask

# Start fairseq translation model in a server
python $WORKING_DIR/app.py $DATA_DIR/data/iwslt2021/txt \
    --task speech_text_joint_to_text --path $model \
    --config-yaml $MODEL_DIR/config.yaml --max-tokens 3700000 \
    --beam 5 --user-dir $FAIRSEQ_DIR/examples/speech_text_joint_to_text \
    --load-speech-only --max-source-positions 3700000 --nbest 1 --batch-size 1 \
    --scoring sacrebleu --lenpen 1.0 --quiet --infer-target-lang $TRGLANG --port $PORT &

sleep 520

# Retranslate every $rate seconds
for file in $(grep -Po 'wav: .+wav' $yaml_file | uniq | awk -F ' ' '{print $2}');
  do grep $file $yaml_file | grep -Po 'offset: \d+.?\d+' | grep -Po '\d+.?\d+' > $output_file.offsets.txt;
  grep $file $yaml_file | grep -Po 'duration: \d+.?\d+' | grep -Po '\d+.?\d+' > $output_file.durations.txt;
  for line in $(paste -d ',' $output_file.offsets.txt $output_file.durations.txt);
    do offset=$(echo $line | awk -F',' '{print $1}');
    duration=$(echo $line | awk -F',' '{print $2}');
    subset=$(echo $rate);
    echo $offset $duration $subset;
    while [ $(bc <<< "$subset < $duration") == "1" ];
      do end=$(echo $offset $subset | awk '{print $1 + $2}');
      ffmpeg -i $wav_files/$file -acodec copy -ss $offset -to $end $temp_wav_files/$subset.wav;
      echo 'P' $end $offset $end >> $output_folder/$output_file.annotation;
      subset=$(bc <<< "$subset+$rate");
    done;
    end=$(echo $offset $duration | awk '{print $1 + $2}');
    ffmpeg -i $wav_files/$file -acodec copy -ss $offset -to $end $temp_wav_files/$subset.wav;
    echo 'C' $end $offset $end >> $output_folder/$output_file.annotation;
    echo $(ls -v1 $temp_wav_files/* ) | python jsonify.py -o $output_folder/$output_file.temp.json;
    curl -X POST -H "Content-Type: application/json" --retry 5 --retry-connrefused -d @$output_folder/$output_file.temp.json http://127.0.0.1:$PORT/translation| python dejsonify.py | grep -P '^D' | awk -F'\t' '{print $3}' >> $output_folder/$output_file.$TRGLANG && rm $temp_wav_files/* && rm $output_folder/$output_file.temp.json ;
    echo '<EOS>' >> $output_folder/$output_file.$TRGLANG ;
  done;
done

rm -r $temp_wav_files
rm $output_file.offsets.txt
rm $output_file.durations.txt
