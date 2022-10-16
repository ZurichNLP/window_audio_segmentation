
DATA_DIR=# path to where prepared data tsv files lie
FAIRSEQ_DIR=# path to fairseq directory
MODEL_DIR=# path where checkpoints of original model lie
OUTPUT_DIR=# path where new checkpoints should be saved

python train.py $DATA_DIR \
    --save-dir $OUTPUT_DIR \
    --num-workers 1 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_m \
    --user-dir $FAIRSEQ_DIR/examples/speech_text_joint_to_text \
    --max-epoch 200 --max-update 123900 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.002 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --guide-alpha 0.8 --disable-text-guide-update-num 5000 \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000 --attentive-cost-regularization 0.02 \
    --text-sample-ratio 0.25 \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --log-format json --langpairs en-de --noise-token '"'"'▁NOISE'"'"' \
    --mask-text-ratio 0.0 --max-tokens-valid 20000 --ddp-backend no_c10d \
    --log-interval 100 --data-buffer-size 50 --config-yaml $MODEL_DIR/config.yaml \
    --keep-last-epochs 10 \
    --restore-file $MODEL_DIR/checkpoint_ave_10.pt
