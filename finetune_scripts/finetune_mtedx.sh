
DATA_DIR=# path to where prepared data tsv files lie
FAIRSEQ_DIR=# path to fairseq directory
MODEL_DIR=# path where checkpoints of original model lie
OUTPUT_DIR=# path where new checkpoints should be saved

python train.py $DATA_DIR \
    --save-dir $OUTPUT_DIR \
    --num-workers 1 \
    --task speech_text_joint_to_text \
    --arch dualinputxmtransformer_base \
    --user-dir $FAIRSEQ_DIR/examples/speech_text_joint_to_text \
    --train-subset train_es_en_tedx,train_fr_en_tedx,train_it_en_tedx,train_pt_en_tedx \
    --valid-subset valid_es_en_tedx,valid_fr_en_tedx,valid_it_en_tedx,valid_pt_en_tedx \
    --max-update 111902 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 5e-05 --update-freq 4 --clip-norm 1.0 --log-format simple \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --guide-alpha 0.8 --disable-text-guide-update-num 5000 \
    --label-smoothing 0.3 --max-tokens 500000 --max-sentences 3 --max-tokens-valid 800000 \
    --max-source-positions 800000 --enc-grad-mult 2.0 --seed 1 \
    --attention-dropout 0.3 --warmup-updates 5000 --attentive-cost-regularization 0.02 \
    --mbart-dropout 0.3 --ddp-backend no_c10d \
    --log-interval 200 --config-yaml $MODEL_DIR/config.yaml \
    --keep-last-epochs 5 --skip-invalid-size-inputs-valid-test --skip-encoder-projection \
    --finetune-w2v-params all --finetune-mbart-decoder-params all --finetune-mbart-encoder-params all \
    --stack-w2v-mbart-encoder --drop-w2v-layers 12 --normalize --load-speech-only \
    --w2v-path $MODEL_DIR/xlsr_53_56k.pt --load-pretrained-mbart-from $MODEL_DIR/mbart.pt \
    --save-interval 1 --restore-file $MODEL_DIR/checkpoint17.pt
