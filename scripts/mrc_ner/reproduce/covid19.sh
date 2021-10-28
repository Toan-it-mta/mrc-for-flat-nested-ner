# -*- coding: utf-8 -*-


TIME=0901
FILE=covid19
REPO_PATH=/mnt/e/Toan/mrc-for-flat-nested-ner-master
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/mnt/e/Toan/mrc-for-flat-nested-ner-master/data/datasets/ner-Covid19/data/new_type
BERT_DIR=/mnt/e/Toan/mrc-for-flat-nested-ner-master/data/models/phobert-base
OUTPUT_BASE=E:/mnt/e/Toan/mrc-for-flat-nested-ner-master/data/ouputs

BATCH=2
GRAD_ACC=4
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
LR_MINI=3e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=200
MAX_NORM=1.0
MAX_EPOCH=1
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=torch.adam
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold


OUTPUT_DIR=${OUTPUT_BASE}/mrc_ner/${TIME}/${FILE}_cased_large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--gpus="1" \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--lr_mini ${LR_MINI}

