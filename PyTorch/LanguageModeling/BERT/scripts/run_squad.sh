#!/usr/bin/env bash

# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Container nvidia build = " $NVIDIA_BUILD_ID

CUR_DIR=$(cd $(dirname $0);pwd)

if [ ! -v SQUAD_DIR ] || [ -z "$SQUAD_DIR" ]; then
    echo "SQUAD_DIR is Null!!! Need to export SQUAD_DIR to find dataset. like:"
    echo "export SQUAD_DIR=/data/pytorch/datasets/BERT/squad/v1.1/"
elif [ ! -d "$SQUAD_DIR" ]; then
    echo "SQUAD_DIR is unavailable. please check."
else
    echo "SQUAD_DIR: $SQUAD_DIR"
fi

pushd $CUR_DIR/../checkpoints/
if [ ! -f "bert_large_qa.pt" ]; then
  wget 'https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_qa_squad11_amp/versions/19.09.0/files/bert_large_qa.pt'
fi
popd

usage() {
    cmd="bash scripts/run_squad.sh checkpoint epochs batch_size learning_rate warmup_proportion precision"
    cmd+=" num_gpu seed squad_dir vocab_file OUT_DIR mode CONFIG_FILE max_steps use_xla use_pjrt use_xla_profiler"
    echo -e "\nCMD: $cmd \n"

    pre_cmd="bash scripts/run_squad.sh ./checkpoints/bert_large_qa.pt 2.0 4 3e-5 0.1 fp32 "
    post_cmd+=" 1 \$SQUAD_DIR ./vocab/vocab output train-eval ./bert_configs/large.json -1 1 1 0"

    echo "======= XLA with single card ======== "
    num_card=1
    echo -e "$pre_cmd $num_card $post_cmd\n"

    echo "======= XLA with ddp 8 ======== "
    num_card=8
    echo -e "$pre_cmd $num_card $post_cmd\n"

    exit 1
}

if [ $# -lt 17 ] ; then
    usage
fi

init_checkpoint=${1:-"/workspace/bert/checkpoints/bert_uncased.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
warmup_proportion=${5:-"0.1"}
precision=${6:-"fp32"}
num_gpu=${7:-"8"}
seed=${8:-"1"}
squad_dir=${9:-"$BERT_PREP_WORKING_DIR/download/squad/v1.1"}
vocab_file=${10:-"$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"}
OUT_DIR=${11:-"/workspace/bert/results/SQuAD"}
mode=${12:-"train eval"}
CONFIG_FILE=${13:-"/workspace/bert/bert_configs/large.json"}
max_steps=${14:-"-1"}
use_xla=${15:-"1"}
use_pjrt=${16:-"0"}
use_xla_profiler=${17:-"0"}

export USE_XLA=$use_xla
export USE_PJRT=$use_pjrt
export USE_XLA_Profiler=$use_xla_profiler

mpi_command=""
if [ "$use_xla" = "1" ] ; then
  export GPU_NUM_DEVICES=$num_gpu
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  OUT_DIR+="_xla_$use_xla"
  OUT_DIR+="_pjrt_$use_pjrt"
  if [ "$use_pjrt" = "1" ] ; then
    export PJRT_DEVICE=GPU
  fi
  if [ "$use_xla_profiler" = "1" ] ; then
    OUT_DIR+="_profile_$use_xla_profiler"
  fi
elif [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

OUT_DIR+="_gpu_${num_gpu}_batch_${batch_size}"

echo -e "\n==== ENV of XLA ===="
echo "  USE_XLA(use xla to training): $use_xla"
echo "  USE_PJRT(use xrt/pjrt backend): $use_pjrt"
echo "  USE_XLA_Profiler(open xla profiler): $use_xla_profiler"
echo "  GPU_NUM_DEVICES(number of XLA device with gpu): $num_gpu"
echo "  TF_FORCE_GPU_ALLOW_GROWTH(Avoid XLA applying for all memory): true"

echo "out_dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
elif [ "$precision" = "pyamp" ] ; then
  echo "pyamp activated!"
  use_fp16=" --pyamp "
fi

CMD="python  $mpi_command run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --json-summary ${OUT_DIR}/dllogger.json "
CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
