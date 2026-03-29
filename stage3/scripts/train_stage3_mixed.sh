#!/usr/bin/env bash
#
set -euo pipefail

# Allow KEY=VALUE overrides passed after the script name.
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    CFG=*|GPUS=*|USE_CLUSTER=*|PARTITION=*|ACCOUNT=*|QOS=*|NNODES=*|MASTER_PORT=*|RESUME_FROM=*)
      key=${arg%%=*}
      value=${arg#*=}
      printf -v "$key" '%s' "$value"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done
set -- "${EXTRA_ARGS[@]}"

CFG="${CFG:-configs/stage3/mixed/stage3_mixed_local_train.yaml}"
GPUS="${GPUS:-4}"
USE_CLUSTER="${USE_CLUSTER:-0}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
NNODES="${NNODES:-1}"
MASTER_PORT="${MASTER_PORT:-29720}"
RESUME_FROM="${RESUME_FROM:-}"

PY_ARGS=(
  -c "${CFG}"
  --use-cluster "${USE_CLUSTER}"
  --num-gpus "${GPUS}"
  --num-nodes "${NNODES}"
)

if [ -n "${PARTITION}" ]; then
  PY_ARGS+=(--partition "${PARTITION}")
fi
if [ -n "${ACCOUNT}" ]; then
  PY_ARGS+=(--account "${ACCOUNT}")
fi
if [ -n "${QOS}" ]; then
  PY_ARGS+=(--qos "${QOS}")
fi
if [ -n "${RESUME_FROM}" ]; then
  PY_ARGS+=(--resume-from "${RESUME_FROM}")
fi

export MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT}"

PYTHONPATH=sam3:. python \
  stage3/train_stage3.py \
  "${PY_ARGS[@]}"
