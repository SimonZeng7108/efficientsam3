#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
RUN_ROOT="${RUN_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_distill_smoke}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/conda_env}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
AMBIENT_HF_HOME="${HF_HOME:-}"
HF_HOME="${DISTILL_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HF_TOKEN_PATH:-}" ] && \
   [ -n "${AMBIENT_HF_HOME}" ] && [ "${AMBIENT_HF_HOME}" != "${HF_HOME}" ] && \
   [ -f "${AMBIENT_HF_HOME}/token" ]; then
  HF_TOKEN_PATH="${AMBIENT_HF_HOME}/token"
fi
PREFLIGHT_INSTALL_DEPS="${PREFLIGHT_INSTALL_DEPS:-1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0+cu128}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.26.0+cu128}"

export CONDA_PKGS_DIRS PIP_CACHE_DIR HF_HOME HF_TOKEN_PATH

mkdir -p "${RUN_ROOT}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}"

LOG_FILE="${RUN_ROOT}/preflight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Preflight log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Env: ${ENV_DIR}"
echo "Install dependencies: ${PREFLIGHT_INSTALL_DEPS}"

if ! command -v conda >/dev/null 2>&1 && [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required." >&2
  exit 1
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "Creating conda environment at ${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.12
else
  echo "Using existing conda environment at ${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ "${PREFLIGHT_INSTALL_DEPS}" = "1" ]; then
  "${PYTHON}" -m pip install -U pip setuptools wheel

  echo "Installing image-distillation dependency set"
  if ! "${PIP}" install -e "${REPO_DIR}[stage1]"; then
    echo "Full stage1 extra install failed; retrying without mmcv-heavy optional dependency resolution."
    "${PIP}" install -e "${REPO_DIR}" --no-deps
    "${PIP}" install --index-url "${PYTORCH_INDEX_URL}" --extra-index-url "${PYPI_INDEX_URL}" \
      "${TORCH_SPEC}" "${TORCHVISION_SPEC}"
    "${PIP}" install \
      "timm>=1.0.17" "numpy>=1.26.4" tqdm "ftfy==6.1.1" regex \
      "iopath>=0.1.10" typing_extensions huggingface_hub psutil \
      "decord>=0.6.0" "mmengine>=0.10.4" "pycocotools>=2.0.7" \
      "yacs>=0.1.8" "Pillow>=10.0.0" "opencv-python>=4.9.0.80" \
      "scipy>=1.10.0" "scikit-image>=0.21.0" "scikit-learn>=1.3.0" \
      "tensorboard>=2.12.0" "einops>=0.7.0" "hydra-core>=1.3.2" \
      "submitit>=1.5.1" "fvcore>=0.1.5.post20221221" \
      "fairscale>=0.4.13" pandas pyyaml segment-anything
  fi
else
  echo "Skipping dependency installation."
fi

cd "${REPO_DIR}"

"${PYTHON}" - <<'PY'
from argparse import Namespace
from pathlib import Path
from stage1.config import get_config
from stage1.model import build_image_student_model

import cv2
import mmengine
import pycocotools.mask
import timm
import torch
import torchvision
import yaml

configs = [
    "stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml",
    "stage1/configs/es_rv_s_5090_smoke.yaml",
    "stage1/configs/es_rv_m_5090_smoke.yaml",
    "stage1/configs/es_rv_l_5090_smoke.yaml",
]

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("cv2", cv2.__version__)
print("mmengine", mmengine.__version__)
print("cuda_available", torch.cuda.is_available())

for cfg in configs:
    yaml.safe_load(Path(cfg).read_text())
    args = Namespace(
        cfg=cfg,
        opts=None,
        batch_size=None,
        data_path=None,
        pretrained=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        disable_amp=False,
        only_cpu=True,
        output=None,
        tag=None,
        eval=False,
        throughput=False,
        local_rank=0,
    )
    config = get_config(args)
    print(
        "config_ok",
        cfg,
        "model=" + str(config.MODEL.NAME),
        "backbone=" + str(config.MODEL.BACKBONE),
        "samples=" + str(config.DATA.NUM_SAMPLES),
        "random_sample=" + str(config.DATA.RANDOM_SAMPLE),
        "batch=" + str(config.DATA.BATCH_SIZE),
        "epochs=" + str(config.TRAIN.EPOCHS),
    )
    if str(config.MODEL.BACKBONE).startswith("repvit"):
        model = build_image_student_model(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(
            "student_model_ok",
            cfg,
            "backbone=" + str(config.MODEL.BACKBONE),
            "params=" + str(num_params),
        )
PY

echo "Preflight complete."
