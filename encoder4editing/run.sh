#!/bin/bash
export SETUPTOOLS_USE_DISTUTILS=stdlib
export SRCDIR=/src/
export WORKDIR=/work/
export PYTHONPATH="${PYTHONPATH}:/src/stylegan-xl"

nvidia-smi
pip freeze

echo $CUDA_VISIBLE_DEVICES

python3 $SRCDIR/xaug/xaug/run.py "${@:1}"

cd $SRCDIR
python3 encoder4editing/encoder4editing/scripts/train.py "${@:1}"

