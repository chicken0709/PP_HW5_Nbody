#/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 bXX"
    exit 1
fi

TC=$1

srun -t 00:10:00 --gres=gpu:2 ./hw5 testcases/b${TC}.in b${TC}.out
python3 ./validate.py b${TC}.out testcases/b${TC}.out