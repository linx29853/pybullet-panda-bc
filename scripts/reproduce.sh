#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
experiment=$(mktemp -d "$project_dir/runs/reproduce_XXXXXX")
echo "实验目录：$experiment"
bash run.sh collect_expert.py --episodes 100 --seed 42 --output "$experiment/expert100.npz"
bash run.sh train_bc.py --data "$experiment/expert100.npz" --run-dir "$experiment/model"
bash run.sh evaluate_bc.py --checkpoint "$experiment/model/best.pt" --episodes 20 --seed 10000 --output "$experiment/eval10000.json"
bash run.sh evaluate_bc.py --checkpoint "$experiment/model/best.pt" --episodes 100 --seed 20000 --output "$experiment/eval20000.json"
echo "实验结果：$experiment"
