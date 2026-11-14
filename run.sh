#!/usr/bin/env bash
# Use the project environment; apply the tested GL fix only for GUI commands.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash run.sh train_bc.py --data data/expert100.npz --check-data'
    echo '       bash run.sh collect_expert.py --gui --episodes 5 --output data/new_demo.npz'
    exit 0
fi
for argument in "$@"; do
    if [[ "$argument" == '--gui' ]]; then
        export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
        break
    fi
done
exec "$project_dir/.venv/bin/python" "$@"
