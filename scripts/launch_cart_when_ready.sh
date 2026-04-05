#!/bin/bash
# Monitor PST900/FMB training, launch CART on first free GPU
MODULE_DIR="/mnt/forge-data/modules/04_wave8/DEF-tuni"
LOG_DIR="/mnt/artifacts-datai/logs/DEF-tuni"

while true; do
    # Check if PST900 finished (GPU 6)
    PST_LOG=$(ls -t ${LOG_DIR}/train_pst900_*.log 2>/dev/null | head -1)
    if [ -n "$PST_LOG" ] && grep -q "\[DONE\]\|EARLY STOP" "$PST_LOG" 2>/dev/null; then
        echo "[$(date)] PST900 done — launching CART on GPU 6"
        cd "$MODULE_DIR"
        PYTHONPATH="" CUDA_VISIBLE_DEVICES=6 \
        nohup .venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from def_tuni.train import load_config, train
cfg = load_config('configs/cart_paper.toml')
cfg['_source'] = 'configs/cart_paper.toml'
train(cfg)
" > ${LOG_DIR}/train_cart_$(date +%Y%m%d_%H%M).log 2>&1 &
        disown
        echo "[$(date)] CART launched PID=$!"
        exit 0
    fi

    # Check if FMB finished (GPU 5)
    FMB_LOG=$(ls -t ${LOG_DIR}/train_20*.log 2>/dev/null | head -1)
    if [ -n "$FMB_LOG" ] && grep -q "\[DONE\]\|EARLY STOP" "$FMB_LOG" 2>/dev/null; then
        echo "[$(date)] FMB done — launching CART on GPU 5"
        cd "$MODULE_DIR"
        PYTHONPATH="" CUDA_VISIBLE_DEVICES=5 \
        nohup .venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from def_tuni.train import load_config, train
cfg = load_config('configs/cart_paper.toml')
cfg['_source'] = 'configs/cart_paper.toml'
train(cfg)
" > ${LOG_DIR}/train_cart_$(date +%Y%m%d_%H%M).log 2>&1 &
        disown
        echo "[$(date)] CART launched PID=$!"
        exit 0
    fi

    sleep 120
done
