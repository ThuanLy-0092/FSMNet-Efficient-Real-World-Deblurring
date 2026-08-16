# 1. Thiết lập tất cả các biến môi trường
export HF_HOME=""
export TORCH_HOME=""
export CUDA_VISIBLE_DEVICES=3

# 2. Chạy nohup trực tiếp vào lệnh torchrun
nohup torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12565 \
    ./train.py -p ./options/train/RSBlur.yml > training_fsm_v2.log 2>&1 &