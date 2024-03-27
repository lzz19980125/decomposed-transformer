export CUDA_VISIBLE_DEVICES=0

python main.py  --lr 0.0001  --win_size 100  --lambda_ 1  --num_epochs 5   --batch_size 128  --drop_out 0.2  --mode train  --dataset SWAT
python main.py  --lr 0.0001  --win_size 100  --lambda_ 1  --num_epochs 5   --batch_size 128  --drop_out 0.2  --mode test  --dataset SWAT




