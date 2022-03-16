for K in 5000 8000 10000
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
done
