for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute sketch --T 5000 --po 50 --K=10000 --cc=100 --nn $nn
done
