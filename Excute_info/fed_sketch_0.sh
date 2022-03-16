for T in 1000 3000 5000 7000 10000
do
    python main_fed.py --dataset mnist --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 0 --all_clients --execute amp --T $T --po 50
done