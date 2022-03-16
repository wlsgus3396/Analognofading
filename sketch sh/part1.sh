for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 0.1 --K=5000 --cc=10000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 0.5 --K=5000 --cc=10000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 10 --K=5000 --cc=10000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 20 --K=5000 --cc=10000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 30 --K=5000 --cc=10000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 40 --K=5000 --cc=10000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 1 --all_clients --execute sketch --T 5000 --po 50 --K=5000 --cc=10000 --setting $setting
done
