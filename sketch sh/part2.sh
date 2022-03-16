for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 2000 --po 1 --K=2000 --cc=4000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 2500 --po 1 --K=2500 --cc=5000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 3000 --po 1 --K=3000 --cc=6000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 3500 --po 1 --K=3500 --cc=7000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 4000 --po 1 --K=4000 --cc=8000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 4500 --po 1 --K=4500 --cc=9000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 5500 --po 1 --K=5500 --cc=11000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 6000 --po 1 --K=6000 --cc=12000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 6500 --po 1 --K=6500 --cc=13000 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 7000 --po 1 --K=7000 --cc=14000 --setting $setting
done
