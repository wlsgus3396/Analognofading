for setting in 1 2 3 4 5 6 7 8 9 10 
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute sketch --T 3000 --po 20 --K=3000 --cc=25 --setting $setting
done

