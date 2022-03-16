for setting in 1 2 3 4 5
do
   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 2 --all_clients --execute amp --T 5000 --po 1 --K=5000 --cc=1 --setting $setting
done


