for setting in  4 
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute psguide --T 5000 --po 1 --K=5000 --cc=1 --setting $setting
done


