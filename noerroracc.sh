for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting $setting --iid True --add_error False
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting $setting --iid True --add_error False
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting $setting --iid False --add_error False
done

for setting in 1 2 3 4 5    
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting $setting --iid False --add_error False
done