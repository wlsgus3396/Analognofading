
python main_fed.py --dataset cifar --num_users 50 --epochs 300 --local_ep 5 --gpu 0 --all_clients --execute avg --T 15000 --po 30 --setting=1 --iid True --add_error True --lr 0.1

python main_fed.py --dataset cifar --num_users 50 --epochs 300 --local_ep 5 --gpu 0 --all_clients --execute avg --T 15000 --po 30 --setting=1 --iid True --add_error True --lr 0.05

python main_fed.py --dataset cifar --num_users 50 --epochs 300 --local_ep 5 --gpu 0 --all_clients --execute avg --T 15000 --po 30 --setting=1 --iid True --add_error True --lr 0.01