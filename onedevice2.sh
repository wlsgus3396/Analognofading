for r in 0.4 0.5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting=1 --r $r --iid 'False' --add_error 'True'
done

for r in 0.4 0.5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting=2 --r $r --iid 'False' --add_error 'True'
done


for r in 0.4 0.5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting=3 --r $r --iid 'False' --add_error 'True'
done


for r in 0.4 0.5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting=4 --r $r --iid 'False' --add_error 'True'
done


for r in 0.4 0.5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting=5 --r $r --iid 'False' --add_error 'True'
done
