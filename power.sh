

for po in  -20
do
    python main_fed2.py --dataset mnist --num_users 51 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 3 --iid False --add_error True
done

for po in  -10
do
    python main_fed2.py --dataset mnist --num_users 51 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 2 --iid False --add_error True
done

