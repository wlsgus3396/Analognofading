for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 0 --all_clients --execute avg --T 5000 --po 30 --setting $setting --iid True --add_error True
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 0 --all_clients --execute psguide --T 5000 --po 30 --setting $setting --iid True --add_error True
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 0 --all_clients --execute onedevice --T 5000 --po 30 --setting $setting --iid True --add_error True
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 0 --all_clients --execute random --T 5000 --po 30 --setting $setting --iid True --add_error True
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 0 --all_clients --execute amp --T 5000 --po 30 --setting $setting --iid True --add_error True
done

for setting in 1 2 3 4 5 
do
    python main_fed.py --dataset mnist --num_users 50 --epochs 100 --gpu 0 --all_clients --execute D_DSGD --T 5000 --po 30 --setting $setting --iid True --add_error True
done



    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T 5000 --po $po --setting 1 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 1 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T 5000 --po $po --setting 1 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T 5000 --po $po --setting 1 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T 5000 --po $po --setting 1 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T 5000 --po $po --setting 1 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T 5000 --po $po --setting 2 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 2 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T 5000 --po $po --setting 2 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T 5000 --po $po --setting 2 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T 5000 --po $po --setting 2 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T 5000 --po $po --setting 2 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T 5000 --po $po --setting 3 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 3 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T 5000 --po $po --setting 3 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T 5000 --po $po --setting 3 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T 5000 --po $po --setting 3 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T 5000 --po $po --setting 3 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T 5000 --po $po --setting 4 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 4 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T 5000 --po $po --setting 4 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T 5000 --po $po --setting 4 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T 5000 --po $po --setting 4 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T 5000 --po $po --setting 4 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T 5000 --po $po --setting 5 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T 5000 --po $po --setting 5 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T 5000 --po $po --setting 5 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T 5000 --po $po --setting 5 --iid True --add_error True
    done

    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T 5000 --po $po --setting 5 --iid True --add_error True
    done


    for po in 10**(-3) 10**(-2.5) 10**(-2) 10**(-1.5) 10**(-1) 10**(-0.5) 10**(0) 10**(5/10) 10**(10/10) 10**(1.5) 10**(2)  
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T 5000 --po $po --setting 5 --iid True --add_error True
    done