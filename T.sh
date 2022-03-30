
    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T $T --po 30 --setting 1 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T $T --po 30 --setting 1 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T $T --po 30 --setting 1 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T $T --po 30 --setting 1 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T $T --po 30 --setting 1 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T $T --po 30 --setting 1 --iid False --add_error True
    done

    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T $T --po 30 --setting 2 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T $T --po 30 --setting 2 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T $T --po 30 --setting 2 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T $T --po 30 --setting 2 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T $T --po 30 --setting 2 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T $T --po 30 --setting 2 --iid False --add_error True
    done

        for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T $T --po 30 --setting 3 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T $T --po 30 --setting 3 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T $T --po 30 --setting 3 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T $T --po 30 --setting 3 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T $T --po 30 --setting 3 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T $T --po 30 --setting 3 --iid False --add_error True
    done

        for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T $T --po 30 --setting 4 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T $T --po 30 --setting 4 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T $T --po 30 --setting 4 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T $T --po 30 --setting 4 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T $T --po 30 --setting 4 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T $T --po 30 --setting 4 --iid False --add_error True
    done

        for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute avg --T $T --po 30 --setting 5 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute psguide --T $T --po 30 --setting 5 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute onedevice --T $T --po 30 --setting 5 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute random --T $T --po 30 --setting 5 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute amp --T $T --po 30 --setting 5 --iid False --add_error True
    done


    for T in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        python main_fed.py --dataset mnist --num_users 50 --epochs 25 --gpu 0 --all_clients --execute D_DSGD --T $T --po 30 --setting 5 --iid False --add_error True
    done