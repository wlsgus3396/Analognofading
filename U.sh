
    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute avg --T 5000 --po 30 --setting 1 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute psguide --T 5000 --po 30 --setting 1 --iid False --add_error True
    done

    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting 1 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting 1 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting 1 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute D_DSGD --T 5000 --po 30 --setting 1 --iid False --add_error True
    done
    
    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute avg --T 5000 --po 30 --setting 2 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute psguide --T 5000 --po 30 --setting 2 --iid False --add_error True
    done

    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting 2 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting 2 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting 2 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute D_DSGD --T 5000 --po 30 --setting 2 --iid False --add_error True
    done

    
    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute avg --T 5000 --po 30 --setting 3 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute psguide --T 5000 --po 30 --setting 3 --iid False --add_error True
    done

    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting 3 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting 3 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting 3 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute D_DSGD --T 5000 --po 30 --setting 3 --iid False --add_error True
    done

    
    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute avg --T 5000 --po 30 --setting 4 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute psguide --T 5000 --po 30 --setting 4 --iid False --add_error True
    done

    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting 4 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting 4 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting 4 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute D_DSGD --T 5000 --po 30 --setting 4 --iid False --add_error True
    done

    
    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute avg --T 5000 --po 30 --setting 5 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute psguide --T 5000 --po 30 --setting 5 --iid False --add_error True
    done

    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute onedevice --T 5000 --po 30 --setting 5 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute random --T 5000 --po 30 --setting 5 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute amp --T 5000 --po 30 --setting 5 --iid False --add_error True
    done


    for U in 5 10 20 30 40 50 60 70 80 90 100
    do
        python main_fed.py --dataset mnist --num_users $U --epochs 25 --gpu 1 --all_clients --execute D_DSGD --T 5000 --po 30 --setting 5 --iid False --add_error True
    done