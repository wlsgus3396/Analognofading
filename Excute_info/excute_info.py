



for setting in 1 2 3 4 5
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute D_DSGD --T 3000 --po 10 --K=3000 --cc=10 --setting $setting
done



for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 10 --epochs 50 --gpu 3 --all_clients --execute sign_major --T 3000 --po 10 --K=3000 --cc=10 --setting $setting
done

for setting in 1 2 3 4 5 6 7 8 9 10
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 0 --all_clients --execute sign_major --T 3000 --po 10 --K=3000 --cc=10 --setting $setting
done



"""21. 4.9실험세팅"""

"""0번: amp, po 1"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 0 --all_clients --execute amp --T 5000 --po 1 --K=5000 --cc=100

"""1번  amp, po 1"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 1 --all_clients --execute amp --T 5000 --po 1 --K=5000 --cc=100

"""2번: sketch po 1 iter 30 """
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30 --gpu 2 --all_clients --execute sketch --T 5000 --po 1 --K=5000 --cc=100 --nn $nn
done
"""3번: amp, po 1"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 3 --all_clients --execute amp --T 5000 --po 1 --K=5000 --cc=100

for nn in 1 2 3 4 5 6 7 8
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 0 --all_clients --execute sketch --T 5000 --po 0.1 --K=5000 --cc=1000 --nn $nn
done

for nn in 1 2 3 4 5 6 7 8
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 50 --gpu 0 --all_clients --execute sketch --T 5000 --po 0.1 --K=5000 --cc=500 --nn $nn
done





"""21. 4.8실험세팅"""

"""0번: amp, po 5"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 0 --all_clients --execute amp --T 5000 --po 5 --K=5000 --cc=100

"""1번  amp, po 5"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 1 --all_clients --execute amp --T 5000 --po 5 --K=5000 --cc=100

"""2번: sketch po 5 iter 30 """
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30 --gpu 2 --all_clients --execute sketch --T 5000 --po 5 --K=5000 --cc=100 --nn $nn
done
"""3번: amp, po 5"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 3 --all_clients --execute amp --T 5000 --po 5 --K=5000 --cc=100





"""21. 4.7 실험세팅"""

"""0번: amp, po 10"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 0 --all_clients --execute amp --T 5000 --po 10 --K=5000 --cc=100

"""1번  amp, po 10"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 1 --all_clients --execute amp --T 5000 --po 10 --K=5000 --cc=100
"""2번: sketch po 10 iter 30 """
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30 --gpu 2 --all_clients --execute sketch --T 5000 --po 10 --K=5000 --cc=100 --nn $nn
done
"""3번: amp, po 10"""
python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 30  --gpu 3 --all_clients --execute amp --T 5000 --po 10 --K=5000 --cc=100




"""21. 4.5 실험세팅"""
#python main_fed.py --dataset mnist --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 5000 --po 50 K=5000 --cc=100

"""0번: avg""" avg.sh
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute avg --T 5000 --po 50 --K=5000 --cc=100 --nn $nn
done

"""1번 sketch+momentum""" sketch_mo.sh
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute sketch --add_momentum --T 5000 --po 50 --K=5000 --cc=100 --nn $nn
done

"""2번: sketch """ sketch.sh
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 2 --all_clients --execute sketch --T 5000 --po 50 --K=5000 --cc=100 --nn $nn
done


"""3번: avg"""
for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute avg --T 5000 --po 50 --K=5000 --cc=100 --nn $nn
done



"""SKETCH: non-iid"""
"""T=5000, PO=50, CC=400, rr=25"""


#python main_fed.py --dataset mnist --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 5000 --po 50 K=10000



"""T=1000, po=50, cc=200, rr=10"""

#python main_fed.py --dataset mnist --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 1000 --po 50 --K=500


"""실험 세팅"""
""" T= 1000 / 3000 / 5000/ 7000 / 10000 """
""" po= 1 / 20 /40 / 50 / 60 / 80 / 100"""

""" K"""
""" C"""

# 2.27일 세팅
# T=5000, PO=1 / 20 / 80 / 100   K=10000, cc=100, rr=100

# 3.2일 세팅
"""1. T=3000, PO=1 / 20 / 40 / 50 / 60 / 80 / 100   K=6000, cc=100, rr=60"""

# for po in 0.01 0.1 80
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 3000 --po $po --K=6000 --cc=100
# done

""" 2. T=1000, PO=1 / 20 / 40 / 50 / 60 / 80 / 100   K=2000, cc=50, rr=40"""

# for po in 1 20 40 50 60 80 100
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute sketch --T 1000 --po $po --K=2000 --cc=50
# done


"""3. T=10000, PO=1 / 20 / 40 / 50 / 60 / 80 / 100   K=20000, cc=200, rr=100"""

# for po in 1 20 40 50 60 80 100
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 2 --all_clients --execute sketch --T 10000 --po $po --K=20000 --cc=200
# done


"""4. T=7000, PO=1 / 20 / 40 / 50 / 60 / 80 / 100   K=14000, cc=140, rr=100"""

# for po in 1 20 40 50 60 80 100
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute sketch --T 7000 --po $po --K=14000 --cc=140
# done



"""3.5 일 실험 세팅"""


"""1. T=5000, K=2000 / 4000 / 5000 / 6000 / 8000 / 10000     cc=100, rr=100"""

# for K in 2000 4000 5000 6000 8000 10000
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done

"""2. T=5000, K=2000 / 4000 / 5000 / 6000 / 8000 / 10000     cc=100, rr=100"""

# for K in 2000 4000 5000 6000 8000 10000
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done

"""3. T=5000, K=2000 / 4000 / 5000 / 6000 / 8000 / 10000     cc=100, rr=100"""

# for K in 2000 4000 5000 6000 8000 10000
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 2 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done

"""4. T=5000, K=2000 / 4000 / 5000 / 6000 / 8000 / 10000     cc=100, rr=100"""

# for K in 2000 4000 5000 6000 8000 10000
# do
#   python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done



"""3.8일 세팅"""

# for K in 2000 5000 8000 10000
# do
#     python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done
#
# for K in 2000 5000 8000 10000
# do
#     python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done
#
# for K in 2000 5000 8000 10000
# do
#     python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 2 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done
#
# for K in 2000 5000 8000 10000
# do
#     python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute sketch --T 5000 --po 50 --K $K --cc=100
# done




# python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 0 --all_clients --execute sketch --T 5000 --po 50 --K=2000 --cc=100
# python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute sketch --T 5000 --po 50 --K=5000 --cc=100
# python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 2 --all_clients --execute sketch --T 5000 --po 50 --K=8000 --cc=100
# python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute sketch --T 5000 --po 50 --K=10000 --cc=100


"""3.10일 세팅"""

for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 1 --all_clients --execute sketch --T 5000 --po 50 --K=10000 --cc=100 --nn $nn
done

for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 2 --all_clients --execute sketch --T 5000 --po 50 --K=2000 --cc=100 --nn $nn
done




for nn in 1 2 3 4
do
    python main_fed.py --dataset mnist  --num_channels 1 --model cnn --num_users 50 --epochs 200 --gpu 3 --all_clients --execute sketch --T 5000 --po 50 --K=5000 --cc=100 --nn $nn
done