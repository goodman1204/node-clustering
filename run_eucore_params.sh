#! /bin/bash
for A in 5000 10000 15000 20000 25000 30000 35000 
do
    python vaece_tkde.py --dataset synthetic --pre_gmm 5 --epochs 10 --cuda -1 --synthetic_num_nodes ${A} >> vaece_synthetic_log 
done
#for A in 5000 10000 15000 20000 25000 30000 35000 
#do
    #python nec.py --dataset synthetic --pre_gmm 5 --epochs 10 --cuda -1 --synthetic_num_nodes ${A} >> nec_synthetic_log 
#done
#for A in 5000 10000 15000 20000 25000 30000 35000 
#do
    #python vae.py --model gcn_ae --dataset synthetic --pre_gmm 5 --epochs 10 --cuda -1 --synthetic_num_nodes ${A} >> ae_synthetic_log 
#done
#for A in 5000 10000 15000 20000 25000 30000 35000 
#do
    #python vae.py --model gcn_vae --dataset synthetic --pre_gmm 5 --epochs 10 --cuda -1 --synthetic_num_nodes ${A} >> vae_synthetic_log 
#done
#for A in 5000 10000 15000 20000 25000 30000 35000 
#do
    #python can.py  --dataset synthetic --pre_gmm 5 --epochs 10 --cuda -1 --synthetic_num_nodes ${A} >> can_synthetic_log 
#done
