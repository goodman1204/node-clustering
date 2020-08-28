# CAN-pytorch
This is a PyTorch implementation of the CAN model described in the paper:
PyTorch version for CAN: Co-embedding Attributed Networks based on code <https://github.com/zfjsail/gae-pytorch> and <https://github.com/mengzaiqiao/CAN> 
>Zaiqiao Meng, Shangsong Liang, Hongyan Bao, Xiangliang Zhang. Co-embedding Attributed Networks. (WSDM2019)


### Requirements
- Python 3.7.4
- PyTorch 1.5.0
- install requirements via 

```
pip install -r requirements.txt
``` 

### How to run

```
python train.py
```


#### Facebook dataset with the default parameter settings

```
Epoch: 0194 train_loss= 0.75375 log_lik= 0.69382 KL= 0.05993 train_acc= 0.73382 val_edge_roc= 0.98516 val_edge_ap= 0.98455 val_attr_roc= 0.95473 val_attr_ap= 0.95721 time= 1.71142
Epoch: 0195 train_loss= 0.75215 log_lik= 0.69217 KL= 0.05998 train_acc= 0.73484 val_edge_roc= 0.98577 val_edge_ap= 0.98492 val_attr_roc= 0.95465 val_attr_ap= 0.95746 time= 1.64731
Epoch: 0196 train_loss= 0.75135 log_lik= 0.69133 KL= 0.06002 train_acc= 0.73486 val_edge_roc= 0.98588 val_edge_ap= 0.98486 val_attr_roc= 0.95322 val_attr_ap= 0.95755 time= 1.64199
Epoch: 0197 train_loss= 0.75140 log_lik= 0.69134 KL= 0.06006 train_acc= 0.73556 val_edge_roc= 0.98545 val_edge_ap= 0.98477 val_attr_roc= 0.95652 val_attr_ap= 0.95914 time= 1.63010
Epoch: 0198 train_loss= 0.75157 log_lik= 0.69146 KL= 0.06010 train_acc= 0.73477 val_edge_roc= 0.98573 val_edge_ap= 0.98490 val_attr_roc= 0.95497 val_attr_ap= 0.95753 time= 1.65039
Epoch: 0199 train_loss= 0.75122 log_lik= 0.69107 KL= 0.06015 train_acc= 0.73400 val_edge_roc= 0.98620 val_edge_ap= 0.98523 val_attr_roc= 0.95420 val_attr_ap= 0.95829 time= 1.66717
Epoch: 0200 train_loss= 0.74931 log_lik= 0.68914 KL= 0.06017 train_acc= 0.73667 val_edge_roc= 0.98601 val_edge_ap= 0.98515 val_attr_roc= 0.95426 val_attr_ap= 0.95744 time= 1.65484
Optimization Finished!
Test edge ROC score: 0.9853779088016957
Test edge AP score: 0.9836879718079673
Test attr ROC score: 0.9578314765862058
Test attr AP score: 0.9577498373032282
```

#### CiteSeer dataset with the default parameter settings  
```
Epoch: 0198 train_loss= 0.81845 log_lik= 0.76834 KL= 0.05011 train_acc= 0.66264 val_edge_roc= 0.94756 val_edge_ap= 0.95467 val_attr_roc= 0.92974 val_attr_ap= 0.92059 time= 1.70837
/Users/storen/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:1558: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
/Users/storen/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:1558: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
Optimization Finished!
Test edge ROC score: 0.9490130318561492
Test edge AP score: 0.95856792990438
Test attr ROC score: 0.9239066109625775
Test attr AP score: 0.9121636661521142
```

#### Cora dataset with the default parameter settings 
```
Epoch: 0197 train_loss= 0.92730 log_lik= 0.85651 KL= 0.07079 train_acc= 0.64933 val_edge_roc= 0.98261 val_edge_ap= 0.97795 val_attr_roc= 0.89457 val_attr_ap= 0.88565 time= 0.85166  
Epoch: 0198 train_loss= 0.92594 log_lik= 0.85511 KL= 0.07083 train_acc= 0.65040 val_edge_roc= 0.98230 val_edge_ap= 0.97761 val_attr_roc= 0.89448 val_attr_ap= 0.88571 time= 0.82273
Epoch: 0199 train_loss= 0.92517 log_lik= 0.85432 KL= 0.07085 train_acc= 0.65058 val_edge_roc= 0.98256 val_edge_ap= 0.97801 val_attr_roc= 0.89523 val_attr_ap= 0.88651 time= 0.81977
Epoch: 0200 train_loss= 0.92596 log_lik= 0.85508 KL= 0.07088 train_acc= 0.64968 val_edge_roc= 0.98289 val_edge_ap= 0.97837 val_attr_roc= 0.89585 val_attr_ap= 0.88720 time= 0.90153
Optimization Finished!
Test edge ROC score: 0.983134251252218
Test edge AP score: 0.9817151099782778
Test attr ROC score: 0.895140476178776
Test attr AP score: 0.8847338611264453
```
