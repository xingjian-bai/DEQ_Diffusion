python3 experiment.py hydra.mode=MULTIRUN device='cuda:0' training.loss=huber,l1,l2 model=deq model/solver=anderson,forward scheduler.type=cosine,sigmoid,linear,quadratic model/stradegy=jacobian_free,upg,default 
python3 experiment.py hydra.mode=MULTIRUN device='cuda:1' training.loss=huber,l1,l2 model=unet scheduler.type=cosine,sigmoid,linear,quadratic  



python3 experiment.py device='cuda:2' dataset=fasion_mnist training.loss=huber scheduler.type=sigmoid model=deq  
python3 experiment.py device='cuda:2' dataset=mnist
