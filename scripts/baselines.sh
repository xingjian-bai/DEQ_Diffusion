python3 experiment.py hydra.mode=MULTIRUN training.epochs=1 device='cuda:1' dataset=fasion_mnist,mnist scheduler.type=cosine,sigmoid,linear,quadratic


python3 experiment.py hydra.mode=MULTIRUN training.epochs=20 device='cuda:1' dataset=fasion_mnist,mnist scheduler.type=cosine,sigmoid,linear,quadratic

python3 experiment.py hydra.mode=MULTIRUN training.epochs=30 device='cuda' dataset=fasion_mnist,mnist scheduler.type=cosine,sigmoid,linear,quadratic
