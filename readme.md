## Requirements

Python 3.6.13, Pytorch 1.9.0, Numpy 1.17.0, argparse and configparser


To reproduce the results reported in paper, simply run
```
cd model
python Run.py --model MLPM --batch_size 4 --seed 216 --finetune True --finetune_scale 100 --normalizer max11 --num_layers=4 --log_step 700
```



