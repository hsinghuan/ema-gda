# Domain Ensembling for Gradual Domain Adaptation

Code for experiments on gradual domain ensembling (GDE) and uncertainty-aware gradual domain ensembling (UA-GDE)

## Dataset Preparation
For the Portraits dataset, please download it [here](https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0).
If you receive an unbearable amount of warning messages while loading the Portraits dataset due to corruption, please go to the folder containing the images and execute the following command to preprocess them.

```
for file in *.png; do pv "$file" | convert - "${file%.png}.png"; done
```

## Experiment Execution
### Train GDE on Rotating MNIST
```
python train.py --dataset rotate-mnist --data_dir /path/to/torchvision/dataset/root/ --method gradual-domain-ensemble 
```

### Train UA-GDE on Rotating MNIST
```
python train.py --dataset rotate-mnist --data_dir /path/to/torchvision/dataset/root/ --method uagde
```


### Train GDE on Portraits
```
python train.py --dataset portraits --data_dir /path/to/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/ --method gradual-domain-ensemble --adapt_epochs 20
```

### Train UA-GDE on Portraits
```
python train.py --dataset portraits --data_dir /path/to/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/ --method uagde --adapt_epochs 20
```

### Test GDE/UA-GDE on Rotating MNIST/Portraits
```
python test.py --dataset [rotate-mnist/portraits] --data_dir [path/to/data/folder] --method [gradual-domain-ensemble/uagde]
```
