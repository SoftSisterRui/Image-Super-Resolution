**In the final result, we integrate ten models, four of which are from NAFSSR, three from SwinIR_LTE, and the remaining three from the RDN_LTE, SSRDEFNET, LIIF, and the fusion strategy is explained in detail in factsheet.**
**The training of the aforementioned models was conducted on 8 Nvidia V100 GPUs.**

# Requirements
```shell
cd mmediting/
pip install -r requirements.txt 
cd LTE/lte/
pip install -r requirements.txt 
```

# Train
## 1. Prepare training data 


### DIV2K
```shell
​    data
​    ├── DIV2K
​    │   ├── DIV2K_train_HR
​    │   │   ├── 0001.png  
​    │   │   ├── 0002.png
​    │   │   ├── ...
​    │   │   └── 0800.png
​    │   ├── DIV2K_train_LR
​    │   │   ├── X4  
​    │   │       ├── 0001x4.png
​    │   │       ├── 0002x4.png
​    │   │       ├── ...
​    │   │       └── 0800x4.png
​    │   ├── DIV2K_val_HR
​    │   │   ├── 0801.png  
​    │   │   ├── 0802.png
​    │   │   ├── ...
​    │   │   └── 0900.png
​    │   ├── DIV2K_val_LR
​    │   │   ├── X4  
​    │   │       ├── 0801x4.png
​    │   │       ├── 0802x4.png
​    │   │       ├── ...
​    │   │       └── 0900x4.png
​    │   ├── DIV2K_test_LR
​    │   │   ├── X4  
​    │   │       ├── 0901x4.png
​    │   │       ├── 0902x4.png
​    │   │       ├── ...
​    │   │       └── 1000x4.png
​    │   ├── train_GT.txt
​    │   ├── val_GT.txt
```


## 2. Begin to train

### model1: LIIF-EDSR
```shell
cd mmediting/
bash tool/dist_train.sh /configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py 8
```

### model2: LIIF-RDN

```shell
cd mmediting/
bash tool/dist_train.sh /configs/liif/liif-rdn-norm_c64b16_1xb16-1000k_div2k.py 8
```

### model3: SwinIR-LTE

```shell
cd LTE/lte
python train.py --config configs/train/train_swinir-lte.yaml --gpu 0,1,2,3,4,5,6,7
```

### model4: RDN_LTE

```shell
cd LTE/lte
python train.py --config configs/train/train_rdn-lte.yaml --gpu 0,1,2,3,4,5,6,7
```

### model5: SwinIR

```shell
cd mmedit
bash tools/dist_train.sh configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py 8
```

# Test
## 1. Prepare test data 
The test set format is consistent with the validation set format.

## 2. Begin to test
### model1: LIIF-EDSR

```shell
cd mmediting/
python tool/test.py /configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py /work_dirs/liif-edsr-norm_c64b16_1xb16-1000k_div2k/iter_990000.pth
python tool/test.py /configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py /work_dirs/liif-edsr-norm_c64b16_1xb16-1000k_div2k/iter_1000000.pth
```

### model2: LIIF-RDN

```shell
cd mmediting/
python tool/test.py /configs/liif/liif-rdn-norm_c64b16_1xb16-1000k_div2k.py /work_dirs/liif-rdn-norm_c64b16_1xb16-1000k_div2k/iter_995000.pth
python tool/test.py /configs/liif/liif-rdn-norm_c64b16_1xb16-1000k_div2k.py /work_dirs/liif-rdn-norm_c64b16_1xb16-1000k_div2k/iter_1000000.pth
```

### model3: SwinIR-LTE

```shell
cd LTE/lte
python test-save.py --config configs/test/test.yaml --model save/_train_swinir-lte/epoch_1.pth --window 8 --gpu 0
python test-save.py --config configs/test/test.yaml --model save/_train_swinir-lte/epoch_2.pth --window 8 --gpu 0
python test-save.py --config configs/test/test.yaml --model save/_train_swinir-lte/epoch_3.pth --window 8 --gpu 0
```

### model4: RDN_LTE

```shell
cd LTE/lte
python test-save.py --config configs/test/test.yaml --model save/_train_rdn-lte/epoch_1.pth --window 8 --gpu 0
```

### model5: SwinIR

```shell
cd mmedit
python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py work_dirs/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k/iter_295000.pth
python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py work_dirs/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k/iter_300000.pth
```
