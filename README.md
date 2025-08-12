**PLUM: A Prompt-guided Lightweight Unified Model for Enhancement of Multi-degraded Underwater Image** 

![Block](PLUM_dia.pdf)

## Datasets
  - [**LSUI**](https://drive.google.com/file/d/10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL/view), 
  - [**UIEB**](https://li-chongyi.github.io/proj_benchmark.html), and
  - [**SUIM-E**](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM).

Requirements are given below.
```
Python 3.5.2
Pytorch '1.0.1.post2'
torchvision 0.2.2
opencv 4.0.0
scipy 1.2.1
numpy 1.16.2
tqdm
```
### [Checkpoints](https://drive.google.com/drive/folders/1lMp9zk5KkjvafEtpKYVso78FiWQFAmQ3?usp=sharing)

### Training
- Use the below command for training:
```
python train.py --checkpoints_dir --batch_size --learning_rate             
```
### Testing
- Use the below command for testing:
```
python test.py  
```
### For Underwater Semantic Segmentation
- To generate segmentation maps on enhanced images, follow [**SUIM**](https://github.com/xahidbuffon/SUIM). 

### Send us feedback
- If you have any queries or feedback, please contact us @(**p.alik@iitg.ac.in**).
