# YOLOv3-ModelCompression-MultidatasetTraining

This project mainly include three parts.

1.Provides training methods for multiple mainstream object detection datasets(coco2017, coco2014, BDD100k, Visdrone, Hand)

2.Provides a mainstream model compression algorithm including pruning, quantization, and knowledge distillation.

3.Provides multiple backbone for yolov3 including Darknet-YOLOv3，Tiny-YOLOv3，Mobilenetv3-YOLOv3

本项目包含三部分内容：

1、提供多个主流目标检测数据集的预处理后文件及训练方法。

2、提供包括剪植，量化，知识蒸馏的主流模型压缩算法实现。

3、提供多backbone训练目前包括Darknet-YOLOv3，Tiny-YOLOv3，Mobilenetv3-YOLOv3。

其中：

源码使用Pytorch实现，以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3源码仓库。基于BN层的剪植方法由[coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning)提供，感谢学长在模型压缩领域的探索。


On January 4, 2020, we will provide the download link and training method of the Visdrone dataset after trimming.

On January 19, 2020, Dior, Bdd100k, and visdrone training will be completed and the converted .weights file will be completed.

On March 1, 2020, YOLOv3 based on mobilenetv3 backbone will be realized.

On April 7, 2020, implement two backbone models based on mobilenetv3, YOLOv3-mobilenet and YOLOv3tiny-mobilene-small, provide pre-training models, extend the normal clipping algorithm to the two models based on mobilenet and the YOLOv3tiny model, delete the tiny clip plant.

The model pre-training of mobilenetv3 was updated on April 27, 2020, and the layer pruning method was added. The method comes from [tanluren/yolov3-channel-and-layer-pruning/yolov3](https://github.com/tanluren/yolov3-channel-and-layer-pruning)， . Thanks for sharing.

On May 22, 2020,[ultralytics/yolov3](https://github.com/ultralytics/yolov3) was updated as the latest optimization of the YOLOv3 source code repository, and the YOLOv4 network structure and weight files were updated.

On May 22, 2020, the 8-bit fixed-point quantization method was updated and some bugs were fixed.

On July 12, 2020, the problem of YOLOv3-mobilenet's map returning to 0 after cutting and planting was fixed, see issue#41 for details.

On July 14, 2020, mobilenet was updated to support two extreme clipping methods based on shortcut and the bn fusion method of depthwise convolution.


# Environment deployment
1. Due to the YOLO implementation of [ultralytics/yolov3](https://github.com/ultralytics/yolov3) , see [ultralytics/yolov3](https://github.com/ultralytics/yolov3) for details of environment setup . Here is a brief description:


- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

You can `pip3 install -U -r requirements.txt`build the environment directly , or use conda to build it based on the .txt file.

# Currently supported features

|<center>Features</center>|<center></center>|
| --- |--- |
|<center>training</center>|
|<center>Normal training</center>|<center>√</center>|
|<center>tiny training</center>|<center>√</center>|
|<center>mobilenetv3 training</center>|<center>√</center>|
|<center>mobilenetv3-small training</center>|<center>√</center>|
|<center>Multiple data sets</center>|
|<center>Dior data set training</center>|<center>√</center>|
|<center>bdd100k data set training</center>|<center>√</center>|
|<center>visdrone dataset training</center>|<center>√</center>|
|<center>Cut plant</center>|
|<center>Sparse training</center>|<center>√</center>  |
|<center>Normal pruning</center>|<center>√</center>|
|<center>pruning</center>|<center>√</center>  |
|<center>Limit pruning(shortcut)</center>|<center>√</center> |
|<center>Layer cutting</center>|<center>√</center> |
|<center>quantify</center>|
|<center>BNN quantification</center>|<center>√</center>  |
|<center>BWN quantification</center>|<center>√</center>  |
|<center>stage-wise quantisation</center>|<center>√</center>  |
|<center>knowledge distillation</center>|<center>√</center>  |

#  Available commands

`python3 train.py --data ... --cfg ... `To train the model instruction, the -pt instruction is required when using the coco pre-training model.

`python3 test.py --data ... --cfg ... ` Test instructions for mAP

`python3 detect.py --data ... --cfg ... --source ...`To reason about the detection instructions, the default address of the source is data/samples, the output results are saved in the output file, and the detection resources can be pictures, videos, etc.

# Multi-dataset training
This project provides preprocessed data sets for YOLOv3 warehouse, configuration files (.cfg), data set index files (.data), data set category files (.names) and anchor box size ( Contains 9 boxes for yolov3 and 6 boxes for tiny-yolov3).

mAP statistics

|<center>data set</center>|<center>YOLOv3-640</center>|<center>YOLOv4-640</center>|<center>YOLOv3-mobilenet-640</center>|
| --- |--- |--- |--- |
|<center>Dior remote sensing dataset</center>|<center>0.749</center>|
|<center>bdd100k autonomous driving dataset</center>|<center>0.543</center>|
|<center>visdrone drone aerial photography dataset</center>|<center>0.311</center>|<center>0.383</center>|<center>0.348</center>|

The download address is as follows, after downloading and decompressing, copy the folder to the data directory to use.

- [COCO2017](https://pan.baidu.com/s/1KysFL6AmdbCBq4tHDebqlw)
  
  Extraction code：hjln

- [COCO2014](https://pan.baidu.com/s/1EoXOR77yEVokqPCaxg8QGg)
  
  Extraction code：rhqx

- [COCO权重文件](https://pan.baidu.com/s/1JZylwRQIgAd389oWUu0djg)

  Extraction code：k8ms
  
Training instruction

```bash
python3 train.py --data data/coco2017.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3.cfg --img-size ... --epochs ...
```


- [Dior remote sensing dataset](https://pan.baidu.com/s/1z0IQPBN16I-EctjwN9Idyg)
  
  Extraction code：vnuq

- [Dior weight file](https://pan.baidu.com/s/12lYOgBAo1R5VkOZqDqCFJQ)

  Extraction code：l8wz
  
training instruction

```bash
python3 train.py --data data/dior.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-onDIOR.cfg --img-size ... --epochs ...
```


- [bdd100k无人驾驶数据集](https://pan.baidu.com/s/157Md2qeFgmcOv5UmnIGI_g)
  
  Extraction code：8duw
  
- [bdd100k权重文件](https://pan.baidu.com/s/1wWiHlLxIaK_WHy_mG2wmAA)

  Extraction code：xeqo
  
Training instruction

```bash
python3 train.py --data data/bdd100k.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-bdd100k.cfg --img-size ... --epochs ...
```

- [visdrone数据集](https://pan.baidu.com/s/1CPGmS3tLI7my4_m7qDhB4Q)
  
  Extraction code：dy4c
  
- [YOLOv3-visdrone权重文件](https://pan.baidu.com/s/1N4qDP3b0tt8TIWuTFefDEw)

  Extraction code：87lf

- [YOLOv4-visdrone权重文件](https://pan.baidu.com/s/1zOFyt_AFiNk0fAFa8yE9RQ)

  Extraction code：xblu
  
 - [YOLOv3-mobilenet-visdrone权重文件](https://pan.baidu.com/s/1BHC8b6xHmTuN8h74QJFt1g)

  Extraction code：fb6y

Training instruction

```bash
python train.py --data data/visdrone.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-visdrone.cfg  --img-size ... --epochs ...
```

- [oxfordhand数据集](https://pan.baidu.com/s/1JL4gFGh-W_gYEEsiIQssZw)
  
  Extraction code：3du4

Training instruction

```bash
python train.py --data data/oxfordhand.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-visdrone.cfg  --img-size ... --epochs ...
```

## 1、Dior dataset
The DIRO dataset is one of the largest, most diverse and publicly available target detection datasets in the Earth observation community. Among them, the number of instances of ships and vehicles is relatively high, achieving a good balance between small instances and large instances. The picture was collected from Google Earth.

[dataset detailed instruction](https://cloud.tencent.com/developer/article/1509762)

### Detection effect
![Detection effect ](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/2.jpg)
![Detection effect ](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/3.jpg)

## 2、bdd100k dataset
bdd100是一个大规模、多样化的驾驶视频数据集，共包含十万个视频。每个视频大约40秒长，研究者为所有10万个关键帧中常出现在道路上的对象标记了边界框。数据集涵盖了不同的天气条件，包括晴天、阴天和雨天、以及白天和晚上的不同时间。

[official website](http://bair.berkeley.edu/blog/2018/05/30/bdd/)

[ Original dataset ](http://bdd-data.berkeley.edu)

[paper](https://arxiv.org/abs/1805.04687)

### Detection effect
![Detection effect ](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/1.jpg)

## 3、Visdrone dataset
The VisDrone2019 dataset was collected by the AISKYEYE team of the Machine Learning and Data Mining Laboratory of Tianjin University, China. The benchmark data set contains 288 video clips, composed of 261,908 frames and 10,209 frames as static images, captured by various drone-mounted cameras, covering a wide range of aspects, including location (from 14 thousands of kilometers away from China) Shooting in different cities), environment (urban and rural), objects (pedestrians, vehicles, bicycles, etc.) and density (sparse and crowded scenes). This data set was collected using various drone platforms (i.e. drones with different models) in various situations and under various weather and lighting conditions. These frames are manually labeled with more than 2.6 million bounding boxes, which are objects that people are often interested in, such as pedestrians, cars, bicycles, and tricycles. Some important attributes are also provided, including scene visibility, object category and occlusion to improve data utilization.

[Official website](http://www.aiskyeye.com/)

### Detection effect YOLOv3
![Detection effect ](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/4.jpg)

### Detection effect YOLOv4
![Detection effect ](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/5.jpg)
![Detection effect ](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/6.png)


# Multiple network structures
Two network structures are designed on the basis of mobilenetv3

|Structure Name |<center>backbone</center>|<center>Post processing</center> |<center>Total parameters</center> |<center>GFLOPS</center> |<center>mAP0.5</center> |<center>mAP0.5:0.95</center> |<center>speed(inference/NMS/total)</center> |<center>FPS</center> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|YOLOv3                      |38.74M  |20.39M  |59.13M  |117.3   |0.580  |0.340  |12.3/1.7/14.0 ms|71.4fps  |
|YOLOv3tiny                  |6.00M   |2.45M   |8.45M   |9.9     |0.347  |0.168  |3.5/1.8/5.3 ms  |188.7fps |
|YOLOv3-mobilenetv3          |2.84M   |20.25M  |23.09M  |32.2    |0.547  |0.346  |7.9/1.8/9.7 ms  |103.1fps |
|YOLOv3tiny-mobilenetv3-small|0.92M   |2.00M   |2.92M   |2.9     |0.379  |0.214  |5.2/1.9/7.1 ms  |140.8fps |
|YOLOv4                      |-       |-       |61.35M  |107.1   |0.650  |0.438  |13.5/1.8/15.3 ms|65.4fps  |
|YOLOv4-tiny                 |-       |-       |5.78M   |12.3    |0.435  |0.225  |4.1/1.7/5.8 ms  |172.4fps |

Training instruction：

1. YOLOv3, YOLOv3tiny and YOLOv4 are trained and tested on coco2014, YOLOv3-mobilenetv3 and YOLOv3tiny-mobilenetv3-small are trained and tested on coco2017.

2. The reasoning speed is tested on GTX2080ti*4, and the input image size is 608.
    
3. The training test set and the training set should match. Mismatch will cause the problem of false height of the map. Refer to [issue](https://github.com/ultralytics/yolov3/issues/970) for the reason

## Training instruction
1、YOLOv3
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3-608.weights --cfg cfg/yolov3/yolov3.cfg --img_size ...
```

Weight file downloaded
- [COCO pretraining weights file](https://pan.baidu.com/s/1JZylwRQIgAd389oWUu0djg)

  Extraction code：k8ms

2、YOLOv3tiny
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3tiny.weights --cfg cfg/yolov3tiny/yolov3-tiny.cfg --img_size ...
```

- [COCO pretraining weights file](https://pan.baidu.com/s/1iWGxdjR3TWxEe37__msyRA)

  Extraction code：udfe
  
3、YOLOv3tiny-mobilenet-small
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3tiny-mobilenet-small.weights --cfg cfg/yolov3tiny-mobilenet-small/yolov3tiny-mobilenet-small-coco.cfg --img_size ...
```

- [COCO pretraining weights file](https://pan.baidu.com/s/1mSFjWLU91H2OhNemsAeiiQ)

  Extraction code：pxz4

4、YOLOv3-mobilenet
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3-mobilenet.weights --cfg cfg/yolov3-mobilenet/yolov3-mobilenet-coco.cfg --img_size ...
```

- [COCO pretraining weights file](https://pan.baidu.com/s/1EI2Xh1i18CRLoZo_P3NVHw)

  Extraction code：3vm8

5、YOLOv4
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov4.weights --cfg cfg/yolov4/yolov4.cfg --img_size ...
```

- [COCO pretraining weights file](https://pan.baidu.com/s/1jAGNNC19oQhAIgBfUrkzmQ)

  Extraction code：njdg
  
# Three, model compression

## 1 Cut and plant

### 剪植特点
|<center>剪枝方案</center> |<center>优点</center>|<center>缺点</center> |
| --- | --- | --- |
|正常剪枝   |不对shortcut剪枝，拥有可观且稳定的压缩率，无需微调，支持tiny-yolov3和mobilenet系列。  |压缩率达不到极致。  |
|极限剪枝   |极高的压缩率。  |需要微调。  |
|极限剪枝2  |采用shortcut融合的方法提升剪植精度。  |针对shortcut最优的方法。|
|规整剪枝   |专为硬件部署设计，剪枝后filter个数均为8的倍数，无需微调，支持tiny-yolov3和mobilenet系列。 |为规整牺牲了部分压缩率。 |
|层剪枝     |以ResBlock为基本单位剪植，利于硬件部署。 |但是只能剪backbone，剪植率有限。 |
|层通道剪植 |先进行通道剪植再进行层剪植，剪植率非常高。 |可能会影响精度。 |

### 步骤

1.正常训练

```bash
python3 train.py --data ... -pt --batch-size ... --weights ... --cfg ...
```

2.稀疏化训练

`-sr`开启稀疏化，`--s`指定稀疏因子大小，`--prune`指定稀疏类型。

其中：

`--prune 0`为正常剪枝和规整剪枝的稀疏化

`--prune 1`为极限剪枝的稀疏化

`--prune 2`为层剪植稀疏化

指令范例：

```bash
python3 train.py --data ... -pt --batch-size 32  --weights ... --cfg ... -sr --s 0.001 --prune 0 
```

3.模型剪枝

- 正常剪枝
```bash
python3 normal_prune.py --cfg ... --data ... --weights ... --percent ...
```
- 规整剪枝
```bash
python3 regular_prune.py --cfg ... --data ... --weights ... --percent ...
```
- 极限剪枝
```bash
python3 shortcut_prune.py --cfg ... --data ... --weights ... --percent ...
```

- 极限剪枝2
```bash
python3 slim_prune.py --cfg ... --data ... --weights ... --percent ...
```

- 层剪植
```bash
python3 layer_prune.py --cfg ... --data ... --weights ... --shortcut ...
```

- 层剪植+通道剪植
```bash
python3 layer_channel_prune.py --cfg ... --data ... --weights ... --shortcut ... --percent ...
```


需要注意的是，这里需要在.py文件内，将opt内的cfg和weights变量指向第2步稀疏化后生成的cfg文件和weights文件。
此外，可通过增大代码中percent的值来获得更大的压缩率。（若稀疏化不到位，且percent值过大，程序会报错。）

### 剪植实验
1、正常剪植 
oxfordhand数据集，img_size = 608，在GTX2080Ti*4上计算推理时间

|<center>模型</center> |<center>剪植前参数量</center> |<center>剪植前mAP</center>|<center>剪植前推理时间</center>|<center>剪植率</center> |<center>剪植后参数量</center> |<center>剪植后mAP</center> |<center>剪植后推理时间</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(不微调)           |58.67M   |0.806   |0.1139s   |0.8    |10.32M |0.802 |0.0844s |
|yolov3-mobilenet(微调)   |22.75M   |0.812   |0.0345s   |0.97   |2.72M  |0.795 |0.0211s |
|yolov3tiny(微调)         |8.27M    |0.708   |0.0144s   |0.5    |1.13M  |0.641 |0.0116s |

2、规则剪植
oxfordhand数据集，img_size = 608，在GTX2080Ti*4上计算推理时间

|<center>模型</center> |<center>剪植前参数量</center> |<center>剪植前mAP</center>|<center>剪植前推理时间</center>|<center>剪植率</center> |<center>剪植后参数量</center> |<center>剪植后mAP</center> |<center>剪植后推理时间</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(不微调)           |58.67M   |0.806   |0.1139s   |0.8    |12.15M |0.805 |0.0874s |
|yolov3-mobilenet(微调)   |22.75M   |0.812   |0.0345s   |0.97   |2.75M  |0.803 |0.0208s |
|yolov3tiny(微调)         |8.27M    |0.708   |0.0144s   |0.5    |1.82M  |0.703 |0.0122s |

## 2、量化

### 低比特量化
`--quantized` 表示选取量化方法，默认值为-1，表示不采用任何量化方法。

`--quantized 0` 使用BNN量化方法。

BinaryNet: Training Deep Neural Networks withWeights and Activations Constrained to +1 or -1
[参考论文](https://arxiv.org/abs/1602.02830)

`--quantized 1` 使用BWN量化方法

XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
[参考论文](https://arxiv.org/abs/1603.05279v4)

#### stage-wise 训练策略
`--qlayers`可以用于选取Darknet中的量化区间，默认为自深层到浅层, 默认值为-1表示无量化层，有效范围为0-74，取0时表示量化所有层，取74时表示无量化层，大于74则无意义。

如：

`--qlayers 63` 表示量化Darknet主体网络中最后四个重复的残差块。

`--qlayers 38` 表示量化Darknet主体网络中从倒数第二八个重复的残差块开始，量化到Darknet主体网络结束。

以此类推，量化时可根据具体情况选择何是的量化层数，以及量化进度，推荐`--qlayers`值自74逐渐下降。

量化指令范例：

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 1 --qlayers 72
```
### 定点量化
`--quantized 2` 使用Dorefa8位定点量化方法

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 2
```

`--quantized 3` 使用Google白皮书8位定点量化方法

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 3
```
## 3、知识蒸馏

### 蒸馏方法
蒸馏方法采用基于Hinton于2015年提出的基本蒸馏方法，并结合检测网络做了部分改进。

Distilling the Knowledge in a Neural Network
[参考论文](https://arxiv.org/abs/1503.02531)

`--t_cfg --t_weights --KDstr` 在命令中加入这两个选项即可以开始蒸馏训练。

其中：

`--t_cfg`表示教师网络配置文件。

`--t_weights`表示教师网络权重文件。

`--KDstr`表示使用的蒸馏策略。

    `--KDstr 1` 直接在tencher网络的输出和student网络的输出求KLloss并加入到整体的loss中
    `--KDstr 2` 对boxloss和classloss有所区分，student不直接向teacher学习。student，teacher和GT分别求l2距离，当student大于teacher时附加一项student和gt的loss。
    `--KDstr 3` 对boxloss和classloss有所区分，student直接向teacher学习。
    `--KDstr 4` 将KDloss分为三项，boxloss，classloss和featureloss。
    `--KDstr 5` 在feature中加入Fine-grain-mask
蒸馏指令范例：

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --t_cfg ... --t_weights ...
```

通常以压缩前模型为teacher模型，压缩后模型为student模型进行蒸馏训练，提高学生网络的mAP。

### 蒸馏实验
oxfordhand数据集，使用yolov3tiny作为teacher网络，normal剪植后的yolov3tiny作为学生网络

|<center>teacher模型</center> |<center>teacher模型mAP</center> |<center>student模型</center>|<center>直接微调</center>|<center>KDstr 1</center> |<center>KDstr 2</center> |<center>KDstr 3</center> |<center>KDstr 4(L1)</center> |<center>KDstr 4(L2)</center> |<center>KDstr 5(L1)</center> |<center>KDstr 5(L2)</center> |<center>KDstr 6(L1)</center> |<center>KDstr 6(L2)</center> |<center>KDstr 7</center> |
| --- | --- | --- | --- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |
|yolov3tiny512   |0.708    |normal剪植yolov3tiny512    |0.637     |0.644    |0.648  |0.652   |0.655   |0.640   |  |
|yolov3tiny608   |0.708    |normal剪植yolov3tiny608    |0.658     |0.666    |0.661  |0.672   |0.669   |0.665   |0.673   |0.670   |0.674   |0.660   |