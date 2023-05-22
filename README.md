# CVPR 2023第一届大模型比赛Track2 第1名方案

### 一、赛题背景
交通场景中高性能的图像检索能力对于交通执法、治安治理具有十分重要的作用，传统的图像检索方式通常使用先对图像进行属性识别再通过与期望属性的对比实现检索能力。随着多模态大模型技术的发展，文本与图像的表征统一和模态转换已有广泛应用，使用该能力可以进一步提升图像检索的精度和灵活性。
### 二、赛题任务
本赛道旨在提升交通场景中文本图像检索的精度。因此我们将多种公开数据集以及网络数据中的交通参与者图像进行了文本描述标注从而构建了多对多的图像-文本对，选手可以在此基础上进行多模态技术的研究工作，提升文本检索图像的精度。
### 三、数据集介绍
本赛题构建了一个多交通参与者的文本检索图像数据集，该数据集以开源数据集为基础，同时使用网络爬虫技术扩充数据的丰富度。在标注方面，首先利用CV大模型丰富图像标注属性，然后利用大语言模型构造图像对应的文本标注。目前数据集的总量有153728张，其中训练集136117张，评测集17611张。数据集包含行人和车辆2类交通参与者，数据分布具体见下表。
|  类别   | 训练集  | 测试集 |
|  ----  | ----  | ---- |
| 行人  | 90000 | 10000 |
| 车辆  | 46117 | 7611 |
| 总数  | 136117 | 17611 |

### 四、流程简介
基于open_clip与blip工程，使用私有数据在开源预训练模型基础上微调，在A榜数据集上再次微调。然后找出在A榜测试集精度较高的的模型，最终融合clip和blip结果。

### 五、任务分析
1.训练集与测试集车辆图像分布差异较大，凸显灾难性遗忘问题，导致在验证集上精度提升策略对测试集无效甚至降低，采用降低学习率，降低迭代次数缓解此问题，只微调2-3个epoch。\
2.训练集噪声数据较多，使用截断loss较大值策略缓解此问题。\
3.中文车辆品牌分词问题，例如BYD，分成BYD，其语义已经完全改变。采用prompt argumentation缓解此问题。\
4.车辆，行人图像比例差异大，采用最大边padding缓解此问题。

#### 比赛数据与模型权重下载地址：
链接: https://pan.baidu.com/s/1N4cPxijlLPAA6_vti91hmQ 提取码: h2s2 

### 六、数据处理
#### padding：
等比缩放最大边224,短边用零像素padding。
#### prompt增强：
 ##### 车辆prompt增强方式：
  从颜色、品牌、车型三个维度进行增强,添加prompt.\
  color_prompt = The color of the vehicle is\
  brand_prompt = The brand of the vehicle is\
  type_prompt = The vehicle's model is\
  text = 原始txt + This is a vehicle + brand_prompt + type_prompt + color_prompt
 ##### 行人prompt增强方式：
  将行人prompt分割为两个部分，作为新增prompt。\
  例如：A male pedestrian is someone who walks on foot, is less than 18 years old, with his body facing the camera and carrying a backpack. He is in a shirt with short sleeves.\
  新增结果分别为：\
  text1 = A male pedestrian is someone who walks on foot, is less than 18 years old, with his body facing the camera and carrying a backpack. \
  text2 = He is in a shirt with short sleeves.
 
#### 图像数据增强：
训练集使用RandomAugment，['Identity','AutoContrast','Brightness','Sharpness','Equalize','ShearX', 'ShearY', 'TranslateX', 'TranslateY']


### 七、模型设计
使用clip和blip作为基本模型进行优化与融合，模型融合遵循的原则是采用尽可能差异大的模型结构和训练策略\
代码主要基于blip和clip\
源码地址：https://github.com/salesforce/BLIP  &&  https://github.com/mlfoundations/open_clip
#### BLIP
采用BLIP-large版本，训练ita+itm
#### CLIP
采用两个不同训练策略的ViT-H-14和xlm-roberta-large-ViT-H-14进行训练

#### 涨点策略
1.数据集padding\
2.prompt增强\
3.数据增强\
4.噪声loss截断\
5.少部分层微调
6.异构模型融合，以及合理的归一化策略
7.TOP10内二次排序

### 八、训练细节
#### 预训练阶段
预训练阶段使用开源，私有车辆行人类型数据以及stable-difusion生成行人数据组成。
#### 微调阶段
由于训练集和测试集车辆分布差异较大，为了降低灾难性遗忘和过拟合，采用较低的学习率和迭代次数。

### 九、模型融合
融合模型：BLIP-large + CLIP-H×3 共四个模型。\
将BLIP ita输出与三个CLIP相似度输出融合，将融合后相似度top10输入BLIP itm头，利用BLIP itm头对top10进行重排序得到最终的相似度。

### 十、注意事项
1.由于存储不够，在训练之前请使用rm -r /home/aistudio/data/data218773 将A榜提交的checkpoint删除（重启会自动恢复）
2.本工程基于V100 32GB *4，如果出现显存不够，可以尝试降低batch size，或者将amp修改为bf16或者fp16等
3.ai studio和本地每个模型结果均有稍许不同，所以融合之后精度与提交结果可能存在稍许差异，可能原因是本地训练clip与blip为两个单独的虚拟环境，而本工程是统一的虚拟环境。


