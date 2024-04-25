# Dalle2-pytorch
本项目是一个从0开始训练的文生图项目，使用dalle2作为整体框架，以及基于coco2014数据集（8w+图文对）进行训练。

### 生成结果

由于训练的数据并不多，所以生成的结果比较一般，能看出大致的语义内容，但是整体的一致性和细节都比较差。

基于coco2014的验证集中描述性文本进行生成，其结果如下所示：

| 3×256×256                                                                        | 3×256×256                                                                        | 3×256×256                                                                        |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| <img src="./result_good/A cheesy pizza sitting on top of a table.jpg" width=224> | <img src="./result_good/There is a small bus with several people standing next to it.jpg" width=224> | <img src="./result_good/A plane flies over water with two islands nearby.jpg" width=224> |
| A cheesy pizza sitting on top of a table. | There is a small bus with several people standing next to it. | A plane flies over water with two islands nearby. |
| <img src="./result_good/Afternoon at a dock with seagulls flying overhead.jpg" width=224>| <img src="./result_good/This table is filled with a variety of different dishes.jpg" width=224>| <img src="./result_good/Bedroom scene with a bookcase, blue comforter and window.jpg" width=224> |
| Afternoon at a dock with seagulls flying overhead. | This table is filled with a variety of different dishes. | Bedroom scene with a bookcase, blue comforter and window.|

### 训练流程

#### （0）环境搭建
```sh
pip install -r requirements.txt
```

#### （1）数据下载

执行以下命令下载coco2014数据集，并解压到data文件夹下面：
```sh
python ./data/coco_download.py
```

#### （2）模型训练

训练prior网络：
```sh
sh scrips/train_prior.sh
```

训练decoder网络：
```sh
sh scrips/train_decoder.sh
```


#### （3）模型推理

执行以下命令来进行模型推理：
```sh
python test_dalle2.py 
--test_img_path your_coco_img_path 
--test_annot_path your_coco_annoration_path
--prior_ckpt saved_prior_model_path
--decoder_ckpt saved_decoder_model_path
```

### 参考内容

1. DALLE2-pytorch: [https://github.com/lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)

2. Dalle2_pytorch_project：[https://github.com/goldiusleonard/Dalle2_pytorch_project](https://github.com/goldiusleonard/Dalle2_pytorch_project)
