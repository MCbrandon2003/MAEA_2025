# MAEA_2025
毕业论文MAEA代码库
# 相关资源下载
# 数据集
将文件放在 ./CDTA/dataset目录下
[birds-400](https://github.com/LiulietLee/CDTA/releases/download/v1.1/birds-400.zip)
[food-101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
[comic books](https://www.kaggle.com/datasets/cenkbircanoglu/comic-books-classification)
[oxford 102 flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

# 预训练模型
将文件放在 ./CDTA/pretrained 目录下
目标模型: [birds-400](https://github.com/LiulietLee/CDTA/releases/download/v1.0/birds-400.zip)
        [food-101](https://github.com/LiulietLee/CDTA/releases/download/v1.0/food-101.zip)
        [comic books](https://github.com/LiulietLee/CDTA/releases/download/v1.0/Comic.Books.zip)
        [oxford 102 flower](https://github.com/LiulietLee/CDTA/releases/download/v1.0/Oxford.102.Flower.zip)
特征提取器: [feature extractor](https://github.com/LiulietLee/CDTA/releases/download/v1.0/simsiam_bs256_100ep_cst.tar)

# MAE检查点数据
预训练MAE： [vit-large-ganloss](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
)
对抗生成器MAE: [MAE_CDTA](https://drive.google.com/file/d/1dgo_3dcoaqFIeb0iRBvabuFN3KYpa_V1/view?usp=drive_link)
             [MAE_FIA](https://drive.google.com/file/d/1-KZKvlsun6i84q7g9CVvMTf222yESyKc/view?usp=drive_link)

# MAE+CDTA:
'''
python train_CDTA.py
'''

# MAE+FIA:
'''
python train_FIA.py
'''