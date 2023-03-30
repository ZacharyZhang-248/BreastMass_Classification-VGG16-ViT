# BreastMass_Classification-VGG16-ViT
breast mass classification by using a global-local network

1. Dataset can be downloaded in https://www.kaggle.com/datasets/tommyngx/breastcancermasses <br/>
<br/>The dataset contains mammography with benign and malignant masses. Images in this dataset were first extracted 106 masses images from INbreast dataset, 53 masses images from MIAS dataset, and 2188 masses images DDSM dataset. Then data augmentation and contrast-limited adaptive histogram equalization are used to preprocess images. After data augmentation, Inbreast dataset has 7632 images, MIAS dataset has 3816 images, DDSM dataset has 13128 images. In addition, INbreast, MIAS, DDSM are integrated together. All the images were resized to 227*227 pixels. (Our experiments use image size of 224\*224). <br/>
<br/>Contributors: Ting-Yu Lin, Mei-Ling Huang

2. `model.py` is the definition of local model(ViT), global model(VGG16) and their fusion model. You can use one of them to train and compare results.
