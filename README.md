# GastritisMIL: an interpretable deep learning model for the comprehensive histological assessment of chronic gastritis

This repository provides scripts to reproduce the results in the paper "GastritisMIL: an interpretable deep learning model for the comprehensive histological assessment of chronic gastritis".

GastritisMIL performed similarly to the two senior pathologists and outperformed the junior pathologist by a large extent, efficiently identifying abnormal alterations with a WSI-level interpretation heatmap and reducing the risk of missed diagnoses.
![image](https://github.com/nicedoctor123/Gastritis-MIL-pathology/blob/main/Workflow.png)

## Pubilications
Accept in press. 
Please download it from this following [link]([https://www.sciencedirect.com/science/article/pii/S2666389925001345]).

## Model Weights
Due to the large size of the weight file, it cannot be uploaded to GitHub. Please download it from this following [Google Drive link](https://drive.google.com/drive/folders/1M5kGVhZDXkNlSYmF4myFfYiVo1JRKgoT?usp=sharing).

## Standard datasets for publication
Our dataset required one and a half years of meticulous collection and rigorous review. This extensive effort has resulted in a valuable resource that bridges a gap in CG assessment. To enable future researchers to utilize similar dataset for developing more efficient models and advancing data science, the data for which permission has been granted (800 WSIs and associated labels) can currently be released.
To the best of our knowledge, these data will currently be the first and largest public collection of digital pathology slides in Chronic Gastritis.

Please see [this](https://www.scidb.cn/detail?dataSetId=83ee1521074742cdaae997cf0b46a7b1&version=V1) in ScienceDB platform.


## Requirements

### Software Stacks
You should install the following native libraries manually in advance.

- CUDA 11.1

CUDA is essential for PyTorch to enable GPU-accelerated deep neural network training. Please see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ .

- Python 3.9.20

The development kit should be installed.
```
sudo apt install python3.9-dev
```

- OpenSlide 1.3.1
OpenSlide is a library for reading whole slide image files (also known as virtual slides).
Please see the installation guide in https://github.com/openslide/openslide.

### Python Packages

  - h5py 3.10.0
  - numpy 1.26.3
  - openslide-python 1.3.1
  - pandas 2.2.0
  - scikit-learn 1.4.0
  - scipy 1.12.0
  - torchvision 0.16.2
  - tensorboard 2.15.1
  - torch 2.1.2
  - torch_geometric 2.4.0
  - utils 1.0.2

## Usage

### Whole-slide images model (GastritisMIL)

Here, we take the config with 40x-magnified input, which possibly for modified for your own dataset.

Our model requires slide-level labels to be trained and tested.

### Prepare Patch Futures
To preprocess WSIs, we used [CLAM](https://github.com/mahmoodlab/CLAM/tree/master#wsi-segmentation-and-patching). 
ResNet50 model and weight can be found in [this](https://github.com/pytorch/vision).
The code for other feature extractors is currently being organized and will be updated upon acceptance of the manuscript. 

### Patching
By utilizing the following code, you can save the corresponding mask of the pathological images, and subsequently segment the image into patches based on the mask to obtain the respective coordinates.


`create_patches_by_hsv2.py`:

```shell
python creat_patches_by_hsv2.py \
--source ./gastritis/Data_Directory \
--patch_size 256 \
--patch_level 0 \
--save_dir ./gastritis/40x_coord \
--downsample_factor 1.0
```

### Feature Extraction
`resnet_custom.py`:
```
def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model
```

`extract_features_new.py`:
```
def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):

	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)

	x, y = dataset[0]
	kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features) 

	if verbose > 0: 
		print(f'processing {file_path}: total of {len(loader)} batches')

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print(f'batch {count}/{len(loader)}, { count * batch_size} files processed')
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path
```
### Train GastritisMIL
The complete code will undergo thorough insepection, organization, and updates after publication.
Note: label 1 = Task 1 Inflammation, label 2 = Task 2 Activity, label 3 = Task 3 Antrum, label 4 = Task 4 Intestinal Metaplasia

`train.py`:

```shell
python train.py \
--seed 1024 \
--model_name GastritisMIL \
--label_name label1 \
--batch_size 128 \
--fea_dir ... \
--feat_size 1024 \
--output_dir ... 
```

### Inference

`Inference.py`:

```
def evaluate(model, dataloader, device):
    model.eval()
    
    with torch.no_grad():
        slide_ids, preds, labels, probs= [],[],[],[]

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0) 

            bag = bag.to(device)
            label = label.to(device)
            output = model(bag)
            _, pred = torch.max(output, dim=1)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

    return slide_ids, labels, preds, probs
```
You can load the aforementioned model weights in the `model_pretrain` section.

```shell
python Inference.py \
--label_name label1 \
--label_dir ... \
--fea_dir ... \
--output_dir ... \
--model_name GastritisMIL \
--model_pretrain ...
```

## Citation

```
@article{xia:patterns:2025,
  title = {GastritisMIL: An interpretable deep learning model for the comprehensive histological assessment of chronic gastritis},
  author = {Xia, Kun and Hu, Yihuang and Cai, Shuntian and Lin, Mengjie and Lu, Mingzhi and Lu, Huadong and Ye, Yuhan and Lin, Fenglian and Gao, Liang and Xia, Qingan and Tian, Ruihua and Lin, Weiping and Xie, Lei and Tan, Decheng and Lu, Yapi and Lin, Xunting and Yang, Xiaoning and Zhong, Lingfeng and Xu, Lei and Zhang, Zhixin and Wang, Liansheng and Ren, Jianlin and Xu, Hongzhi},
  journal = {Patterns},
  year = {2025},
  doi = {https://doi.org/10.1016/j.patter.2025.101286}
}

@misc{Xia2025,
  author       = {Xia, K. and Hu, Y. and Wang, L. and Xu, H.},
  title        = {Comprehensive Assessment of Chronic Gastritis on WSI Data},
  year         = {2025},
  note         = {ScienceDB},
  doi          = {10.57760/sciencedb.19700},
  url          = {https://doi.org/10.57760/sciencedb.19700},
}
```




