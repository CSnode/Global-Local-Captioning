# Global-Local-Captioning
Source code for Chinese image captioning method based on global-local feature fusion.

### License
This code is released under the MIT License (refer to the LICENSE file for details).

## Dependencies
The code dependencies are managed with virutalenv, for installation of virtualenv:

	pip install virtualenv

For convenience, you can alse use requirements.txt to install python dependencies:

	virtualenv --no-site-packages Packages
	source Packages/bin/activate
	pip install -r requirements.txt
  
To use the evaluation script: see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## Hardware
We highly recommend you to equip a GPU card. To select gpu id, please use

	export CUDA_VISIBLE_DEVICES=id

## Caption model
We have four differt model to generate captions. EsenNIC is a basic model for our global local feature fusion method, MultiEsenNIC use Multi Attention, and MaxEsenNIC use Maxout Network to capture most important feature of multiple attention map. For different model, please look at EsenNIC, MultiEsenNIC and MaxEsenNIC, training and testing command for eash caption model is the same. For simplicity, we use EsenNIC as an example:

	cd EsenNIC

## Prepare data
To generate training data, please use make_dataset.py script:

	python make_dataset.py --input ../Data/flickr8k-cn.json --word_min 3 --output_caption ../Data/annotations --output_json ../Data/annotations.json --output_path ../Data/features --image_path ../Data/

## Train model
Please run train_model.py using gpu:

	CUDA_VISIBLE_DEVICES=0 python train_model.py --input_json ../Data/annotations.json --input_captions ../Data/annotations.h5 --input_global_feature ../Data/features_global --input_local_feature ../Data/features_local

## Generate caption
Use generate_caps.py to load models and generate caption:

	python generate_caps.py --vocab ./vocabulary.pkl --checkpoint ./model-final.pth --image ../Data/test/xxx.jpg

## Test metrics
For metrics evaluation process:

	python test.py --vocab ./vocabulary.pkl --checkpoint ./model-final.pth --input_json ../Data/annotations.json --input_captions ../Data/annotations.h5 --input_global_feature ../Data/features_global/ --input_local_feature ../Data/features_local/
   
## Valid metrics
For valid metrics, please look at the generated pkl file during training:

	ipython
	import cPickle
 	with open('records.pkl') as f:
 	  data = cPickle.load(f)
