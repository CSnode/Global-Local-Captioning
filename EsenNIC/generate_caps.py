import argparse
import torch
from extract_conv_feats import Res
from extract_conv_feats import CNN
from extract_conv_feats import BlocksModule
import skimage
import skimage.io
import torchvision
import cPickle
import os
from model import EsenNICModel
import numpy as np


def onehot2sentence(arg1, arg2):
	row_count, column_count = arg2.size()
	sentence = list()
	for i in xrange(row_count):
		word = ''
		for j in xrange(column_count):
			index = arg2[i,j]
			if index <= 0 :
				break
			else:
				if j >= 1:
					word += ' '
				word += arg1[str(index)]

		sentence.append(word)
	return sentence

def main(params):
	with open(params['vocab']) as f:
		vocab = cPickle.load(f)

	params = {\
				"mini_batch": 1,
				"dataset": params["dataset"], 
				"model": params["model"], 
				"dim_word": params["dim_word"],
				"ctx_dim": params["ctx_dim"],
				"mini_batch": params["mini_batch"],
				"input_json": params["input_json"],
				"input_captions": params["input_captions"],
				"input_global_feature": params["input_global_feature"],
				"input_local_feature": params["input_local_feature"],
				"prev2out": params["prev2out"],
				"ctx2out": params["ctx2out"],
				"learning_rate": params["learning_rate"], 
				"optimizer": params["optimizer"],
				"beam": 1,
				"max_epochs": params["max_epochs"],
				"use_dropout": params["use_dropout"],
				"save_per_epoch": params["save_per_epoch"], 
				"validFreq": params["validFreq"],
				"use_metrics": params["use_metrics"],
				"vocab": params["vocab"],
				"checkpoint": params["checkpoint"],
				"image": params["image"],
				"banba_num": len(vocab),
				"semini_num": 40
			}
	cn = CNN(BlocksModule, [3, 4, 23, 3])
	cn.load_state_dict(torch.load(os.path.join('./data/res','res1.pth')))
	resnet = Res(cn)
	resnet.cuda()
	resnet.eval()
	temp = skimage.io.imread(params['image'])
	if len(temp.shape) == 2:
		temp = temp[:,:,np.newaxis]
		temp = np.concatenate((temp,temp,temp), axis=2)

	temp = temp.astype('float32')/255.0
	temp = torch.from_numpy(temp.transpose([2,0,1])).cuda()
	temp = torch.autograd.Variable(torchvision.transforms.Compose([torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(temp), volatile=True)
	global_feature, local_feature = resnet(temp, 14)

	esen = EsenNICModel(argparse.Namespace(**params))
	esen.load_state_dict(torch.load(params['checkpoint']))
	esen.cuda()
	esen.eval()
	global_feature = torch.autograd.Variable(torch.from_numpy(np.expand_dims(global_feature.data.cpu().float().numpy(), axis=0)), volatile=True).cuda()
	local_feature = torch.autograd.Variable(torch.from_numpy(np.expand_dims(local_feature.data.cpu().float().numpy(), axis=0)), volatile=True).cuda()

	params['mini_batch'] = 1

	data_l, odd = esen.next(global_feature, local_feature, params)

	generated = onehot2sentence(vocab, data_l)

	print generated[0]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="generate caption for image")

	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', required=True, 
						help='vocab')  
	parser.add_argument('--checkpoint', required=True, 
						help='checkpoint')  
	parser.add_argument('--image', required=True, 
						help='image')  

	args = parser.parse_args()

	defaults = {"mini_batch": 10,
				"dataset": "flickr8kcn", 
				"model": "esennic", 
				"dim_word": 512,
				"ctx_dim": 512,
				"mini_batch": 10,
				"input_json": "../Data/annotations.json",
				"input_captions": "../Data/annotations.h5",
				"input_global_feature": "../Data/features_global",
				"input_local_feature": "../Data/features_local",
				"prev2out": True,
				"ctx2out": True,
				"learning_rate": 0.0001, 
				"optimizer": "adam",
				"max_epochs": 20,
				"use_dropout": True,
				"save_per_epoch": False, 
				"validFreq": 200,
				"use_metrics": True
				}

	defaults.update(eval("dict({})".format(vars(args))))
	main(defaults)

