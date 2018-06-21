import argparse
import torch
from model import MaxEsenNICModel
import pycocotools
import pycocotools.coco
import pycocoevalcap
import pycocoevalcap.eval
import json
import homogeneous_data
from model import Model
import cPickle
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

def evaluate(arg1, arg2, arg3, arguments={}):
	arg1.eval()
	arg3.init_it('test')

	all_images_num = 0
	cost = 0
	all_cost = 0
	process_cost = 1e-8
	result = []
	while True:
		input_feature = arg3.homogeneous_iter('test')
		all_images_num += arg3.quo_num

		input_iter = [input_feature['global_feature'], input_feature['local_feature'], input_feature['captions'], input_feature['musk']]
		input_iter = [torch.autograd.Variable(torch.from_numpy(input_iter[0]), volatile=True).cuda(),torch.autograd.Variable(torch.from_numpy(input_iter[1]), volatile=True).cuda(),torch.autograd.Variable(torch.from_numpy(input_iter[2]), volatile=True).cuda(),torch.autograd.Variable(torch.from_numpy(input_iter[3]), volatile=True).cuda()]
		global_feature, local_feature, captions, musk = input_iter

		cost = arg2(arg1(global_feature, local_feature, captions), captions[:,1:], musk[:,1:]).data[0]
		all_cost += cost
		process_cost += 1

		features = [input_feature['global_feature'][np.arange(arg3.quo_num) * 5], input_feature['local_feature'][np.arange(arg3.quo_num) * 5]]
		features = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in features]
		global_feature, local_feature = features
		data_l, odd = arg1.next(global_feature, local_feature, arguments)
		
		generated = onehot2sentence(arg3.banba(), data_l)

		for k, instance in enumerate(generated):
			item = {'id': input_feature['add'][k]['id'], 'generated': instance}
			result.append(item)
			print('generate %s for image %s' %(item['generated'], item['id']))

		current_index = input_feature['point']['current_point']
		max_index = input_feature['point']['max_point']

		print('val %d of %d images...' %(current_index - 1, max_index))

		if input_feature['point']['end']:
			break

	scores = score(result)

	arg1.train()
	return all_cost/process_cost, result, scores

def score(arg):
	reference_path = 'data/flickr8kcn_all.json'
	reference = pycocotools.coco.COCO(reference_path)
	iamge_ids = reference.getImgIds()

	gts = [generated for generated in arg if generated['id'] in iamge_ids]

	for i, x in enumerate(gts):
		gts[i]['image_id'] = gts[i]['id']
		gts[i]['caption'] = gts[i]['generated']
	
	with open('/tmp/gts.json', 'w') as f:
		json.dump(gts, f)
	
	gts_json = reference.loadRes('/tmp/gts.json')
	cocoapi = pycocoevalcap.eval.COCOEvalCap(reference, gts_json)
	cocoapi.params['image_id'] = gts_json.getImgIds()
	cocoapi.evaluate()

	performances = {}
	for item, performance in cocoapi.eval.items():
		performances[item] = performance

	return performances

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
				"banba_num": len(vocab),
				"semini_num": 40
			}

	maxesen = MaxEsenNICModel(argparse.Namespace(**params))
	maxesen.load_state_dict(torch.load(params['checkpoint']))
	maxesen.cuda()
	maxesen.eval()

	homogeneousdata = homogeneous_data.HomogeneousData(argparse.Namespace(**params))

	build_model = Model()

	loss, result, scores = evaluate(maxesen, build_model, homogeneousdata, params)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="test model")

	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', required=True, 
						help='vocab')  
	parser.add_argument('--checkpoint', required=True, 
						help='checkpoint')  
	parser.add_argument('--input_json', required=True, 
						help='input json')  
	parser.add_argument('--input_captions', required=True, 
						help='input captions')
	parser.add_argument('--input_global_feature', required=True, 
						help='input global feature path')
	parser.add_argument('--input_local_feature', required=True, 
						help='input local_feature path')

	args = parser.parse_args()

	defaults = {"mini_batch": 10,
				"dataset": "flickr8kcn", 
				"model": "maxesennic", 
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
