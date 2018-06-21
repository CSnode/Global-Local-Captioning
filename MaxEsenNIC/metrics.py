import numpy as np
import torch
import pycocotools
import pycocotools.coco
import pycocoevalcap
import pycocoevalcap.eval
import json


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
	arg3.init_it('val')

	all_images_num = 0
	cost = 0
	all_cost = 0
	process_cost = 1e-8
	result = []
	while True:
		input_feature = arg3.homogeneous_iter('val')
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

