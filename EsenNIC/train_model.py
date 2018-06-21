import torch
import numpy as np
import argparse
from model import train


def main(params):
	train(argparse.Namespace(**{\
				"mini_batch": params["mini_batch"],
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
				"max_epochs": params["max_epochs"],
				"use_dropout": params["use_dropout"],
				"save_per_epoch": params["save_per_epoch"], 
				"validFreq": params["validFreq"],
				"use_metrics": params["use_metrics"]
		}))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="train model")

	parser = argparse.ArgumentParser()
	parser.add_argument("--mini_batch", type=int, default=10,
						help="mini batch")
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
				"learning_rate": 0.0005, 
				"optimizer": "adam",
				"max_epochs": 20,
				"use_dropout": True,
				"save_per_epoch": False, 
				"validFreq": 400,
				"use_metrics": True
				}

	defaults.update(eval("dict({})".format(vars(args))))
	main(defaults)
