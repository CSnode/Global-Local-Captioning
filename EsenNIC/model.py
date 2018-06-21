import homogeneous_data
import torch
import time
import metrics
import cPickle


def gen_step(arg1, arg2, arg3, arg4, arg5, arg6, arg7):
	cap,index = torch.sort(arg1,1,True)
	stack = []
	height = min(arg2, cap.size(1))
	weight = arg2
	if arg3 == 0:
		weight = 1
	for temp_h in xrange(height):
		for temp_w in xrange(weight):
			temp_p = cap[temp_w,temp_h]
			stack_p = arg6[temp_w] + temp_p
			stack.append({'temp_h':index[temp_w,temp_h], 'temp_w':temp_w, 'stack_p':stack_p, 'temp_p':temp_p})
	stack = sorted(stack,  key=lambda faw: -faw['stack_p'])
	
	cc0 = [temp_cc0.clone() for temp_cc0 in arg7]
	
	if arg3 < 1:
		pass
	else:
		cap_pre = arg4[:arg3].clone()
		cap_pre_p = arg5[:arg3].clone()
	for hindex in xrange(arg2):
		h = stack[hindex]
		
		if arg3 >= 1:
			arg4[:arg3, hindex] = cap_pre[:, h['temp_w']]
			arg5[:arg3, hindex] = cap_pre_p[:, h['temp_w']]
		
		for temp_c in xrange(len(cc0)):
		
			cc0[temp_c][:, hindex] = arg7[temp_c][:, h['temp_w']]
		
		arg4[arg3, hindex] = h['temp_h']
		arg5[arg3, hindex] = h['temp_p']
		arg6[hindex] = h['stack_p']
	arg7 = cc0
	return arg4, arg5, arg6, arg7, stack

class gen_model(torch.nn.Module):
	def __init__(self):
		super(gen_model, self).__init__()

	def gen_cap(self, arg1, arg2, *args, **kwargs):
		arguments = kwargs['arguments']
		quo_num = arguments.get('beam', 1)

		cap_iter = torch.LongTensor(self.semini_num, quo_num).zero_()
		cap_iter_p = torch.FloatTensor(self.semini_num, quo_num).zero_()
		cap_iter_p_cum = torch.zeros(quo_num)
		fin_cap = []

		for temp in xrange(self.semini_num):
			pfi = arg2.data.float()
			pfi[:,pfi.size(1)-1] = pfi[:, pfi.size(1)-1] - 1000  
		
			cap_iter, cap_iter_p, cap_iter_p_cum, arg1, _ = gen_step(pfi, quo_num, temp, cap_iter, cap_iter_p, cap_iter_p_cum, arg1)

			for hindex in range(quo_num):

				if cap_iter[temp, hindex] == 0 or temp == self.semini_num - 1:
					result = {'cap': cap_iter[:, hindex].clone(), 'lp': cap_iter_p[:, hindex].clone(),'p': cap_iter_p_cum[hindex]}
					fin_cap.append(result)
					cap_iter_p_cum[hindex] = -1000

			vec = cap_iter[temp]
			arg2, arg1 = self.get_cap_p_iter(torch.autograd.Variable(vec.cuda()), *(args + (arg1,)))

		fin_cap = sorted(fin_cap, key=lambda faw: -faw['p'])[:quo_num]
		return fin_cap

class att_ctx(torch.nn.Module):
	def __init__(self, arguments):
		super(att_ctx, self).__init__()
		self.att_emb = torch.nn.Linear(512, 512)
		self.weight_emb = torch.nn.Linear(512, 1)

	def forward(self, arg1, arg2, arg3):
		feature = arg3.view(-1, 196, 512)
		
		feature_embed = self.att_emb(arg1)                   
		feature_embed = feature_embed.unsqueeze(1)
		feature_embed = feature_embed.expand_as(feature)     
		mul_feature = feature + feature_embed                
		mul_feature = torch.nn.functional.tanh(mul_feature)   
		mul_feature = mul_feature.view(-1, 512)               
		mul_feature = self.weight_emb(mul_feature)            
		mul_feature = mul_feature.view(-1, 196)                   
		
		softmax_weight = torch.nn.functional.softmax(mul_feature, dim=1) 
		temp_feature = arg2.view(-1, 196, 512) 
		aligned_feature = torch.bmm(softmax_weight.unsqueeze(1), temp_feature).squeeze(1) 

		return aligned_feature

class caption_solver(gen_model):
	def __init__(self, arguments):
		super(caption_solver, self).__init__()
		self.banba_num = arguments.banba_num
		self.semini_num = arguments.semini_num
		self.rand_dropout_p = 0.0

		self.word_emb = torch.nn.Sequential(torch.nn.Embedding(self.banba_num + 1, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5))
		self.feature_emb = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5))
		self.output_layer = torch.nn.Linear(512, self.banba_num + 1)
		self.middle_layer = torch.nn.Linear(512, 512)

	def init_tparams(self, arg1):
		temp = next(self.parameters()).data
		return (torch.autograd.Variable(temp.new(1, arg1, 512).zero_()),
				torch.autograd.Variable(temp.new(1, arg1, 512).zero_()))

	def forward(self, arg1, arg2, arg3):
		mini_batch = arg1.size(0)
		tparams = self.init_tparams(mini_batch)

		seq_output = []

		temp_feature = self.feature_emb(arg2.view(-1, 2048))
		arg2 = temp_feature.view(*(arg2.size()[:-1] + (512,)))

		embed_feature = self.middle_layer(arg2.view(-1, 512))
		embed_feature = embed_feature.view(*(arg2.size()[:-1] + (512,)))

		for i in xrange(arg3.size(1) - 1):
			if self.training and i >= 1 and self.rand_dropout_p > 0.0:
				rand_p = arg1.data.new(mini_batch).uniform_(0, 1)
				temp_musk = rand_p < self.rand_dropout_p
				if temp_musk.sum() != 0:
					select_i = temp_musk.nonzero().view(-1)
					index = arg3[:, i].data.clone()
					p_p1 = torch.exp(seq_output[-1].data)
					index.index_copy_(0, select_i, torch.multinomial(p_p1, 1).view(-1).index_select(0, select_i))
					index = torch.autograd.Variable(index, requires_grad=False)
				else:
					index = arg3[:, i].clone()
			else:
				index = arg3[:, i].clone()          
			if i >= 1 and arg3[:, i].data.sum() == 0:
				break

			word_embed = self.word_emb(index)

			step_i, tparams = self.lstm_step(word_embed, arg1, arg2, embed_feature, tparams)
			step_i = torch.nn.functional.log_softmax(self.output_layer(step_i), dim=1)
			seq_output.append(step_i)

		return torch.cat([temp_step.unsqueeze(1) for temp_step in seq_output], 1)

	def get_cap_p_iter(self, arg1, arg2, arg3, arg4, arg5):
		word_embed = self.word_emb(arg1)

		step_i, tparams = self.lstm_step(word_embed, arg2, arg3, arg4, arg5)
		step_i = torch.nn.functional.log_softmax(self.output_layer(step_i), dim=1)

		return step_i, tparams

	def f_next(self, arg1, arg2, arguments={}):
		beam = arguments.get('beam', 1)
		mini_batch = arg1.size(0)

		temp_feature = self.feature_emb(arg2.view(-1, 2048))
		arg2 = temp_feature.view(*(arg2.size()[:-1] + (512,)))

		embed_feature = self.middle_layer(arg2.view(-1, 512))
		embed_feature = embed_feature.view(*(arg2.size()[:-1] + (512,)))

		cap_iter = torch.LongTensor(self.semini_num, mini_batch).zero_()
		cap_iter_p = torch.FloatTensor(self.semini_num, mini_batch)

		self.fin_cap = [[] for x in xrange(mini_batch)]
		for k in range(mini_batch):
			tparams = self.init_tparams(beam)
			temp_global_feature = arg1[k:k+1].expand(beam, arg1.size(1))
			temp_local_feature = arg2[k:k+1].expand(*((beam,)+arg2.size()[1:])).contiguous()
			temp_embed_feature_p = embed_feature[k:k+1].expand(*((beam,)+embed_feature.size()[1:])).contiguous()

			for t in xrange(1):
				if t == 0:
					global_feature_input = arg1.data.new(beam).long().zero_()
					word_embed = self.word_emb(torch.autograd.Variable(global_feature_input, requires_grad=False))

				step_i, tparams = self.lstm_step(word_embed, temp_global_feature, temp_local_feature, temp_embed_feature_p, tparams)
				step_i_p = torch.nn.functional.log_softmax(self.output_layer(step_i), dim=1)

			self.fin_cap[k] = self.gen_cap(tparams, step_i_p, temp_global_feature, temp_local_feature, temp_embed_feature_p, arguments=arguments)
			cap_iter[:, k] = self.fin_cap[k][0]['cap'] 
			cap_iter_p[:, k] = self.fin_cap[k][0]['lp']
		return cap_iter.transpose(0, 1), cap_iter_p.transpose(0, 1)

	def next(self, arg1, arg2, arguments={}):
		beam = arguments.get('beam', 1)
		flag = 1
		wrapped = False
		if beam > 1:
			return self.f_next(arg1, arg2, arguments)

		mini_batch = arg1.size(0)
		tparams = self.init_tparams(mini_batch)

		temp_feature = self.feature_emb(arg2.view(-1, 2048))
		arg2 = temp_feature.view(*(arg2.size()[:-1] + (512,)))

		embed_feature = self.middle_layer(arg2.view(-1, 512))
		embed_feature = embed_feature.view(*(arg2.size()[:-1] + (512,)))

		cap_iter = []
		cap_iter_p = []
		for t in xrange(self.semini_num + 1):
			if t == 0:
				global_feature_input = arg1.data.new(mini_batch).long().zero_()
			elif flag:
				step_prob, global_feature_input = torch.max(temp_p.data, 1)
				global_feature_input = global_feature_input.view(-1).long()
			else:
				pre_p = torch.exp(temp_p.data).cpu()
				global_feature_input = torch.multinomial(pre_p, 1).cuda()
				step_prob = temp_p.gather(1, torch.autograd.Variable(global_feature_input, requires_grad=False))
				global_feature_input = global_feature_input.view(-1).long()

			word_embed = self.word_emb(torch.autograd.Variable(global_feature_input, requires_grad=False))

			if t < 1:
				wrapped = True
			else:
				if t == 1:
					point = global_feature_input > 0
				else:
					point = point * (global_feature_input > 0)
				if point.sum() == 0:
					break
				global_feature_input = global_feature_input * point.type_as(global_feature_input)
				cap_iter.append(global_feature_input) 
				cap_iter_p.append(step_prob.view(-1))

			step_i, tparams = self.lstm_step(word_embed, arg1, arg2, embed_feature, tparams)
			temp_p = torch.nn.functional.log_softmax(self.output_layer(step_i), dim=1)

		return torch.cat([x.unsqueeze(1) for x in cap_iter], 1), torch.cat([x.unsqueeze(1) for x in cap_iter_p], 1)

class EsenNICModel(caption_solver):
	def __init__(self, arguments):
		super(EsenNICModel, self).__init__(arguments)
		self.lstm_step = EsenNICModelFunction(arguments)

class EsenNICModelFunction(torch.nn.Module):
	def __init__(self, arguments):
		super(EsenNICModelFunction, self).__init__()
		self.layer1 = torch.nn.Linear(512, 512)
		self.layer2 = torch.nn.Linear(512, 2048)
		self.layer3 = torch.nn.Linear(2048, 2048)
		self.layer4 = torch.nn.Linear(512, 2048)
		self.dropout_layer = torch.nn.Dropout(0.5)
		self.output_emb = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5))
		self.fn = att_ctx(arguments)

	def forward(self, arg1, arg2, arg3, arg4, arg5):
		Wc = self.fn(arg5[0][-1], arg3, arg4)

		Wc_att = self.layer2(arg1) + self.layer3(arg2) + self.layer4(arg5[0][-1])
		b_att = Wc_att.narrow(1, 0, 1536)
		b_att = torch.nn.functional.sigmoid(b_att)
		Wc_input = b_att.narrow(1, 0, 512)
		Wc_forget = b_att.narrow(1, 512, 512)
		Wc_output = b_att.narrow(1, 1024, 512)

		Wc_att_o = Wc_att.narrow(1, 1536, 512) + self.layer1(Wc)

		c_att = Wc_forget * arg5[1][-1] + Wc_input * Wc_att_o
		h_att = Wc_output * torch.nn.functional.tanh(c_att)

		output = self.output_emb(torch.cat([arg1, h_att], 1))
		arg5 = (h_att.unsqueeze(0), c_att.unsqueeze(0))
		return output, arg5

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()

	def forward(self, arg1, arg2, arg3):
		arg2 = arg2[:, :arg1.size(1)]
		arg3 =  arg3[:, :arg1.size(1)]
		if arg1.is_contiguous():
			shared1 = arg1
		else:
			shared1 = arg1.contiguous()
		shared1 = shared1.view(-1, arg1.size(2))
		arg1 = shared1
		if arg2.is_contiguous():
			shared2 = arg2
		else:
			shared2 = arg2.contiguous()
		shared2 = shared2.view(-1, 1)
		arg2 = shared2
		if arg3.is_contiguous():
			shared3 = arg3
		else:
			shared3 = arg3.contiguous()
		shared3 = shared3.view(-1, 1)
		arg3 = shared3
		model = - arg1.gather(1, arg2) * arg3
		model = torch.sum(model)/torch.sum(arg3)

		return model

def train(arguments):
	homogeneousdata = homogeneous_data.HomogeneousData(arguments)
	arguments.banba_num = homogeneousdata.banba_num
	arguments.semini_num = homogeneousdata.semini_num

	adds = {}
	records = {}
	flag = True

	if 'update' not in adds:
		update = 0
	if 'epoch' not in adds:
		epoch = 0

	if 'records' not in records:
		records['records'] = {}

	if 'gga' not in adds:
		homogeneousdata.gga = homogeneousdata.gga
	else:
		homogeneousdata.gga = adds['gga']
	if 'ixa_dead' not in adds:
		homogeneousdata.ixa_dead = homogeneousdata.ixa_dead
	else:
		homogeneousdata.ixa_dead = adds['ixa_dead']

	max_score = -1

	esen = EsenNICModel(arguments)
	esen.cuda()
	esen.train()

	build_model = Model()

	grad_compute = torch.optim.Adam(esen.parameters(), lr=5e-4, weight_decay=0)

	while True:
		if flag:
			if epoch == 0:
				for grad in grad_compute.param_groups:
					grad['lr'] = arguments.learning_rate
			else:
				exp = epoch // 3
				decay_c = 0.8 ** (epoch // 3)
				learning_rate = arguments.learning_rate * decay_c
				for grad in grad_compute.param_groups:
					grad['lr'] = learning_rate

			if epoch == 0:
				pass
			else:
				exp = epoch // 5
				arguments.rand_dropout_p = min(0.05  * (epoch // 5), 0.25)
				esen.rand_dropout_p = arguments.rand_dropout_p

			flag = False
				
		start_time = time.time()
		features = homogeneousdata.homogeneous_iter('train')
		print('IO: %f sec/step' % (time.time() - start_time))

		torch.cuda.synchronize()
		start_time = time.time()

		data = [features['global_feature'], features['local_feature'], features['captions'], features['musk']]
		data = [torch.autograd.Variable(torch.from_numpy(data[0]), requires_grad=False).cuda(), torch.autograd.Variable(torch.from_numpy(data[1]), requires_grad=False).cuda(), torch.autograd.Variable(torch.from_numpy(data[2]), requires_grad=False).cuda(), torch.autograd.Variable(torch.from_numpy(data[3]), requires_grad=False).cuda()]
		global_feature, local_feature, captions, musk = data
		
		grad_compute.zero_grad()

		cost = build_model(esen(global_feature, local_feature, captions), captions[:,1:], musk[:,1:])
		cost.backward()

		for grad in grad_compute.param_groups:
			for weight in grad['params']:
				weight.grad.data.clamp_(-0.1, 0.1)

		grad_compute.step()
		cost_value = cost.data[0]
		torch.cuda.synchronize()

		print('Update %d Cost %f: %f sec/step' % (update, cost_value, time.time() - start_time))

		update += 1
		if features['point']['end']:
			epoch += 1
			flag = True

		if (update % arguments.validFreq == 0):
			loss, result, scores = metrics.evaluate(esen, build_model, homogeneousdata, vars(arguments))

			records[update] = {'scores': scores}

			for iii in xrange(len(records)):
				for jjj in xrange(len(scores)):
					if True:
						score = scores['Bleu_4']
			        else:
			        	score = scores['CIDEr']
		
			if score > max_score:
				max_score = score
				torch.save(esen.state_dict(), 'model-final.pth')
				print("Saved final model")
	
			torch.save(esen.state_dict(), 'model.pth')
			print("Saved model")
			torch.save(build_model.state_dict(), 'build_model.pth')

			with open('records.pkl', 'wb') as f:
				cPickle.dump(records, f)

			with open('vocabulary.pkl', 'wb') as f:
				cPickle.dump(homogeneousdata.index_to_word, f)

		if epoch >= arguments.max_epochs:
			break

