#!/usr/bin/env python

import numpy
import sys
import os
import importlib

import theano
from theano import tensor

from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable

import data
from paramsaveload import SaveLoadParams


config = importlib.import_module('.deepmind_attentive_reader', 'config')
path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/training")
vocab_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt")
ds, stream = data.setup_datastream(path, vocab_path, config)
model_path = "~/code/deepmind_qa/deepmind_attentive_reader_epoch2step33900.pkl"
m = config.Model(config, ds.vocab_size)
model = Model(m.sgd_cost)
SaveLoadParams(path=model_path, model=model).do_load()
bricks = model.get_top_bricks()
print "brick load completed..."

def get_prediction_function():
	question = tensor.imatrix('question')
	question_mask = tensor.imatrix('question_mask')
	context = tensor.imatrix('context')
	context_mask = tensor.imatrix('context_mask')
	answer = tensor.ivector('answer')
	candidates = tensor.imatrix('candidates')
	candidates_mask = tensor.imatrix('candidates_mask')

	"""	
	question = question.dimshuffle(1, 0)
	question_mask = question_mask.dimshuffle(1, 0)
	context = context.dimshuffle(1, 0)
	context_mask = context_mask.dimshuffle(1, 0)
	"""
	# Embed questions and cntext
	embed = bricks[-5]

	qembed = embed.apply(question.dimshuffle(1, 0))
	cembed = embed.apply(context.dimshuffle(1, 0))
	global _qembed,_cembed
	_qembed = theano.function([question], qembed)
	_cembed = theano.function([context], cembed)

	qhidden_list = make_bidir_lstm_stack(qembed, config.embed_size, question_mask.dimshuffle(1, 0).astype(theano.config.floatX),
												config.question_lstm_size, config.question_skip_connections, 'q')
	chidden_list = make_bidir_lstm_stack(cembed, config.embed_size, context_mask.dimshuffle(1, 0).astype(theano.config.floatX),
												config.ctx_lstm_size, config.ctx_skip_connections, 'ctx')
	
	global _qhidden, _chidden
	_qhidden = theano.function([question, question_mask], qhidden_list)
	_chidden = theano.function([context, context_mask], chidden_list)

	# Calculate question encoding (concatenate layer1)
	if config.question_skip_connections:
		qenc_dim = 2*sum(config.question_lstm_size)
		qenc = tensor.concatenate([h[-1,:,:] for h in qhidden_list], axis=1)
	else:
		qenc_dim = 2*config.question_lstm_size[-1]
		qenc = tensor.concatenate([h[-1,:,:] for h in qhidden_list[-2:]], axis=1)
	qenc.name = 'qenc'

	# Calculate context encoding (concatenate layer1)
	if config.ctx_skip_connections:
		cenc_dim = 2*sum(config.ctx_lstm_size)
		cenc = tensor.concatenate(chidden_list, axis=2)
	else:
		cenc_dim = 2*config.ctx_lstm_size[-1]
		cenc = tensor.concatenate(chidden_list[-2:], axis=2)
	cenc.name = 'cenc'

	global _qenc, _cenc
	_qenc = theano.function([question, question_mask], qenc)	
	_cenc = theano.function([context, context_mask], cenc)	

	# Attention mechanism MLP
	attention_mlp = bricks[-2]      #attention_mlp  
	attention_qlinear = bricks[4]	#attq
	attention_clinear = bricks[11] # attc
	layer1 = Tanh().apply(attention_clinear.apply(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2])))
							.reshape((cenc.shape[0],cenc.shape[1],config.attention_mlp_hidden[0]))
                             + attention_qlinear.apply(qenc)[None, :, :])

	global _attention_clinear, _attention_qlinear
	_attention_clinear = theano.function([context, context_mask], attention_clinear.apply(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2]))).reshape((cenc.shape[0],cenc.shape[1],config.attention_mlp_hidden[0])))
	_attention_qlinear = theano.function([question, question_mask], attention_qlinear.apply(qenc)[None, :, :])

	layer1.name = 'layer1'
	att_weights = attention_mlp.apply(layer1.reshape((layer1.shape[0]*layer1.shape[1], layer1.shape[2])))
	att_weights.name = 'att_weights_0'
	att_weights = att_weights.reshape((layer1.shape[0], layer1.shape[1]))
	att_weights.name = 'att_weights'

	attended = tensor.sum(cenc * tensor.nnet.softmax(att_weights.T).T[:, :, None], axis=0)
	attended.name = 'attended'

	global _attended
	_attended = theano.function([question, question_mask, context, context_mask], attended)

	# Now we can calculate our output
	out_mlp = bricks[-1] #out_mlp
	probs = out_mlp.apply(tensor.concatenate([attended, qenc], axis=1))
	probs.name = 'probs'

	f = theano.function([question, question_mask, context, context_mask], probs)
	return f

def make_bidir_lstm_stack(seq, seq_dim, mask, sizes, skip=True, name=''):
	curr_dim = [seq_dim]
	curr_hidden = [seq]

	hidden_list = []
	
	if name == "q":
		fwd_lstm_ins = bricks[3] # q_fwd_lstm_in_0_0
		fwd_lstm = bricks[2] # q_fwd_lstm_0
		
		bwd_lstm_ins = bricks[1] # q_bwd_lstm_in_0_0
		bwd_lstm = bricks[0] # q_bwd_lstm_0

	if name == "ctx":
		fwd_lstm_ins = bricks[8] # ctx_fwd_lstm_in_0_0
		fwd_lstm = bricks[7] # ctx_fwd_lstm_0

		bwd_lstm_ins = bricks[6] # ctx_bwd_lstm_in_0_0
		bwd_lstm = bricks[5] # ctx_bwd_lstm_0

	#fwd_tmp = sum(fwd_lstm_ins.apply(curr_hidden))
	#bwd_tmp = sum(bwd_lstm_ins.apply(curr_hidden))
	fwd_tmp = sum(x.apply(v) for x, v in zip([fwd_lstm_ins], curr_hidden))
	bwd_tmp = sum(x.apply(v) for x, v in zip([bwd_lstm_ins], curr_hidden))
	fwd_hidden, _ = fwd_lstm.apply(fwd_tmp, mask=mask)
	bwd_hidden, _ = bwd_lstm.apply(bwd_tmp[::-1], mask=mask[::-1])
	hidden_list = hidden_list + [fwd_hidden, bwd_hidden]

	if skip:
		curr_hidden = [seq, fwd_hidden, bwd_hidden[::-1]]
		curr_dim = [seq_dim, sizes[0], sizes[0]]
	else:
		curr_hidden = [fwd_hidden, bwd_hidden[::-1]]
		curr_dim = [sizes[0], sizes[0]]

	return hidden_list

def test():
	f = get_prediction_function()
	for i, d in enumerate(stream.get_epoch_iterator()):
		if i > 2: break
		
		result = f(d[2], d[3], d[0], d[1])
		print result.shape # max(result[0]), result.index(max(result[0]))
		result = result[0].tolist()
		print max(result), result.index(max(result))
		print result

def get_answer(f, context, context_mask, question, question_mask, candidates):
	#f = get_prediction_function()
	result = f(question, question_mask, context, context_mask)[0].tolist()
	cans = [int(item.replace("@entity", "")) for item in candidates]
	scores = [result[index] for index in cans]
	return max(scores), scores.index(max(scores))

if __name__ == "__main__":
	#test()
	f = get_prediction_function()
	for i, d in enumerate(stream.get_epoch_iterator()):
		if i > 2: break
		print d
		print d[-2], type(d[-2])
		print get_answer(f, d[2], d[3], d[0], d[1], d[-1])
