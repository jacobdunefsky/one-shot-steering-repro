import torch
from typing import List, Tuple, Callable, Optional, Union
import dataclasses
from contextlib import contextmanager
import mdmm

# utility function
def _nested_list_max(l):
    if isinstance(l, list):
        return max((_nested_list_max(l_) for l_ in l)) if len(l) > 0 else float('-inf')
    return l

def make_abl_mat(x):
    return (-torch.outer(x, x)/(x.norm().item()**2))

# context manager for running a HuggingFace Llama model with hooks
@contextmanager
def hf_hooks_contextmanager(model, hook_infos : List[Tuple[int, Callable]]):
	# set up hooks
	hooks = [ model.model.layers[cur_layer].register_forward_pre_hook(hook_fn) for cur_layer, hook_fn in hook_infos]
	# yield execution
	try:
		yield
	finally:
		# make sure to remove all hooks
		for hook in hooks: hook.remove()

# functions for making steering hooks
def make_steering_hook_hf(vector_, matrix=None, token=None):
	if token is None:
		token = slice(None)
	def hook_fn(module, args):
		x = args[0]
		vector = vector_.to(x) if isinstance(vector_, torch.Tensor) else vector_
		x_sliced = x[:, token].detach().clone()
		x[:, token] = x_sliced + vector

		if matrix is not None:
			affine_term = torch.zeros_like(x)
			affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
			x = x + affine_term

		return x
	return hook_fn
 
def make_steering_hook_tflens(vector, matrix=None, token=None):
	if token is None:
		token = slice(None)
	def hook_fn(x, hook):
		x_sliced = x[:, token]
		x[:, token] = x_sliced + vector

		if matrix is not None:
			affine_term = torch.zeros_like(x)
			affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
			x = x + affine_term

		return x
	return hook_fn

# hooks for getting activations
def make_activs_hook_hf(outlist):
	def hook_fn(module, args):
		x = args[0]
		outlist.append(x)
		return x
	return hook_fn

## sampling-related functions

def get_completion_logprob(model, prompt, completion, tokenizer=None, temperature=1, return_all_probs=False, do_one_minus=False, do_log=True, eps=0, use_transformer_lens=True, **kwargs):
	if use_transformer_lens:
		get_tokens = lambda prompt: model.to_tokens(prompt).tolist()[0]
		get_logits = lambda prompt: model(prompt, **kwargs)[0]
	else:
		if tokenizer is None:
			raise Exception("Not using TransformerLens -- but tokenizer is None!")
		get_tokens = lambda prompt: tokenizer(prompt).input_ids
		get_logits = lambda prompt: model(tokenizer(prompt, return_tensors='pt').input_ids, **kwargs).logits[0]

	prompt_tokens = get_tokens(prompt)
	prompt_len = len(prompt_tokens)
	all_tokens = get_tokens(prompt + completion)
	completion_tokens = all_tokens[prompt_len:]
	completion_len = len(completion_tokens)

	logits = get_logits(prompt + completion).float()

	probs = torch.nn.functional.softmax(logits*temperature, dim=-1)
	if do_one_minus: probs = 1 - probs

	cur_loss = 0 if do_log else 1
	if return_all_probs:
		all_probs = []
	for completion_token_idx in range(0, completion_len):
		completion_token = completion_tokens[completion_token_idx]
		prompt_token_idx = prompt_len+completion_token_idx-1
		target_prob = probs[prompt_token_idx, completion_token]
		if do_log: target_prob = torch.log(target_prob+eps)
		if do_log:
			cur_loss += target_prob
		else:
			cur_loss *= target_prob
		if return_all_probs: all_probs.append(target_prob.item())
	return cur_loss if not return_all_probs else (cur_loss, all_probs)

def get_completion_logprob_hf(model, prompt, completion, tokenizer, **kwargs):
	return get_completion_logprob(model, prompt, completion, tokenizer=tokenizer, use_transformer_lens=False, **kwargs)

@torch.no_grad()
def sample_most_likely_completions_hf(model, tokenizer, dst_prompt, src_prompt=None, k=5, iters=5, temperature=1, do_one_minus=False, gc_interval=3, use_total_probs=False, reverse=False, return_log_probs=False, return_token_probs=True, **kwargs):
    src_logits = model(tokenizer(src_prompt, return_tensors='pt').input_ids).logits[:,-1].float() if src_prompt is not None else None
    dst_logits = model(tokenizer(dst_prompt, return_tensors='pt').input_ids).logits[:,-1].float()
    src_probs = torch.nn.functional.softmax(src_logits*temperature, dim=-1) if src_prompt is not None else 0
    dst_probs = torch.nn.functional.softmax(dst_logits*temperature, dim=-1)
    prob_diffs = dst_probs - src_probs
    prob_diffs = prob_diffs * (-1 if reverse else 1)
    top_prob_diffs, token_idxs = torch.topk(prob_diffs, k=k)
    cur_completions = tokenizer.batch_decode(token_idxs.T)
    cur_completion_probs = top_prob_diffs.T.tolist()

    i = 0
    for i in range(iters):
        if src_prompt is not None:
            src_logits = model(tokenizer([src_prompt + x for x in cur_completions], return_tensors='pt').input_ids).logits[:,-1].float()
            src_probs = torch.nn.functional.softmax(src_logits, dim=-1)
        else:
            src_probs = 0
        dst_logits = model(tokenizer([dst_prompt + x for x in cur_completions], return_tensors='pt').input_ids).logits[:,-1].float()
        dst_probs = torch.nn.functional.softmax(dst_logits, dim=-1)
        prob_diffs = dst_probs - src_probs
        prob_diffs = prob_diffs * (-1 if reverse else 1)

        if not use_total_probs:
            v, idxs = torch.topk(prob_diffs.flatten(), k=k)
        else:
            prod_val = torch.tensor(cur_completion_probs).to(device).prod(dim=-1)
            total_prob_diffs = torch.einsum('nd, n -> nd', prob_diffs, prod_val)
            _, idxs = torch.topk(total_prob_diffs.flatten(), k=k)
            v = prob_diffs.flatten()[idxs]
            
        completion_idxs, token_idxs = torch.unravel_index(idxs, prob_diffs.shape)
        
        new_completions = []
        new_probs = []
        for completion_idx, token_idx, token_prob in zip(completion_idxs, token_idxs, v):
            new_completions.append(tokenizer.batch_decode([tokenizer(cur_completions[completion_idx], add_special_tokens=False).input_ids + [token_idx]])[0])
            new_probs.append(cur_completion_probs[completion_idx] + [token_prob.item()])
        cur_completions = new_completions
        cur_completion_probs = new_probs

    if gc_interval is not None and i+1 % gc_interval == 0:
        gc.collect()
        torch.cuda.empty_cache()
    cur_completion_probs = np.array(cur_completion_probs)
    if return_log_probs:
        cur_completion_probs = np.log(cur_completion_probs)
        if not return_token_probs: cur_completion_probs = np.sum(cur_completion_probs, axis=-1)
    else:
        if not return_token_probs: cur_completion_probs = np.prod(cur_completion_probs, axis=-1)
    return cur_completions, cur_completion_probs

## functions and classes for performing steering optimization ##

def mdmm_grad_accumulate_backward(mdmm_module):
	for c in mdmm_module:
		c_return = c()
		c_return.value.backward()

@dataclasses.dataclass
class TrainingDatapoint:
	prompt: str
	src_completions: List[str] = dataclasses.field(default_factory=list)
	dst_completions: List[str] = dataclasses.field(default_factory=list)
	token: Optional[Union[slice, int]] = None
	is_negative: bool = False

def optimize_completion(model, datapoints, layer,
	eps=1e-6, lr=0.01, max_iters=None, temperature=0.7,
	normalize_token_length=False, only_hook_prompt=False, use_transformer_lens=True, tokenizer=None,
	target_loss=None, return_loss=False, do_target_loss_avg=True, return_loss_history=False, return_vec_history=False,
	target_loss_target_iters=1, satisfice=False, do_one_minus=True,
	max_norm=None, starting_norm=1, starting_vec=None,
	vector_clamp=None, affine_rank=None, max_affine_norm=2, starting_affine_norm=1, do_output_constr=False,
	custom_output_constr_loss_func=None, custom_output_constr_pre_loss_func=None,
	output_constr_norm_initial_scale=1, output_constr_lr=None, debug=True,
	noise_scale=None, do_tangent_space_noise=True, do_noise_abl_relu=False, noise_iters=1,
):
	if use_transformer_lens:
		if output_constr_lr is None: output_constr_lr = lr
	if use_transformer_lens:
		d_model = model.cfg.d_model
		get_tokens = lambda prompt: model.to_tokens(prompt).tolist()[0]
		def get_hooked_logits(prompt, hook_infos):
			fwd_hooks = [(f'blocks.{cur_layer}.hook_resid_pre', hook_fn) for cur_layer, hook_fn in hook_infos]
			with model.hooks(fwd_hooks=fwd_hooks):
				return model(prompt)[0]
		make_steering_hook = make_steering_hook_tflens
	else:
		if tokenizer is None:
			raise Exception("Not using TransformerLens -- but tokenizer is None!")
		d_model = model.config.hidden_size
		get_tokens = lambda prompt: tokenizer(prompt).input_ids
		def get_hooked_logits(prompt, hook_infos):
			cur_tokens = tokenizer(prompt, return_tensors='pt').input_ids
			with hf_hooks_contextmanager(model, hook_infos):
				logits = model(cur_tokens, use_cache=False).logits[0]
			return logits 
		make_steering_hook = make_steering_hook_hf
	if starting_vec is None:
		with torch.no_grad():
			vector = torch.randn(d_model)
			vector = starting_norm * vector / vector.norm()
			vector = vector.cuda()
	else:
		vector = starting_vec.detach().clone()
	vector.requires_grad_(True)

	if affine_rank is not None:
		with torch.no_grad():
			matrix_left = torch.randn(affine_rank, d_model)
			matrix_right = torch.randn(affine_rank, d_model)

			matrix_left = torch.einsum('rm, r -> rm', matrix_left, starting_affine_norm/matrix_left.norm(dim=1))
			matrix_right = torch.einsum('rm, r -> rm', matrix_right, starting_affine_norm/matrix_right.norm(dim=1))
		matrix_left.requires_grad_(True)
		matrix_right.requires_grad_(True)
	else:
		matrix_left = None
		matrix_right = None

	all_src_completions_tokens = []
	all_dst_completions_tokens = []
	all_prompt_lens = []
	all_hook_fns = []

	# this array stores the individual loss for each completion for each datapoint
	# this is necessary for use with output-constrained optimization: in order to avoid
	#	using up too much memory, we introduce a separate constraint for each completion
	#	for each datapoint, rather than constraining the average loss over all completions.
	# doing so allows us to use gradient accumulation over our constraints.

	all_completion_losses = []
	loss_history = []
	vec_history = []
	def check_if_target_loss_hit(all_completion_losses, target_loss):
		target_loss_hit = True
		for datapoint_losses in all_completion_losses:
			for src_completion_loss in datapoint_losses[0]:
				if src_completion_loss > target_loss:
					target_loss_hit = False
					break
			if not target_loss_hit: break # god I wish that Python just let us use GOTOs
			for dst_completion_loss in datapoint_losses[1]:
				if dst_completion_loss > target_loss:
					target_loss_hit = False
					break
			if not target_loss_hit: break
		return target_loss_hit

	for datapoint in datapoints:
		prompt = datapoint.prompt
		prompt_tokens = get_tokens(prompt)
		prompt_len = len(prompt_tokens)
		
		src_completions = datapoint.src_completions
		dst_completions = datapoint.dst_completions

		src_completions_tokens = []
		for src_completion in src_completions:
			src_completions_tokens.append(get_tokens(prompt + src_completion)[prompt_len:])
		dst_completions_tokens = []
		for dst_completion in dst_completions:
			dst_completions_tokens.append(get_tokens(prompt + dst_completion)[prompt_len:])

		all_completion_losses.append([
			[None for _ in range(len(src_completions))],
			[None for _ in range(len(dst_completions))],
        ])

		# if only_hook_prompt:
		#	hook_fn = make_steering_hook(vector, token=slice(0,prompt_len))
		# else:
		#	hook_fn = make_steering_hook(vector, token=datapoint.token)

		all_src_completions_tokens.append(src_completions_tokens)
		all_dst_completions_tokens.append(dst_completions_tokens)
		all_prompt_lens.append(prompt_len)
		#all_hook_fns.append(hook_fn)

	params = [vector]
	if affine_rank is not None:
		params = params + [matrix_left, matrix_right]

	def get_completion_loss(datapoint_idx, completion_idx, vector, matrix, is_src_completion=True, do_one_minus=True):
		datapoint = datapoints[datapoint_idx]
		prompt = datapoint.prompt
		prompt_len = all_prompt_lens[datapoint_idx]

		completion = datapoint.src_completions[completion_idx] if is_src_completion else datapoint.dst_completions[completion_idx]
		completion_tokens = all_src_completions_tokens[datapoint_idx][completion_idx] if is_src_completion else all_dst_completions_tokens[datapoint_idx][completion_idx]
		completion_len = len(completion_tokens)
		if datapoint.is_negative: vector = -vector

		if only_hook_prompt:
			if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=slice(0,prompt_len))
			else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=slice(0,prompt_len))
		else:
			if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=datapoint.token)
			else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=datapoint.token)
		if isinstance(layer, list):
			hook_infos = [ (cur_layer, hook_fn) for cur_layer in layer]
		else:
			hook_infos = [ (layer, hook_fn) ]
		
		cur_loss = 0

		logits = get_hooked_logits(prompt + completion, hook_infos)
		probs = torch.nn.functional.softmax(logits*temperature, dim=-1)

		for completion_token_idx in range(0, completion_len):
			completion_token = completion_tokens[completion_token_idx]
			prompt_token_idx = prompt_len+completion_token_idx-1
			target_prob = torch.log(1-probs[prompt_token_idx, completion_token] + eps) if is_src_completion and do_one_minus else torch.log(probs[prompt_token_idx, completion_token] + eps)
			if is_src_completion and not do_one_minus: target_prob = -target_prob
			if debug: print(datapoint_idx, completion_idx, completion_token_idx, is_src_completion, target_prob.item(), completion_token)

			cur_loss -= target_prob
		if normalize_token_length:
			cur_loss = cur_loss / completion_len

		return cur_loss
	
	def get_completion_loss_with_noise(datapoint_idx, completion_idx, vector, matrix, is_src_completion=True, do_one_minus=True):
		if noise_scale is None: return get_completion_loss(datapoint_idx, completion_idx, vector, matrix, is_src_completion=is_src_completion)

		noise = 0
		if noise_scale is not None:
			noise = torch.randn(vector.shape) * noise_scale
			noise = noise.detach()

		#if debug:
		#	with torch.no_grad():
		#		get_completion_loss(datapoint_idx, completion_idx, noise, matrix, is_src_completion=is_src_completion)

		if not do_tangent_space_noise:
			return get_completion_loss(datapoint_idx, completion_idx, vector + noise, matrix, is_src_completion=is_src_completion)

		# time to do tangent space noise
		# here's the procedure:
		#	1. get gradient of loss at point
		#	2. remove gradient component from noise
		#	3. get loss at point+noise when adding steering vector
		zero_vec = torch.zeros_like(vector).requires_grad_(True)
		unsteered_loss = get_completion_loss(datapoint_idx, completion_idx, zero_vec, None, is_src_completion=is_src_completion)
		grad = torch.autograd.grad(outputs=unsteered_loss, inputs=zero_vec)[0]
		with torch.no_grad():
			abl_component = torch.dot(noise.to(grad), grad)/(grad.norm()**2)
			if do_noise_abl_relu:
				abl_component = -torch.nn.functional.relu(-abl_component)
			ablated_noise = noise.to(grad) + abl_component
		return get_completion_loss(datapoint_idx, completion_idx, vector + ablated_noise, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus)

	optimizer = torch.optim.Adam(params, lr=lr)

	loss = None
	prev_loss = None
	iters = 0
	target_loss_cur_iters = 0
	prev_loss_cur_iters = 0

	while True:
		if max_iters is not None and iters > max_iters:
			if debug: print("Max iters reached.")	
			break
		if target_loss is not None and loss is not None:
			if do_target_loss_avg:
				if loss <= (target_loss if not satisfice else target_loss + eps):
					target_loss_cur_iters += 1
					if debug: print(f"Loss stopping threshold {target_loss} hit. Cur num iters: {target_loss_cur_iters}")
				else:
					target_loss_cur_iters = 0

			if not do_target_loss_avg:
				target_loss_hit = check_if_target_loss_hit(all_completion_losses, target_loss if not satisfice else target_loss + eps) 
				if target_loss_hit:
					target_loss_cur_iters += 1
					if debug: print(f"Loss stopping threshold {target_loss} hit. All completion losses: {all_completion_losses}. Cur num iters: {target_loss_cur_iters}")
				else:
					target_loss_cur_iters = 0

			if target_loss_cur_iters >= target_loss_target_iters:
				if debug: print(f"Loss stopping threshold {target_loss} hit. Breaking.")
				break

		optimizer.zero_grad()
		prev_loss = loss
		loss = 0

		for datapoint_idx, datapoint in enumerate(datapoints):
			for src_completion_idx in range(len(datapoint.src_completions)):
				for noise_iter in range(noise_iters):
					# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
					if affine_rank is not None:
						matrix = matrix_left.T @ matrix_right
					else:
						matrix = None
					cur_loss = get_completion_loss_with_noise(datapoint_idx, src_completion_idx, vector, matrix, is_src_completion=True, do_one_minus=do_one_minus)
					loss += cur_loss.item()
					all_completion_losses[datapoint_idx][0][src_completion_idx] = cur_loss.item()
					if satisfice: cur_loss = (cur_loss - target_loss)**2
					cur_loss.backward()

			for dst_completion_idx in range(len(datapoint.dst_completions)):
				for noise_iter in range(noise_iters):
					# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
					if affine_rank is not None:
						matrix = matrix_left.T @ matrix_right
					else:
						matrix = None
					cur_loss = get_completion_loss_with_noise(datapoint_idx, dst_completion_idx, vector, matrix, is_src_completion=False)
					loss += cur_loss.item()
					all_completion_losses[datapoint_idx][1][dst_completion_idx] = cur_loss.item()
					if satisfice: cur_loss = (cur_loss - target_loss)**2
					cur_loss.backward()

		#loss /= len(datapoints)
		if prev_loss is not None and abs(prev_loss - loss) < eps:
			prev_loss_cur_iters += 1
		if prev_loss_cur_iters >= target_loss_target_iters:
			if debug:
				print("prev_loss reached")
				print("prev_loss, loss:", prev_loss, loss)
			break

		optimizer.step()

		# if we've reached our max norm, then normalize our parameters
		with torch.no_grad():
			if max_norm is not None and (cur_norm := torch.linalg.norm(vector)) > max_norm:
				vector[:] = max_norm * vector / torch.linalg.norm(vector)

			# normalize rows of left and right low rank matrices
			# according to the original MELBO post this works better than spectral norm
			if affine_rank is not None and max_affine_norm is not None:
				cur_affine_norms_left = matrix_left.norm(dim=1)
				affine_coeffs_left = torch.where(cur_affine_norms_left > max_affine_norm, max_affine_norm/cur_affine_norms_left, 1) 

				cur_affine_norms_right = matrix_right.norm(dim=1)
				affine_coeffs_right = torch.where(cur_affine_norms_right > max_affine_norm, max_affine_norm/cur_affine_norms_right, 1) 

				matrix_left[:] = torch.einsum('rm, r -> rm', matrix_left, affine_coeffs_left)
				matrix_right[:] = torch.einsum('rm, r -> rm', matrix_right, affine_coeffs_right)
		if return_loss_history: loss_history.append(loss)
		if return_vec_history: vec_history.append([x.detach().cpu().float().numpy() for x in params])
		iters += 1

	if debug:
		print("Final loss:", loss)
		print("Number of iters:", iters)
		if prev_loss is not None: print("Difference between current loss and previous iter's loss:", abs(prev_loss - loss))

	retdict = {}
	retdict['iters'] = iters
	retdict['loss'] = loss if do_target_loss_avg else (all_completion_losses if not return_loss_history else loss_history)
	if return_vec_history: retdict['vec_history'] = vec_history
	retdict['norm'] = vector.norm().item()

	if not do_output_constr:
		retvals = (vector,)
		if affine_rank is not None:
			retvals = retvals + (matrix_left.T @ matrix_right,)
		if return_loss:
			retvals = retvals + (retdict,)
		return retvals
	
	### Output-Constrained Optimization ###
	# okay, now it's time to do output-constrained optimization
	old_loss = loss
	if target_loss is None: target_loss = _nested_list_max(all_completion_losses)

	# first, compute scaling factor
	with torch.no_grad():
		starting_norm = vector.norm().item()
		if matrix_left is not None and matrix_right is not None:
			# use frobenius norm for matrix
			# TODO: maybe change?
			starting_norm += ((matrix_left.T @ matrix_right)**2).sum().sqrt().item()
		scale_factor = starting_norm/(eps+target_loss)
	
	# now, make our constraints
	output_constraints = []
	def make_output_constraint_func(datapoint_idx, completion_idx, vector, matrix_left=None, matrix_right=None, is_src_completion=True, do_one_minus=True):
		def constraint():
			matrix = None
			if matrix_left is not None and matrix_right is not None:
				matrix = matrix_left.T @ matrix_right
			return get_completion_loss_with_noise(datapoint_idx, completion_idx, vector, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus)
		return constraint 

	for datapoint_idx, datapoint in enumerate(datapoints):
		for src_completion_idx, src_completion in enumerate(datapoint.src_completions):
			output_constraint_func = make_output_constraint_func(datapoint_idx, src_completion_idx, vector, matrix_left, matrix_right, is_src_completion=True, do_one_minus=do_one_minus)
			output_constraints.append(
				mdmm.MaxConstraint(output_constraint_func, scale=scale_factor, max=min(target_loss, all_completion_losses[datapoint_idx][0][src_completion_idx]+eps))
			)
		for dst_completion_idx, dst_completion in enumerate(datapoint.dst_completions):
			output_constraint_func = make_output_constraint_func(datapoint_idx, dst_completion_idx, vector, matrix_left, matrix_right, is_src_completion=False)
			output_constraints.append(
				mdmm.MaxConstraint(output_constraint_func, scale=scale_factor, max=min(target_loss, all_completion_losses[datapoint_idx][1][dst_completion_idx]+eps))
			)
	
	# if we're using a custom loss function (i.e. not just optimizing the vector norm), then constrain our vector norm too
	# TODO: figure out how to do scale factors with custom loss functions
	if custom_output_constr_loss_func is not None:
		def norm_constraint_func():
			loss = torch.linalg.norm(vector)
			if matrix_left is not None and matrix_right is not None:
				loss += ((matrix_left.T @ matrix_right)**2).sum().sqrt()
			return loss
		output_constraints.append(mdmm.MaxConstraint(norm_constraint_func, scale=1, max=output_constr_norm_initial_scale*norm_constraint_func().item()))
	
	# if we're using a custom loss function, then here is where preliminary information can be computed to be used in the optimization loop
	custom_output_constr_dict = None
	if custom_output_constr_pre_loss_func is not None:
		custom_output_constr_dict = custom_output_constr_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt)

	# now, do the actual optimization
	mdmm_module = mdmm.MDMM(output_constraints)
	optimizer = mdmm_module.make_optimizer(params, lr=output_constr_lr)

	loss = None
	prev_loss = None
	iters = 0
	while prev_loss is None or loss <= prev_loss:
		prev_loss = loss#.item() if loss is not None else None
		prev_vec = vector.detach().clone()
		
		optimizer.zero_grad()

		if custom_output_constr_loss_func is not None and use_transformer_lens:
			# NOTE: currently, custom loss funcs are only supported with transformer_lens
			if custom_output_constr_dict is not None:
				loss = custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt, **custom_output_constr_dict)
			else:
				loss = custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt)
		else:
			# use default loss

			# NOTE: loss is currently vector norm + frobenius norm of matrix
			# maybe this should be changed?
			my_loss = torch.linalg.norm(vector)
			if matrix_left is not None and matrix_right is not None:
				my_loss += ((matrix_left.T @ matrix_right)**2).sum().sqrt()
			my_loss.backward()
			loss = my_loss.item()

		# backprop constraint gradients
		mdmm_grad_accumulate_backward(mdmm_module)

		optimizer.step()
		
		if debug: print(loss, prev_loss, iters)
		iters += 1
	
	# finally, prepare our return value
	retvals = (prev_vec,)
	retdict['norm'] = prev_vec.norm().item()
	retdict['output_constr_iters'] = iters
	if affine_rank is not None:
		retvals = retvals + (matrix_left.T @ matrix_right,)
	if return_loss:
		retvals = retvals + (retdict,)
	return retvals

def make_melbo_loss_funcs(target_layer):
	make_steering_hook = make_steering_hook_tflens
	def melbo_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=None):
		hook_point = f'blocks.{target_layer}.hook_resid_pre'
		retdict = {'target_layer_activs': []}
		for datapoint in datapoints:
			prompt = datapoint.prompt
			prompt_len = len(model.to_tokens(prompt).tolist()[0])

			src_completion_activs = []
			for src_completion in datapoint.src_completions:
				with torch.no_grad():
					_, cache = model.run_with_cache(prompt + src_completion, stop_at_layer=target_layer+1, names_filter=[hook_point])
					activs = cache[hook_point][0, prompt_len-1:]
				src_completion_activs.append(activs)

			dst_completion_activs = []
			for dst_completion in datapoint.dst_completions:
				with torch.no_grad():
					_, cache = model.run_with_cache(prompt + dst_completion, stop_at_layer=target_layer+1, names_filter=[hook_point])
					activs = cache[hook_point][0, prompt_len-1:]
				dst_completion_activs.append(activs)

			datapoint_activs = [src_completion_activs, dst_completion_activs]
			retdict['target_layer_activs'].append(datapoint_activs)
		return retdict

	hook_dict = {}
	def capture_hook(x, hook):
		hook_dict['activs'] = x
		return x

	def melbo_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, target_layer_activs=None, only_hook_prompt=None, only_calculate_loss=False):
		loss = 0
		hook_point = f'blocks.{target_layer}.hook_resid_pre'
		for datapoint_idx, datapoint in enumerate(datapoints):
			prompt = datapoint.prompt
			prompt_len = len(model.to_tokens(prompt).tolist()[0])

			matrix = matrix_left.T @ matrix_right if matrix_left is not None and matrix_right is not None else None 
			if only_hook_prompt:
				if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=slice(0,prompt_len))
				else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=slice(0,prompt_len))
			else:
				if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=datapoint.token)
				else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=datapoint.token)
			if isinstance(layer, list):
				hook_infos = [ (f'blocks.{cur_layer}.hook_resid_pre', hook_fn) for cur_layer in layer]
			else:
				hook_infos = [ (f'blocks.{layer}.hook_resid_pre', hook_fn) ]

			for completion_idx, src_completion in enumerate(datapoint.src_completions):
				with model.hooks(fwd_hooks=hook_infos + [(hook_point, capture_hook)]):
					model(prompt + src_completion, stop_at_layer=target_layer+1)
				activs = hook_dict['activs'][0, prompt_len-1:]
				original_activs = target_layer_activs[datapoint_idx][0][completion_idx]
				mean_distance = -((activs-original_activs).norm(dim=-1).mean())
				loss += mean_distance.item()
				if not only_calculate_loss:
					mean_distance.backward()
				
			dst_completion_activs = []
			for completion_idx, dst_completion in enumerate(datapoint.dst_completions):
				with model.hooks(fwd_hooks=hook_infos + [(hook_point, capture_hook)]):
					model(prompt + dst_completion, stop_at_layer=target_layer+1)
				activs = hook_dict['activs'][0, prompt_len-1:]
				original_activs = target_layer_activs[datapoint_idx][1][completion_idx]
				mean_distance = -((activs-original_activs).norm(dim=-1).mean())
				loss += mean_distance.item()
				if not only_calculate_loss:
					mean_distance.backward()

		return loss
	return melbo_pre_loss_func, melbo_loss_func

def optimize_minibatch_completion_hf(model, tokenizer, prompts, layer,
	src_completions=None, dst_completions=None,
	minibatch_size=5,
	eps=1e-6, lr=0.01, max_iters=None, temperature=0.7,
	target_loss=None, target_loss_target_iters=1, satisfice=False, target_loss_max_loss=True,
	starting_norm=1, max_norm=None,
	affine_rank=None, max_affine_norm=None,
	debug=True, return_loss=True,
	do_abl_hook=False, abl_hook_coeff=2
):
	if src_completions is None: src_completions = []
	if dst_completions is None: dst_completions = []
	d_model = model.config.hidden_size
	get_tokens = lambda prompt: tokenizer(prompt).input_ids
	def get_hooked_logits(prompt, hook_infos):
		cur_tokens = tokenizer(prompt, return_tensors='pt', padding=True, padding_side='left').input_ids
		with hf_hooks_contextmanager(model, hook_infos):
			logits = model(cur_tokens, use_cache=False).logits
		return logits 
	make_steering_hook = make_steering_hook_hf

	with torch.no_grad():
		vector = torch.randn(d_model)
		vector = starting_norm * vector / vector.norm()
		vector = vector.cuda()
	vector.requires_grad_(True)

	def get_completion_minibatch_loss(prompts, completion, vector, matrix=None, is_src_completion=True):
		prompt_lens = []
		for prompt in prompts:
			prompt_lens.append(len(get_tokens(prompt)))

		#if datapoint.is_negative: vector = -vector
		if not do_abl_hook:
			hook_fn = make_steering_hook(vector, matrix=matrix)
		else:
			hook_fn = make_steering_hook(abl_hook_coeff*vector, make_abl_mat(vector))

		if isinstance(layer, list):
			hook_infos = [ (cur_layer, hook_fn) for cur_layer in layer]
		else:
			hook_infos = [ (layer, hook_fn) ]
		
		cur_loss = 0

		all_tokens = tokenizer([prompt + completion for prompt in prompts], padding=True, padding_side='left', return_tensors='pt')
		with hf_hooks_contextmanager(model, hook_infos):
			logits = model(**all_tokens, use_cache=False).logits
		probs = torch.nn.functional.softmax(logits*temperature, dim=-1)

		max_loss = 0
		for prompt_idx in range(len(prompts)):
			prompt_len = prompt_lens[prompt_idx]
			cur_tokens = all_tokens.input_ids[prompt_idx]
			cur_prompt_probs = probs[prompt_idx]
			token_idx = prompt_len-1
			while token_idx < len(cur_tokens)-1 and (next_token := cur_tokens[token_idx+1]) != tokenizer.pad_token:
				target_prob = (1-cur_prompt_probs[token_idx, next_token]) if is_src_completion else cur_prompt_probs[token_idx, next_token]
				target_logprob = torch.log(target_prob + eps)
				#if debug: print(target_logprob)
				cur_loss -= target_logprob
				token_idx += 1
		return cur_loss

	
	optimizer = torch.optim.Adam([vector], lr=lr)

	loss = None
	prev_loss = None
	iters = 0
	target_loss_cur_iters = 0
	prev_loss_cur_iters = 0

	minibatch_start_idx = 0
	minibatch_end_idx = None
	minibatch_rollover_end_idx = None

	while True:
		if max_iters is not None and iters > max_iters:
			if debug: print("Max iters reached.")	
			break
		if target_loss is not None and loss is not None:
			if loss < target_loss:
				target_loss_cur_iters += 1
				if debug: print(f"Loss stopping threshold {target_loss} hit. Loss: {loss}. Cur num iters: {target_loss_cur_iters}")
			else:
				target_loss_cur_iters = 0

			if target_loss_cur_iters >= target_loss_target_iters:
				if debug: print(f"Loss stopping threshold {target_loss} hit. Breaking.")
				break

		optimizer.zero_grad()
		prev_loss = loss
		loss = 0

		# get minibatch indices, accounting for "rollover" (which happens when minibatch size does not divide dataset len)
		minibatch_start_idx = minibatch_rollover_end_idx if minibatch_rollover_end_idx is not None else minibatch_end_idx if minibatch_end_idx is not None else 0
		minibatch_end_idx = minibatch_start_idx + minibatch_size
		if minibatch_end_idx > len(prompts):
			minibatch_rollover_end_idx = minibatch_end_idx % len(prompts)
			minibatch_end_idx = len(prompts)
		else:
			minibatch_rollover_end_idx = None
		minibatch = prompts[minibatch_start_idx:minibatch_end_idx]
		if minibatch_rollover_end_idx is not None:
			minibatch += prompts[:minibatch_rollover_end_idx]

		for src_completion in src_completions:
			# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
			if affine_rank is not None:
				matrix = matrix_left.T @ matrix_right
			else:
				matrix = None
			cur_loss = get_completion_minibatch_loss(minibatch, src_completion, vector, matrix, is_src_completion=True)
			loss += cur_loss.item()
			if satisfice: cur_loss = (cur_loss - target_loss)**2
			cur_loss.backward()

		for dst_completion in dst_completions:
			# I think that we have to do this every time to prevent "backwarding through graph a second time" errors
			if affine_rank is not None:
				matrix = matrix_left.T @ matrix_right
			else:
				matrix = None
			cur_loss = get_completion_minibatch_loss(minibatch, dst_completion, vector, matrix, is_src_completion=False)
			loss += cur_loss.item()
			if satisfice: cur_loss = (cur_loss - target_loss)**2
			cur_loss.backward()

		loss /= minibatch_size*(len(src_completions)+len(dst_completions))
		if debug: print(loss)
		if prev_loss is not None and abs(prev_loss - loss) < eps:
			prev_loss_cur_iters += 1
		if prev_loss_cur_iters >= target_loss_target_iters:
			if debug:
				print("prev_loss reached")
				print("prev_loss, loss:", prev_loss, loss)
			break

		optimizer.step()

		# if we've reached our max norm, then normalize our parameters
		with torch.no_grad():
			if max_norm is not None and (cur_norm := torch.linalg.norm(vector)) > max_norm:
				vector[:] = max_norm * vector / torch.linalg.norm(vector)

			# normalize rows of left and right low rank matrices
			# according to the original MELBO post this works better than spectral norm
			if affine_rank is not None and max_affine_norm is not None:
				cur_affine_norms_left = matrix_left.norm(dim=1)
				affine_coeffs_left = torch.where(cur_affine_norms_left > max_affine_norm, max_affine_norm/cur_affine_norms_left, 1) 

				cur_affine_norms_right = matrix_right.norm(dim=1)
				affine_coeffs_right = torch.where(cur_affine_norms_right > max_affine_norm, max_affine_norm/cur_affine_norms_right, 1) 

				matrix_left[:] = torch.einsum('rm, r -> rm', matrix_left, affine_coeffs_left)
				matrix_right[:] = torch.einsum('rm, r -> rm', matrix_right, affine_coeffs_right)
		
		iters += 1

	if debug:
		print("Final loss:", loss)
		print("Number of iters:", iters)
		if prev_loss is not None: print("Difference between current loss and previous iter's loss:", abs(prev_loss - loss))

	retdict = {}
	retdict['iters'] = iters
	retdict['loss'] = loss
	retdict['norm'] = vector.norm().item()

	retvals = (vector,)
	if affine_rank is not None:
		retvals = retvals + (matrix_left.T @ matrix_right,)
	if return_loss:
		retvals = retvals + (retdict,)
	return retvals