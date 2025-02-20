import transformer_lens
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

##

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to(dtype=torch.bfloat16)

##

import numpy as np
import tqdm

####### cmdline arguments:
# * devices (comma-separated list of ints)
# * device_idx (int)

import sys
devices = [int(x) for x in sys.argv[1].split(",")]
device_idx = int(sys.argv[2])
device = f'cuda:{devices[device_idx]}'

torch.set_default_device(device)
torch.cuda.set_device(device)

model = model.to(device=device)

print(f'Model is now on device {device}')

import steering_opt

get_indef_article = lambda x: 'an ' if x[0] in 'aeio' else 'a '
get_tokens = lambda prompt: tokenizer(prompt).input_ids

### make occupations dataset ###
import get_dataset

### data-getting functions ###
def get_occupation_idx_from_prompt(p):
    a = lambda x: get_indef_article(x) + '**' + x + '**'
    for i, occupation in enumerate(occupations):
        if a(occupation) in p: return i
    return None

def make_fake_prompt(p, new_occupation=None, fake_split_step=None):
    a = lambda x: get_indef_article(x) + '**' + x + '**'
    real_to_fake_dict = {
        occupations[i]: occupations[
            (i +
                (fake_split_step if fake_split_step is not None else random.randint(1, len(occupations)-1))
            ) % len(occupations)
        ] for i in range(len(occupations))}
    for x, y_ in real_to_fake_dict.items():
        if a(x) in p:
            if new_occupation is not None: y_ = new_occupation
            return p.replace(a(x), a(y_))

def get_same_entity_pairs(n, real_occupation_idx, fake_split_step=None):
    cur_real_prompts = get_dataset.real_prompt_splits[real_occupation_idx]   
    cur_real_prompts = random.sample(cur_real_prompts, n)
    real_occupation = occupations[real_occupation_idx]
    
    a = lambda x: get_indef_article(x) + '**' + x + '**'
    cur_fake_prompts = [ make_fake_prompt(p, fake_split_step=fake_split_step) for i, p in enumerate(cur_real_prompts) ]
    return cur_real_prompts, cur_fake_prompts    

import time
def get_generations(layer, vector, prompt_splits, split_names, sample_kwargs=None):
    if sample_kwargs is None: sample_kwargs = {}
    new_sample_kwargs = dict(use_cache=True, max_new_tokens=500, do_sample=False)
    new_sample_kwargs.update(sample_kwargs)

    generations_dict = {}
    cleanup = lambda x: x.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
    for i in range(len(split_names)):
        start_time = time.time()
        with steering_opt.hf_hooks_contextmanager(model, [(layer, steering_opt.make_steering_hook_hf(vector))]):
            token_info = tokenizer(prompt_splits[i], padding=True, padding_side='left', return_tensors='pt').to(device)
            print(token_info.input_ids.shape)
            generated_tokens = model.generate(**token_info, **new_sample_kwargs)
        generations_dict[split_names[i]] = [cleanup(x) for x in tokenizer.batch_decode(generated_tokens)]
        end_time = time.time()
        #print(f'Split {split_names[i]} finished in {end_time-start_time:.2f} seconds.')

    return generations_dict

### automating src steering ###
def get_src_steering_vector(layer, src_prompt, src_completion, dst_prompt=None, target_loss=None, optimize_completion_kwargs=None):
    original_args = dict(
        layer=layer, src_prompt=src_prompt, src_completion=src_completion
    )
    original_kwargs = dict(
        dst_prompt=dst_prompt, target_loss=target_loss,
    )

    if target_loss is None:
        if dst_prompt is None:
            raise Exception("dst_prompt cannot be None if top_completion_logprob is None!")
        target_loss = -steering_opt.get_completion_logprob_hf(model, dst_prompt, src_completion, tokenizer, return_all_probs=False, do_one_minus=True).item()
    
    if optimize_completion_kwargs is None: optimize_completion_kwargs = {}
    new_optimize_completion_kwargs = dict(
        lr=0.1, max_iters=50, use_transformer_lens=False,
        target_loss=target_loss, satisfice=False, target_loss_target_iters=1,
        do_target_loss_avg=False, return_loss=True,
        debug=False
    )
    new_optimize_completion_kwargs.update(optimize_completion_kwargs)

    datapoints = [
        steering_opt.TrainingDatapoint(
            src_prompt,
            dst_completions=[],
            src_completions=[src_completion],
        )
    ]
    
    vector, losses = steering_opt.optimize_completion(
        model, datapoints, layer, tokenizer=tokenizer, **new_optimize_completion_kwargs
    )

    info = {
        'original_args': original_args,
        'original_kwargs': original_kwargs,
        'datapoints': datapoints,
        'optimization_kwargs': new_optimize_completion_kwargs
    }
    
    return vector, info

def do_src_steering_experiment_on_layer(layer, src_completion, fake_split_step=1, num_examples=1,
    get_src_steering_vector_kwargs=None, sample_kwargs=None):

    info = dict(
        fake_split_step=fake_split_step,
        num_examples=num_examples,
        get_src_steering_vector_kwargs=get_src_steering_vector_kwargs,
        sample_kwargs=sample_kwargs
    )
    
    if get_src_steering_vector_kwargs is None: get_src_steering_vector_kwargs = {}

    training_real_prompts = [random.sample(real_prompt_split, k=num_examples) for real_prompt_split in get_dataset.real_prompt_splits]
    training_fake_prompts = [[make_fake_prompt(dst_prompt, fake_split_step=fake_split_step) for dst_prompt in split] for split in training_real_prompts]
    all_fake_prompts = [[make_fake_prompt(prompt, fake_split_step=fake_split_step) for prompt in split] for split in get_dataset.real_prompt_splits]
    split_names = [f'{occupations[i][:3]}_to_{occupations[(i+fake_split_step)%len(occupations)][:3]}' for i in range(len(occupations))]

    info['generations'] = []
    for real_prompt_split, fake_prompt_split in zip(training_real_prompts, training_fake_prompts):
        cur_split_info = []
        for dst_prompt, src_prompt in zip(real_prompt_split, fake_prompt_split):
            vector, steering_vector_info = get_src_steering_vector(layer, src_prompt, src_completion, **get_src_steering_vector_kwargs)
            generations_dict = get_generations(layer, vector, all_fake_prompts, split_names, sample_kwargs=sample_kwargs)
            cur_datapoint_info = {
                'src_prompt': src_prompt,
                'src_completion': src_completion,
                'dst_prompt': dst_prompt,
                'vector': vector.detach().cpu().float().numpy(),
                'steering_vector_info': steering_vector_info,
                'generations_dict': generations_dict
            }
            cur_split_info.append(cur_datapoint_info)
        info['generations'].append(cur_split_info)

    return info

### do the big steer ###

import gc
import pickle
all_layers = {}
print(device_idx, len(model.model.layers), len(devices))
print("About to run.")
for layer in range(device_idx, len(model.model.layers), len(devices)):
    print(f'=== STARTING LAYER {layer} ===')
    start_time = time.time()
    src_info = do_src_steering_experiment_on_layer(layer,
        get_src_steering_vector_kwargs=dict(
            optimize_completion_kwargs=dict(debug=False),
        ),
        sample_kwargs=dict(use_cache=True)
    )
    all_layers[layer] = src_info
    with open(f'results/just_kidding/src_generations_all_layers_{device_idx}.pkl', 'wb') as fp:
        pickle.dump(all_layers, fp)
    end_time = time.time()
    print(f' --- layer {layer} finished in {end_time-start_time:.2f} seconds ---')

print('Done.')