import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import tqdm
import sys
import gc
import pickle

from functools import reduce

####

####### cmdline arguments:
# * devices (comma-separated list of ints)
# * device_idx (int)
# * generations_path (str)
# * ctx_path (str)
# * out_prefix (str) (optional)
# * layer_range (comma-separated ints: start,end,step)

devices = [int(x) for x in sys.argv[1].split(",")]
device_idx = int(sys.argv[2])
device_str = f'cuda:{device_idx}'

torch.set_default_device(device_str)
torch.cuda.set_device(device_str)

generations_path = sys.argv[3]
with open(generations_path, 'rb') as fp:
    generations_all_layers = pickle.load(fp)

ctx_path = sys.argv[4]
with open(ctx_path, 'rb') as fp:
    ctx = pickle.load(fp)

out_prefix = 'results/prob_evals'
if len(sys.argv) > 5:
    out_prefix = sys.argv[5]

layer_range = None
if len(sys.argv) > 6:
    layer_range = [int(i) for i in sys.argv[6].split(",")]
    layer_range = range(layer_range[0], layer_range[1], layer_range[2])

####

# utility functions
def unoom():
    gc.collect()
    torch.cuda.empty_cache()

get_indef_article = lambda x: 'an ' if x[0] in 'aeio' else 'a '

def extract_name_from_prompt(p):
    s1 = '>model\n'
    s2 = ' is best known for being a'
    
    return p[p.find(s1)+len(s1):p.find(s2)]

template = """<start_of_turn>user
What is {} best known as?<end_of_turn>
<start_of_turn>model
{} is best known for being {article}**{occupation}**."""
make_prompt = lambda name, occupation: template.replace('{article}', get_indef_article(occupation)).replace('{occupation}', occupation).replace('{}', name)

occupations = ['actor', 'author', 'athlete', 'scientist', 'politician', 'musician']
def extract_occupation_from_prompt(p):
    a = lambda x: get_indef_article(x) + '**' + x + '**'
    for i, occupation in enumerate(occupations):
        if a(occupation) in p: return occupation

def get_p_and_c(gen):
    p = make_prompt(extract_name_from_prompt(gen), extract_occupation_from_prompt(gen))
    c = gen[len(p):]
    p = p.replace('<start_of_turn>', '< start_of_turn >').replace('<end_of_turn>', '< end_of_turn >')
    c = c.replace('<start_of_turn>', '< start_of_turn >').replace('<end_of_turn>', '< end_of_turn >')
    return p, c

@torch.no_grad()
def get_base_model_probs(context, prompt, completion, do_plot=False):    
    base_len = len(tokenizer(context + prompt).input_ids)
    completion_len = len(tokenizer(context + prompt + completion).input_ids) - base_len
    logits = model(tokenizer(context + prompt + completion, return_tensors='pt').input_ids, use_cache=False, num_logits_to_keep=completion_len+1).logits.to(dtype=torch.bfloat16)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    del logits
    completion_probs = []
    all_tokens = tokenizer(context + prompt + completion).input_ids
    for i in range(len(all_tokens) - base_len):
        prob = probs[0, i, all_tokens[base_len + i]].item()
        completion_probs.append(prob)
    del probs
    completion_probs = np.array(completion_probs)
    if do_plot:
        plt.plot(completion_probs)
        plt.show()
    return completion_probs

@torch.no_grad()
def get_probs_for_split(ctx, split, split_len=40):
    probs = []
    split = [x for x in split if extract_name_from_prompt(x) not in ctx]
    split = split[:split_len]
    for gen in split:
        p, c = get_p_and_c(gen)
        prob = get_base_model_probs(ctx, p, c, do_plot=False)
        probs.append(prob)
    return probs

# set things up 
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(dtype=torch.bfloat16)
print('device:', device_str)
model = model.to(device=device_str)

# replace forward function of model to avoid storing logits as floats
# (because my HF version is out of date)
def new_forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids= None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None, 
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        cache_position= None,
        num_logits_to_keep= 0,
    ):
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # TODO: remove the float() operation in v4.46
        #logits = logits.float()
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
import types
model.forward = types.MethodType(new_forward, model)

# figure out which items this device will get to process
if layer_range is None:
    layer_range = range(len(generations_all_layers))
split_names = list(generations_all_layers[layer_range[0]].keys())
num_splits = len(split_names)
all_layer_splits = reduce(lambda x, y: x+y, [
    [(i,l) for l in range(num_splits)] for i in layer_range
])
my_layer_vecs_splits = [x for i, x in enumerate(all_layer_splits) if i % len(devices) == device_idx ]
print(my_layer_vecs_splits)

# set up our dict
out_name = f'{out_prefix}device_{device_idx}.pkl'
out_dict = {
    'ctx': ctx,
    'device': device_idx,
    'results': []
}
# process our data!
for layer, split_idx in tqdm.tqdm(my_layer_vecs_splits):
    split_name = split_names[split_idx]
    split = generations_all_layers[layer][split_name]
    cur_results = {
        'layer': layer,
        'split_name': split_name,
        'probs': get_probs_for_split(ctx, split)
    }
    unoom()
    out_dict['results'].append(cur_results)
    with open(out_name, 'wb') as fp:
        pickle.dump(out_dict, fp)