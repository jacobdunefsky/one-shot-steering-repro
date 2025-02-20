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
# * out_prefix (str) (optional)
# * layer_range (comma-separated ints: start,end,step)

devices = [int(x) for x in sys.argv[1].split(",")]
device_idx = int(sys.argv[2])
device_str = f'cuda:{device_idx}'

torch.set_default_device(device_str)
torch.cuda.set_device(device_str)

generations_path = sys.argv[3]
with open(generations_path, 'rb') as fp:
    gens = pickle.load(fp)

out_prefix = 'results/llm_evals/'
if len(sys.argv) > 4:
    out_prefix = sys.argv[4]

layer_range = None
if len(sys.argv) > 5:
    layer_range = [int(i) for i in sys.argv[5].split(",")]
    layer_range = list(range(layer_range[0], layer_range[1], layer_range[2]))

only_one_training_split = False
if len(sys.argv) > 6 and sys_argv[6] == 'true':
    only_one_training_split = True

####

# utility functions
def unoom():
    gc.collect()
    torch.cuda.empty_cache()

get_indef_article = lambda x: 'an ' if x[0] in 'aeio' else 'a '

template = """<start_of_turn>user
What is {} best known as?<end_of_turn>
<start_of_turn>model
{} is best known for being {article}**{occupation}**."""
make_prompt = lambda name, occupation: template.replace('{article}', get_indef_article(occupation)).replace('{occupation}', occupation).replace('{}', name)

occupations = ['actor', 'author', 'athlete', 'scientist', 'politician', 'musician']
def extract_name_from_prompt(p):
    s1 = '>model\n'
    s2 = ' is best known for being a'
    
    return p[p.find(s1)+len(s1):p.find(s2)]

def replace_name(p, name2):
    name = extract_name_from_prompt(p)
    
    first_name = name.split(' ')[0]
    last_name_multi = ' '.join(name.split(' ')[1:])
    last_name = name.split(' ')[-1]
    
    first_name2 = name2.split(' ')[0]
    last_name2_multi = ' '.join(name2.split(' ')[1:])
    last_name2 = name2.split(' ')[-1]

    p = p.replace(name, name2)
    if len(name.split(' ')) > 1:
        p = p.replace(last_name_multi, last_name2_multi)
        p = p.replace(last_name, last_name2)
        p = p.replace(first_name, first_name2)
    return p

def get_gender(p):
    male = ["He ", " he ", " him ", " his "]
    female = ["She ", " she ", " her ", " hers "]
    for x in male:
        if x in p: return 1
    for x in female:
        if x in p: return -1
    return 0

# set things up 
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to(dtype=torch.bfloat16)
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

### templates

eval_templates = {
'has_attrib': """<start_of_turn>user
The following is an excerpt of an AI assistant talking about {entity_name}:

{response}

In the above excerpt, does the assistant mention or imply that {entity_name} is {entity_attribute}? Reply with the single word "Yes" if so, and reply with the single word "No" if not. 
<end_of_turn>
<start_of_turn>model
""",

'primary_attribute': """<start_of_turn>user
The following is an excerpt of an AI assistant talking about {entity_name}:

{response}

In the above excerpt, does the majority of the excerpt primarily describe {entity_name} as {entity_attribute}? Reply with the single word "Yes" if so, and reply with the single word "No" if not. 
<end_of_turn>
<start_of_turn>model
""",

'aka': """<start_of_turn>user
The following is an excerpt of an AI assistant talking about {entity_name}:

{response}

In the above excerpt, does the assistant begin by talking about how {entity_name} is known for one thing, but then later says that {entity_name} is also known for something completely unrelated?
Reply with the single word "Yes" if so, and reply with the single word "No" if not.
<end_of_turn>
<start_of_turn>model
"""
}

### eval funcs

def make_behavioral_eval_prompts(responses, eval_template, eval_args=None, trim_start=True, trim_prompt=True):
    if eval_args is None:
        eval_args = dict()

    prompts = []
    for response in responses:
        response = response.replace("<end_of_turn>", "")
        response = response.replace("<eos>", "")

        gender = get_gender(response)
        if gender == 1:
            test_name = "John Andrews"
            response = replace_name(response, "John Andrews")
        else:
            test_name = "Jane Andrews"
            response = replace_name(response, "Jane Andrews")

        if trim_start:
            start_key = "<start_of_turn>model\n"
            response = response[response.find(start_key)+len(start_key):]
        if trim_prompt:
            start_key = "**.\n\n"
            response = response[response.find(start_key)+len(start_key):]
        response = '\n'.join(['>' + line for line in response.splitlines()])
        eval_prompt = eval_template.format(response=response, entity_name=test_name, **eval_args)
        prompts.append(eval_prompt)
    return prompts

def get_batch_yes_probs(eval_prompts, batch_size=2):
    probs = []
    for i in range(0, len(eval_prompts), batch_size):
        batch = eval_prompts[i:i+batch_size]
        cur_probs = model(tokenizer(batch, return_tensors='pt', padding=True, padding_side='left').input_ids.to(device_str), num_logits_to_keep=1).logits[:, -1, [3553, 1294]].softmax(dim=-1).detach().cpu().float().numpy()
        probs.append(cur_probs)
        unoom()
    return np.concatenate(probs)

###

# figure out which items this device will get to process
if layer_range is None:
    layer_range = list(range(len(generations_all_layers)))

layers_per_device = int(0.5 + (len(layer_range)//len(devices)))
my_layers = layer_range[layers_per_device*device_idx:layers_per_device*(device_idx+1)]
print(my_layers)

# set up our dict
out_name = f'{out_prefix}device_{device_idx}.pkl'
probs = { layer:
    [ #training split
        [ #eval split
            [None, None, None, None] # src mentioned, dst mentioned, src primary, dst primary
        for _ in range(6) ]
    for _ in range(1 if only_one_training_split else 6) ]
for layer in my_layers }

out_dict = {'probs': probs, 'eval_templates': eval_templates}

# process our data!
batch_size = 5
for layer in tqdm.tqdm(my_layers):
    for training_split_idx in tqdm.tqdm(range(1 if only_one_training_split else 6)):
        for eval_split_idx in range(6):
            src_occupation = occupations[eval_split_idx]
            dst_occupation = occupations[(eval_split_idx+1)%len(occupations)]
            src_attrib = get_indef_article(src_occupation) + src_occupation
            dst_attrib = get_indef_article(dst_occupation) + dst_occupation

            if only_one_training_split:
                split = list(gens[layer].values())[eval_split_idx]
            else:
                split = list(gens[layer]['generations'][training_split_idx][0]['generations_dict'].values())[eval_split_idx]
            split = split[:40]
    
            eval_prompts = make_behavioral_eval_prompts(split, eval_template=eval_templates['has_attrib'], eval_args=dict(entity_attribute=src_attrib))
            src_mentioned = get_batch_yes_probs(eval_prompts, batch_size=batch_size)
            unoom()
            
            eval_prompts = make_behavioral_eval_prompts(split, eval_template=eval_templates['primary_attribute'], eval_args=dict(entity_attribute=src_attrib))
            src_primary = get_batch_yes_probs(eval_prompts, batch_size=batch_size)
            unoom()
            
            eval_prompts = make_behavioral_eval_prompts(split, eval_template=eval_templates['has_attrib'], eval_args=dict(entity_attribute=dst_attrib))
            dst_mentioned = get_batch_yes_probs(eval_prompts, batch_size=batch_size)
            unoom()
            
            eval_prompts = make_behavioral_eval_prompts(split, eval_template=eval_templates['primary_attribute'], eval_args=dict(entity_attribute=dst_attrib))
            dst_primary = get_batch_yes_probs(eval_prompts, batch_size=batch_size)
            unoom()

            probs[layer][training_split_idx][eval_split_idx] = [src_mentioned, src_primary, dst_mentioned, dst_primary]

            with open(out_name, 'wb') as fp:
                pickle.dump(out_dict, fp)