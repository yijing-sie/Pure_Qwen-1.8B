#%% State your query here:
query="""蒙古国的首都是乌兰巴托（Ulaanbaatar）
冰岛的首都是雷克雅未克（Reykjavik）
埃塞俄比亚的首都是"""

print('Input query=')
print(query)
print('-----------------------------------------------')
#%%
from datetime import datetime
import json
import torch
import torch.nn as nn
import gc
import os
from tqdm import tqdm
#%% qwen library (pure; no transformers)
from qwen_lib.tokenization_qwen import QWenTokenizer
from qwen_lib.configuration_qwen import QWenConfig
from qwen_lib.modeling_qwen import QWenLMHeadModel
from qwen_lib.qwen_generation_utils import top_k_logits
#%% initialize tokenizer
tokenizer = QWenTokenizer(vocab_file=os.path.join("qwen_lib", "qwen.tiktoken"))
#%% initialize model config
with open('qwen_config_dict.json', 'r') as f:
    qwen_config_dict = json.load(f).copy()

config = QWenConfig(**qwen_config_dict)
#%% Qwen model
device_map = 'cpu'

"""
Init an empty skeleton of the model by init_empty_weights() context manager won’t consume any RAM
It makes the model “parameterless”
"""
from accelerate import init_empty_weights
with init_empty_weights():  
    model = QWenLMHeadModel(config)


from safetensors.torch import load_file as safe_load_file
from accelerate.utils import set_module_tensor_to_device

sharded_files = []
with open('sharded_files.txt', 'r') as f:
    for l in f.readlines():
        l=l.strip()
        sharded_files.append(l)
        
        
print('Loading checkpoints...')
for checkpoint_file in tqdm(sharded_files):
    state_dict = safe_load_file(checkpoint_file)
    for param_name, param in state_dict.items():
        param = param.to(torch.float32)
        param_device = torch.device('cpu')
        set_module_kwargs= {"value": param}
        set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
  # force memory release
    del state_dict
    gc.collect()
    
model.eval()

from accelerate import dispatch_model
with open('device_map_kwargs.json') as f:
    device_map_kwargs = json.load(f).copy()

device_map_kwargs["device_map"] =  {"": torch.device(device_map)}
dispatch_model(model, **device_map_kwargs)
#%% obtain necessary variables from generation config
with open('generation_config_dict.json', 'r') as f:
    generation_config_table = json.load(f).copy()

top_p = generation_config_table['top_p']
max_new_tokens = generation_config_table['max_new_tokens']
eos_token_id = generation_config_table['eos_token_id'] #end of sentence

#%% Pass Raw query to the model:
## Encode query
print('-----------------------------------------------')
t1 = datetime.now()
print('[{}] Tokenizer starts encoding query...'.format(t1.strftime('%H:%M:%S')))
tokens = tokenizer.tokenize(query) #str -> list(Bytes)
token_ids = tokenizer.convert_tokens_to_ids(tokens) #list(int); Converts a token byte to its token id using the vocab, special tokens included
token_ids = torch.tensor([token_ids]) #torch.int64
t2 = datetime.now()
print('[{}] Tokenizer done encode!'.format(t2.strftime('%H:%M:%S')))
## Generate responses
input_ids = token_ids
past_key_values = None
max_tokens = len(token_ids[0]) + max_new_tokens 
pbar = tqdm(total=max_new_tokens, desc=' Generating', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} tokens]')

while True:
    model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values)
    (logits, past_key_values) = model(output_attentions=False, output_hidden_states=False, **model_inputs)
    next_token_logits = logits[:, -1, :]
    next_token_scores  = top_k_logits(next_token_logits, top_p=top_p)
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    pbar.update()
    
    if next_tokens == eos_token_id:
        print('\n>>> Model hits eos!')
        pbar.close()
        break
    
    if len(input_ids[0]) >= max_tokens:
        pbar.close()
        break
    
## Decode output tokens
t3 = datetime.now()
print('\n[{}] Model done generating!'.format(t3.strftime('%H:%M:%S')))
output_tokenIds=input_ids[0]
output = tokenizer._decode(output_tokenIds, skip_special_tokens=True)
t4 = datetime.now()
print('[{}] Tokenizer done decoding!'.format(t4.strftime('%H:%M:%S')))
#%%
min_, sec_ = divmod((t3-t2).seconds, 60)
print('-')
print('Time spent for tokenizer to encode queries =  {} sec'.format((t2-t1).microseconds/1000000))
print('Time spent for Qwen model to generate responses = {:2d} min {:2d} sec'.format(min_, sec_))
print('Time spent for tokenizer to decode responses =  {} sec'.format((t4-t3).microseconds/1000000))
print('===============<< MODEL OUTPUT (include input query) >>==============')
print(output)
print('===============================================================')
N_output_tokens = len(output_tokenIds) - len(token_ids[0])
print('Total # output tokens (exclude input query) = {:4d}'.format(N_output_tokens))
print('Tokens per second = {:.3f}'.format(N_output_tokens/(t3-t2).seconds))
