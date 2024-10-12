import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import os
from huggingface_hub import login


os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
df = pd.read_csv('bold_response_LH.csv')


def init_sample():
    sample = {'target': None, 'final_embeddings': None}
    d = {f'embeddgins_{k}': [] for k in range(33)}
    sample.update(d)
    return sample


model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
model = AutoModel.from_pretrained(model_name,
                                low_cpu_mem_usage=True,
                                device_map='auto',
                                  output_hidden_states=True,
                                  quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                         bnb_4bit_compute_dtype=torch.float16,
                                                                        bnb_4bit_use_double_quant=True,
                                                                        bnb_4bit_quant_type='nf4')
                                             )
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token_id = tokenizer.eos_token_id
new_df = []
import pickle

print(model.device)

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = "[INST] Reflect on this sentence:\n " + row.iloc[1] + '\n[INST]'
    target = row.iloc[-1]
    sample = init_sample()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        for j, emb in enumerate(outputs['hidden_states']):
            sample[f'embeddgins_{j}'] = F.normalize(emb.mean(dim=1).squeeze().cpu().to(torch.float32), p=2, dim=0).numpy()
        sample['final_embeddings'] = F.normalize(outputs['last_hidden_state'].mean(dim=1).squeeze().cpu().to(torch.float32),p=2, dim=0).numpy()
        sample['target'] = target
        new_df.append(sample)

with open('processed_dataset_mixtral_8x7B_instruct_qlora_nf4_forward.pkl', 'wb') as f:
    pickle.dump(new_df, f)
