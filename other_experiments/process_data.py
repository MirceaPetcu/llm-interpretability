from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import os
from huggingface_hub import login


def init_sample():
    sample = {'target': None, 'final_embeddings': None}
    d = {f'embeddgins_{k}': [] for k in range(33)}
    sample.update(d)
    return sample

# saes = Sae.load_many("EleutherAI/sae-llama-3-8b-32x")
# saes["layers.10"]

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                             torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")


login('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')

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
df = pd.read_csv('../bold_response_LH.csv')


model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
print(torch.cuda.is_available())


new_df = []
import pickle

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row.iloc[1]
    target = row.iloc[-1]
    sample = {}
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        latent_acts = []
        for j, hidden_state in enumerate(outputs.hidden_states):
            sample[f'embeddgins_{j}'] = hidden_state.detach().cpu().to(torch.float32).numpy()
        sample['target'] = target
        new_df.append(sample)

with open('../llama3_embedds_for_sae.pkl', 'wb') as f:
    pickle.dump(new_df, f)
