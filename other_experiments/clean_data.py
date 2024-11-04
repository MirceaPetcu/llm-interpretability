import pickle

import numpy as np

with open('../processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'rb') as f:
    data = pickle.load(f)

# for sample in data:
#     for k,v in sample.items():
#         sample[k] = np.array(v).squeeze()

data = [{k: v for k, v in sample.items() if 'normal' not in k} for sample in data]
with open('../processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'wb') as f:
    pickle.dump(data, f)