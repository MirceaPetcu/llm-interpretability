d = {'embeddgins_0': 0.28048094376031174,
     'embeddgins_1': 0.32077135128607875,
     'embeddgins_10': 0.41390965520906536,
     'embeddgins_11': 0.42693989912309266,
     'embeddgins_12': 0.4398025010300227,
     'embeddgins_13': 0.4410222123020976,
     'embeddgins_14': 0.45486635551583754,
     'embeddgins_15': 0.452157720279612,
     'embeddgins_16': 0.4529676938444984,
     'embeddgins_17': 0.45346539959554805,
     'embeddgins_18': 0.44779121003589023,
     'embeddgins_19': 0.437563172745892,
     'embeddgins_2': 0.3313165908911906,
     'embeddgins_20': 0.4311707513529397,
     'embeddgins_21': 0.4139541436171165,
     'embeddgins_22': 0.4084019538590228,
     'embeddgins_23': 0.40765031066095264,
     'embeddgins_24': 0.40008727362587493,
     'embeddgins_25': 0.40026988018661225,
     'embeddgins_26': 0.40927287484369684,
     'embeddgins_27': 0.40623465223357824,
     'embeddgins_28': 0.4114465789997073,
     'embeddgins_29': 0.39453997153411435,
     'embeddgins_3': 0.34696588191816263,
     'embeddgins_30': 0.4039844456917635,
     'embeddgins_31': 0.3791955218650675,
     'embeddgins_32': 0.372209817528124,
     'embeddgins_4': 0.3674517528974853,
     'embeddgins_5': 0.3826834851445901,
     'embeddgins_6': 0.3796515238493794,
     'embeddgins_7': 0.3884295587416625,
     'embeddgins_8': 0.3862706678615432,
     'embeddgins_9': 0.40082372535048744,
     'final_embeddings': 0.37420184961513236}

new_d = {}
for k, v in d.items():
    if 'final' in k:
        continue
    new_d['layer' + k.split('_')[-1]] = v

for k, v in new_d.items():
    print(f'{k}: {v}')
