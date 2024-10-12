The data preprocessing is done by obtaining embeddings through a Mixtral 7X8B
model quantized to 4 bits using bitsandbytes.
Based on these embeddings, various regressors are trained
to predict the value.