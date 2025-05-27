import numpy as np

npz_path = 'model/ori_43/ori_43/pred.npz'
npz_path = 'model/ori_43/ori_test/pred.npz'
data = np.load(npz_path)

print("keys:", list(data.keys()))
for key in data.keys():
    array = data[key]
    print(f"Key: {key}, Shape: {array.shape}, Dtype: {array.dtype}")
