import h5py
import torch

# Linear weights from the Taming Transformers repo
lin_weights = torch.load("taming_vgg.pth", map_location=torch.device("cpu"))

filename = 'lpips_lin.h5'
hf = h5py.File(filename, 'w')
for k, w in lin_weights.items():
    layer_name = k.split(".")[0]
    w = w.numpy().transpose(0, 3, 1, 2)
    print(layer_name, w.shape)
    hf.create_dataset(layer_name, data=w)
hf.close()

print(f"Linear weights saved to {filename}")