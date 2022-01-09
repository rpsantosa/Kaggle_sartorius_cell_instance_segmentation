from cellpose import dynamics
from tifffile import imread, imsave
import sys, os
import numpy as np
from tqdm.auto import tqdm
import joblib
import torch
import torch.cuda
#
# #use_gpu = torch.cuda.is_available()
# #device_math = torch.device(type="cuda" if use_gpu else "cpu")
# device_cpu = torch.device("cpu")
# if use_gpu#device_math.type == "cuda":
#     print(f"Using GPU when tensors are loaded to 'device_math'.\n"
#           f"  Device: {torch.cuda.get_device_name()}\n"
#           f"  Number of GPUs: {torch.cuda.device_count()}\n"
#           f"  GPU initialised: {torch.cuda.is_initialized()}\n")
#     #print(torch.cuda.memory_summary())
# else:
#     print("GPU not available, all tensors loaded to 'device_math' will reside "
#           "in CPU memory.")

DATA_DIR = "C:/kaggletemp/sartorius-cell-instance-segmentation"
TDIR = DATA_DIR + "/" + "CP_TRAINING/"
LABEL = TDIR + '/LABEL/'
directory = LABEL  # sys.argv[1]
omni = False
use_gpu = False

if use_gpu:   #device_math.type == "cuda":
    print(f"Using GPU when tensors are loaded to 'device_math'.\n"
          f"  Device: {torch.cuda.get_device_name()}\n"
          f"  Number of GPUs: {torch.cuda.device_count()}\n"
          f"  GPU initialised: {torch.cuda.is_initialized()}\n")
    #print(torch.cuda.memory_summary())
else:
    print("GPU not available, all tensors loaded to 'device_math' will reside "
          "in CPU memory.")

#device =torch.device(type="cuda" ) #device_math

if use_gpu:
    device = torch.device('cuda')
    print("using gpu")
else:
    device =  torch.device("cpu")
    print("using cpu")


mask_files = [x for x in os.listdir(directory) if 'masks.tif' in x]


def compute_and_save_flow(directory, filename):
    mask_filename = os.path.join(directory, filename)
    flow_filename = mask_filename.replace('masks', 'flows')
    mask = imread(mask_filename)
    labels, dist, heat, veci = dynamics.masks_to_flows(mask, use_gpu=use_gpu, device=device, omni=omni)
    if omni:
        flow = np.concatenate((labels[np.newaxis, :, :], dist[np.newaxis, :, :], veci,
                               heat[np.newaxis, :, :]), axis=0).astype(np.float32)
    else:
        flow = np.concatenate((labels[np.newaxis, :, :], labels[np.newaxis, :, :] > 0.5, veci), axis=0).astype(
            np.float32)
    imsave(flow_filename, flow)


_ = joblib.Parallel(n_jobs=8)(
    joblib.delayed(compute_and_save_flow)(directory, filename) for filename in tqdm(mask_files))
