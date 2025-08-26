import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .train import get_data
from .models import *
from .test import get_model

import time
import numpy as np
import torch.distributions as dist
from torch.nn.functional import kl_div


import torch
from .train import get_data
import time
import numpy as np

A = torch.load('average_jacobian_pedpred3_200.pt')  # [1, 10, 4, 36, 12, 1, 10, 4, 36, 12]
A = A.squeeze(0).squeeze(4).reshape(17280, 17280)  # reshape to [D, D]

b = torch.load('estimated_bias.pt')  # [17280]
b = b.view(1, -1)  # [1, D]



# Stefan_model = PedPred()
# my_model = PedPred()
#
# Stefan_model.load_state_dict(torch.load("corridor_8D_Stefancode_mymodel.pth"))
# my_model.load_state_dict(torch.load("corridor_8D_Stefancode_mymodel.pth"))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Stefan_model.to(device).eval()
# my_model.to(device).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_kld(student_logits, teacher_logits, temperature=1.0):
    """
    Computes the KL Divergence between the output distributions of the teacher and student.
    """
    T = temperature
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    kld = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
    return kld

# 🔍 Run comparison
total_kld = 0.0
total_stefan_time = 0.0
total_my_time = 0.0
num_batches = 0

# default_model = 'corridor_train28d_Stefan_8epoch.pth' #Stefan model 8 epoch train with 28 days
# default_model = 'corridor_train28d_Stefan.pth'  # Stefan model 40 epoch train with 28 days
# default_model = 'normal-emu_train_model_28D.pth'  # my model PedPred2
# default_model = 'apt-ibex_train_model_28D.pth'  # my model PedPred3

Stefan_model_8epoch = PedPred()
Stefan_model_8epoch.load_state_dict(torch.load('corridor_train28d_Stefan_8epoch.pth',map_location=device))

Stefan_model_40epoch = PedPred()
Stefan_model_40epoch.load_state_dict(torch.load('corridor_train28d_Stefan.pth',map_location=device) )

my_model_PedPred2 =PedPred2()
checkpoint = torch.load('normal-emu_train_model_28D.pth', map_location=device)
my_model_PedPred2.load_state_dict(checkpoint['model'])

my_model_PedPred3 = PedPred3()
checkpoint = torch.load('apt-ibex_train_model_28D.pth', map_location=device)
my_model_PedPred3.load_state_dict(checkpoint['model'])

def density_mae(pred_density, true_density):
    return torch.abs(pred_density - true_density).mean()

def velocity_mae(target_density, pred_vel, true_vel):
    return torch.norm(target_density * (pred_vel - true_vel), dim=-1).mean()

def directional_consistency(target_density, pred_var, true_var):
    return torch.abs(target_density * (pred_var - true_var)).mean()


# def directional_consistency(pred_vel, true_vel):
#     # Cosine similarity between velocity vectors
#     cos_sim = F.cosine_similarity(pred_vel, true_vel, dim=-1)
#     return cos_sim.mean()  # Range [-1, 1], higher = better

with torch.no_grad():
    data = get_data('test2')
    inputs, targets, preds = [], [], []
    stefan_all_times =[]
    mine2_all_times = []
    mine3_all_times = []
    linear_all_times = []
    # Initialize KLD accumulators


    num_samples = 5000
    density_mae_stefan_8epoch =[]
    density_mae_stefan_40epoch = []
    density_mae_my_model2 = []
    density_mae_my_model3 = []
    density_mae_linear = []

    velocity_mae_stefan_8epoch = []
    velocity_mae_stefan_40epoch = []
    velocity_mae_my_model2 = []
    velocity_mae_my_model3 = []
    velocity_mae_linear = []

    direction_stefan_8epoch = []
    direction_stefan_40epoch = []
    direction_my_model2 = []
    direction_my_model3 = []
    direction_linear=[]
    for i, (input, target) in enumerate(data):
        if i > num_samples:
            break
        # print(f"testing sample {i}")

        x_flat = input.view(1, -1)
        start = time.time()
        output_flat = x_flat @ A.T + b  # [1, D]
        stefan_time = time.time() - start
        linear_all_times.append(stefan_time)
        linear_output = output_flat.view(1, 10, 4, 36, 12)
        density_mae_s = 0
        velocity_mae_s = 0
        direction_inconsistancy_s = 0
        for t in range(10):
            density_mae_s += density_mae(linear_output[0, t][0], target[0, t][0])
            velocity_mae_s += velocity_mae(target[0, t][0], linear_output[0, t][1:3], target[0, t][1:3])
            direction_inconsistancy_s += directional_consistency(target[0, t][0], linear_output[0, t][3], target[0, t][3])
        density_mae_linear.append(density_mae_s)
        velocity_mae_linear.append(velocity_mae_s)
        direction_linear.append(direction_inconsistancy_s)

        #Stefan
        start_time = time.time()
        stefan8_output = Stefan_model_8epoch(input, horizon=target.shape[1])
        # stefan8_probs = torch.softmax(stefan8_output.data, dim=-1)  # Access .data attribute
        stefan_time = time.time() - start_time
        stefan_all_times.append(stefan_time)
        density_mae_s = 0
        velocity_mae_s = 0
        direction_inconsistancy_s = 0
        for t in range(10):
            density_mae_s += density_mae(stefan8_output[0,t][0], target[0,t][0])
            velocity_mae_s += velocity_mae(target[0, t][0], stefan8_output[0, t][1:3], target[0, t][1:3])
            direction_inconsistancy_s += directional_consistency(target[0, t][0], stefan8_output[0, t][3], target[0, t][3])
        density_mae_stefan_8epoch.append(density_mae_s)
        velocity_mae_stefan_8epoch.append(velocity_mae_s)
        direction_stefan_8epoch.append(direction_inconsistancy_s)

        start_time = time.time()
        stefan40_output = Stefan_model_40epoch(input, horizon=target.shape[1])
        stefan40_time = time.time() - start_time
        density_mae_s = 0
        velocity_mae_s = 0
        direction_inconsistancy_s = 0
        for t in range(10):
            density_mae_s += density_mae(stefan40_output[0, t][0], target[0, t][0])
            velocity_mae_s += velocity_mae(target[0, t][0], stefan40_output[0, t][1:3], target[0, t][1:3])
            direction_inconsistancy_s += directional_consistency(target[0, t][0], stefan40_output[0, t][3],
                                                                 target[0, t][3])
        density_mae_stefan_40epoch.append(density_mae_s)
        velocity_mae_stefan_40epoch.append(velocity_mae_s)
        direction_stefan_40epoch.append(direction_inconsistancy_s)


        start_time = time.time()
        my_output2 = my_model_PedPred2(input, horizon=target.shape[1])
        my_time = time.time() - start_time
        mine2_all_times.append(my_time)
        density_mae_s = 0
        velocity_mae_s = 0
        direction_inconsistancy_s = 0
        for t in range(10):
            density_mae_s += density_mae(my_output2[0, t][0], target[0, t][0])
            velocity_mae_s += velocity_mae(target[0, t][0], my_output2[0, t][1:3], target[0, t][1:3])
            direction_inconsistancy_s += directional_consistency(target[0, t][0], my_output2[0, t][3],
                                                                 target[0, t][3])
        density_mae_my_model2.append(density_mae_s)
        velocity_mae_my_model2.append(velocity_mae_s)
        direction_my_model2.append(direction_inconsistancy_s)

        start_time = time.time()
        my_output3 = my_model_PedPred3(input, horizon=target.shape[1])
        my_time = time.time() - start_time
        mine3_all_times.append(my_time)
        density_mae_s = 0
        velocity_mae_s = 0
        direction_inconsistancy_s = 0
        for t in range(10):
            density_mae_s += density_mae(my_output3[0, t][0], target[0, t][0])
            velocity_mae_s += velocity_mae(target[0, t][0], my_output3[0, t][1:3], target[0, t][1:3])
            direction_inconsistancy_s += directional_consistency(target[0, t][0], my_output3[0, t][3],
                                                                 target[0, t][3])
        density_mae_my_model3.append(density_mae_s)
        velocity_mae_my_model3.append(velocity_mae_s)
        direction_my_model3.append(direction_inconsistancy_s)

def report_timing(name, times):
    mean = np.mean(times)
    std = np.std(times)
    print(f"{name:15s} Avg Inference Time: {mean:.4f} ± {std:.4f} sec")

def report_results(name, times):
    mean = np.mean(times)
    std = np.std(times)
    print(f"{name:15s} Avg: {mean:.4f} ± {std:.4f}")

report_timing("Stefan Model", stefan_all_times)
report_timing("My Model", mine2_all_times)
report_timing("My Model", mine3_all_times)
report_timing("My Model", linear_all_times)

print('density')
report_results("Stefan Model 8epoch ", density_mae_stefan_8epoch)
report_results("Stefan Model 40 epoch", density_mae_stefan_40epoch)
report_results("My Model PedPred2", density_mae_my_model2)
report_results("My Model PedPred3", density_mae_my_model3)
report_results("My Model linear", density_mae_linear)

print('velocity')
report_results("Stefan Model 8epoch ", velocity_mae_stefan_8epoch)
report_results("Stefan Model 40 epoch", velocity_mae_stefan_40epoch)
report_results("My Model PedPred2", velocity_mae_my_model2)
report_results("My Model PedPred3", velocity_mae_my_model3)
report_results("My Model linear", velocity_mae_linear)

print('direction')
report_results("Stefan Model 8epoch ", direction_stefan_8epoch)
report_results("Stefan Model 40 epoch", direction_stefan_40epoch)
report_results("My Model PedPred2", direction_my_model2)
report_results("My Model PedPred3", direction_my_model3)
report_results("My Model linear", direction_linear)
