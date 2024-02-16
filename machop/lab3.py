import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

# figure out the correct path
machop_path = Path(".").resolve().parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()
print(data_module.dataset_info)

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=True,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)



## ---------------- Defining a search space ----------------
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": 8,
                "data_in_frac_width": 4,
                # weight
                "weight_width": 8,
                "weight_frac_width": 4,
                # bias
                "bias_width": 8,
                "bias_frac_width": 4,
            }
    },
}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        # Joachim: If deep copy was not used, the next iteration of the loop
        # would overwrite the passed arguments.
        search_spaces.append(copy.deepcopy(pass_args))
# grid search



## ---------------- Defining a search strategy and runner ----------------

import torch
from torchmetrics.classification import MulticlassAccuracy


from chop.passes.graph.transforms.quantize.quantize import (
    quantize_transform_pass      
);

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# metric = MulticlassAccuracy(num_classes=5)
# num_batchs = 50
# # This first loop is basically our search strategy,
# # in this case, it is a simple brute force search

# recorded_accs = []
# for i, config in enumerate(search_spaces):
#     mg, _ = quantize_transform_pass(mg, config)
#     j = 0

#     # this is the inner loop, where we also call it as a runner.
#     acc_avg, loss_avg = 0, 0
#     accs, losses = [], []
#     for inputs in data_module.train_dataloader():
#         xs, ys = inputs
#         preds = mg.model(xs)
#         loss = torch.nn.functional.cross_entropy(preds, ys)
#         acc = metric(preds, ys)
#         accs.append(acc)
#         losses.append(loss)
#         if j > num_batchs:
#             break
#         j += 1
#     acc_avg = sum(accs) / len(accs)
#     loss_avg = sum(losses) / len(losses)
#     recorded_accs.append(acc_avg)

#     l_config = config["linear"]
#     print(f"{acc_avg}")


## ------------------ Exploring with additional metrics ----------
from deepspeed.profiling.flops_profiler import FlopsProfiler

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

# Use the deepspeed flopsprofiler, see https://deepspeed.readthedocs.io/en/latest/flops-profiler.html
profiler = FlopsProfiler(model)

# New metrics.
# Note: For this search, FLOPS, MACS and params remain unchanged.
# The code for this is simply for demonstration.
rec_flops, rec_macs, rec_params, rec_avg_latencies = [], [], [], []

recorded_accs = []
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
        
    j = 0
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []

    flops, macs, params = None, None, None
    latencies = []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs

        profiler.start_profile()

        preds = mg.model(xs)
        
        flops = profiler.get_total_flops()
        macs = profiler.get_total_macs()
        params = profiler.get_total_params()
        latency = profiler.get_total_duration()

        profiler.end_profile()
        latencies.append(latency)
        
        loss = torch.nn.functional.cross_entropy(preds, ys)
        
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
    
    # FLOPS, MACS and params should be stable across each iteration of
    # the inner loop. Just pick the values from the last iteration.
    rec_flops.append(flops)
    rec_macs.append(macs)
    rec_params.append(params)

    latency_avg = sum(latencies) / len(latencies)
    rec_avg_latencies.append(latency_avg) 

    l_config = config["linear"]
    
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)

    latency_deadline = 0.0001600
    max_flops_feasible = 1000
    comb_metric = acc_avg - max(latency_avg - latency_deadline, 0) * 100 - max(max_flops_feasible - flops, 0) / 1000  

    print(f"COMB_METRIC: {comb_metric} FLOPS{flops} MACS: {macs} PARAMS: {params} LATENCY: {latency_avg} CONFIG: {l_config}")
    
    