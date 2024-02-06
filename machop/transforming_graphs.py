import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp


from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")

# Set up the MaseDataModule using the jsc data set.
batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"

    

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()


# Load the previously trained jsc model.
CHECKPOINT_PATH = "/home/jlsand/mase/mase_output/jsc-tiny_classification_jsc_2024-02-04/software/training_ckpts/best.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)


# Torch fx uses a symbolic tracer where fake values/proxies
# are fed through the network. Operations on the proxies are recorded.
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# Generate dummy values
# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)


# Generate a MaseGraph. Takes in a previously trained model.
mg = MaseGraph(model=model)

# Two types of passes can be run by MASE: 
# - Analysis pass. Does not change the graph, adds metadata etc.
# - Transform passes. Does not change graph.
# Similar to the dichotomy of analysis/transform passes in LLVM.

mg, _ = init_metadata_analysis_pass(mg, None)
# Here we pass in the dummy values generated earlier.
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# report graph is an analysis pass that shows you the detailed information in the graph
from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)


# Run the statistics analysis pass
pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}

mg, _ = profile_statistics_analysis_pass(mg, pass_args)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})




# Run the quantisation analysis pass. ----------
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph


# Save the original mase graph for the sake of comparison with quantised MaseGraph
ori_mg = MaseGraph(model=model)
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})

def quantize_and_compare(mg: MaseGraph, ori_mg: MaseGraph):

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



    mg, _ = quantize_transform_pass(mg, pass_args)
    summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")

quantize_and_compare(mg, ori_mg)

## Load own network
own_model = get_model(
    name="jsc-ownnetwork",
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

own_mg = MaseGraph(model=own_model)

OWN_NETWORK_CHECK_POINT_PATH = "/home/jlsand/mase/mase_output/jsc-jsc-ownnetwork_classification_jsc_2024-02-06/software/training_ckpts/best.ckpt" 
own_model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=own_model)

own_ori_mg = MaseGraph(model=model)
own_ori_mg, _ = init_metadata_analysis_pass(own_ori_mg, None)
own_ori_mg, _ = add_common_metadata_analysis_pass(own_ori_mg, {"dummy_in": dummy_in})

quantize_and_compare(own_mg, own_ori_mg)



# Own traversal of the original and quantised graphs -----
from tabulate import tabulate

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.tools.logger import get_logger

logger = get_logger(__name__)


def compared_pre_post_quantized_graphs(
    ori_graph, graph, save_path=None, silent=False
):
    """List all nodes in the graph and compare the original and quantized nodes."""

    def get_type_str(node):
        if node.op == "call_module":
            return type(get_node_actual_target(node)).__name__
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
            "patched_func",
        ]:
            return get_node_actual_target(node).__name__
        elif get_mase_type(node) in ["implicit_func"]:
            actual_target = get_node_actual_target(node)
            if isinstance(actual_target, str):
                return actual_target
            else:
                return actual_target.__name__
        else:
            return node.target

    headers = [
        "Ori name",
        "New name",
        "MASE_TYPE",
        "Mase_OP",
        "Original type",
        "Quantized type",
        "Changed",
    ]
    rows = []
    for ori_n, n in zip(ori_graph.fx_graph.nodes, graph.fx_graph.nodes):
        rows.append(
            [
                ori_n.name,
                n.name,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )
    if not silent:
        logger.debug("Compare nodes:")
        logger.debug("\n" + tabulate(rows, headers=headers, tablefmt="orgtbl"))
    logger.info("\n" + tabulate(rows, headers=headers))
    
compared_pre_post_quantized_graphs(ori_mg, mg, save_path=None, silent=False)
