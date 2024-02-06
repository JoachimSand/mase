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

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph


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


## Load own network

own_model_name = "jsc-ownnetwork"
dataset_name = "jsc"
batch_size = 8

own_data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=own_model_name,
    num_workers=0,
)
own_data_module.prepare_data()
own_data_module.setup()

own_model_info = get_model_info(own_model_name)
own_model = get_model(
    name="jsc-ownnetwork",
    task="cls",
    dataset_info=own_data_module.dataset_info,
    pretrained=False)

input_generator = InputGenerator(
    data_module=own_data_module,
    model_info=own_model_info,
    task="cls",
    which_dataloader="train",
)

# Generate dummy values
# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = own_model(**dummy_in)


own_mg = MaseGraph(model=own_model)

OWN_NETWORK_CHECK_POINT_PATH = "/home/jlsand/mase/mase_output/jsc-ownnetwork_classification_jsc_2024-02-06/software/training_ckpts/last-v4.ckpt" 
own_model = load_model(load_name=OWN_NETWORK_CHECK_POINT_PATH, load_type="pl", model=own_model)


own_mg, _ = init_metadata_analysis_pass(own_mg, None)
own_mg, _ = add_common_metadata_analysis_pass(own_mg, {"dummy_in": dummy_in})

own_ori_mg = MaseGraph(model=own_model)
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


compared_pre_post_quantized_graphs(own_ori_mg, own_mg, save_path=None, silent=False)
