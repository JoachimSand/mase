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

