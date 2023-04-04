import importlib
import os


from ..session.plt_wrapper import get_model_wrapper
from ..session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
from ..session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
from ..session.plt_wrapper.nlp.translation import NLPTranslationModelWrapper
from ..session.plt_wrapper.vision import VisionModelWrapper


def _import_config_from_py_file(model_name: str, file_path: str):
    """
    load a config dict from .py file where a ignore_modules included nn.Module classes to be ignored in profiling
    """
    # default config
    config = {
        "print_profile": True,
        "detailed": True,
        "module_depth": -1,
        "top_modules": 1,
        "warm_up": 10,
        "as_string": True,
        "output_file": "estimate_sw_reports/{}.txt".format(
            model_name.replace("/", "-")
        ),
        "ignore_modules": [],
    }
    # import the custom config from .py file
    if file_path is not None:
        assert os.path.isfile(file_path) and file_path.endswith(
            ".py"
        ), "The config file should be an existing .py file"
        spec = importlib.util.spec_from_file_location("config_py", file_path)
        config_py = spec.loader.load_module()
        imported_config = config_py.config
        config.update(imported_config)

    return config


def get_input_args(model_name, model, task, data_loader, info):
    wrapper_cls = get_model_wrapper(model_name, task)
    plt_model = wrapper_cls(model, info=info, learning_rate=1e-4)

    input_args = []
    if isinstance(plt_model, VisionModelWrapper):
        data_loader.setup()
        data_loader.prepare_data()
        batch_x, _ = next(iter(data_loader.train_dataloader()))
        input_args = [batch_x[[0], ...]]
    elif isinstance(
        plt_model,
        (
            NLPClassificationModelWrapper,
            NLPLanguageModelingModelWrapper,
        ),
    ):
        batch = next(iter(data_loader.train_dataloader))
        # breakpoint()
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            None,
        ]
    elif isinstance(plt_model, NLPTranslationModelWrapper):
        batch = next(iter(data_loader.train_dataloader))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            batch["decoder_input_ids"][[0], ...],
            batch["decoder_attention_mask"][[0], ...],
        ]
    else:
        raise RuntimeError("Unsupported model class")
    return plt_model, input_args