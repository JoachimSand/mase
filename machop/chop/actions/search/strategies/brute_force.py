import torch
import copy
import logging
from tabulate import tabulate

from .base import SearchStrategyBase

logger = logging.getLogger(__name__)

class SearchStrategyBruteForce(SearchStrategyBase):
    # Additional setup for subclass instance.
    def _post_init_setup(self) -> None:
        print("post_init setup for brute force search strat.")
        self.current_indexes = None

        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))

        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

        

    # Perform the search
    """ 
    search_strategy is responsible for:
    - setting up the data loader
    - perform search, which includes:
        - sample the search_space.choice_lengths_flattened to get the indexes
        - call search_space.rebuild_model to build a new model with the sampled indexes
        - calculate the software & hardware metrics through the sw_runners and hw_runners
    - save the results
    """

    
    def search(self, search_space):

        if self.current_indexes is None:
            first_search = True
            self.current_indexes = []
            for _ in search_space.choice_lengths_flattened.items():
                self.current_indexes.append(0)

        # Best configuration for each metric
        best_configs = [None for _ in range(0, len(self.metric_names))]
        
        # Metrics observed for each best (per-metric) configuration
        best_configs_metrics = [None for _ in range(0, len(self.metric_names))]
        
        trial = 0
        while True:
            if trial != 0:
                self.current_indexes[0] += 1
    
            done = False
            sampled_indexes = {}           
            for i, (name, length) in enumerate(search_space.choice_lengths_flattened.items()):
                if self.current_indexes[i] == length:
                    # We overflowed the last choice. Stop.
                    if i == (len(self.current_indexes) - 1):
                        done = True   
                    
                    self.current_indexes[i] = 0
                    if i + 1 < len(self.current_indexes):
                        self.current_indexes[i+1] += 1
                sampled_indexes[name] = self.current_indexes[i]

            if done:
                break
            print(self.current_indexes)
            print(sampled_indexes)

            sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

            # From here on out we follow the same approach as the optuna search strategy
            # No big surprises        
            is_eval_mode = self.config.get("eval_mode", True)
            model = search_space.rebuild_model(sampled_config, is_eval_mode)


            # Compute and record metrics from evaluating the new model
            software_metrics = self.compute_software_metrics(
                model, sampled_config, is_eval_mode
            )
            hardware_metrics = self.compute_hardware_metrics(
                model, sampled_config, is_eval_mode
            )
            metrics = software_metrics | hardware_metrics
            scaled_metrics = {}
            for metric_name in self.metric_names:
                scaled_metrics[metric_name] = (
                    self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
                )
            def update_best_configs(k):
                best_configs[k] = copy.deepcopy(sampled_config)
                best_configs_metrics[k] = copy.deepcopy(scaled_metrics)
            
            if not self.sum_scaled_metrics:
                for k, metric in enumerate(scaled_metrics):
                    if best_configs_metrics[k] is None:
                        update_best_configs(k)
                        continue
                    
                    if self.directions[k] == "maximize":
                        print(best_configs_metrics[k][metric])
                        if scaled_metrics[metric] > best_configs_metrics[k][metric]:
                            update_best_configs(k)
                    elif self.directions[k] == "minimize":
                        if scaled_metrics[metric] < best_configs_metrics[k][metric]:
                            update_best_configs(k)
                print("List of scaled metrics: ", list(scaled_metrics.items()))
            else:
                if sum(scaled_metrics.items()) > best_configs_metrics[0]:
                    update_best_configs(0)
                    
                print("Sum of scaled metrics: ", )

            self.visualizer.log_metrics(metrics=scaled_metrics, step=trial)
            trial += 1
            # if best_config is None:
            #     best_config = sampled_config
            #     best_config_metrics = scaled_metrics

        logger.info(f"Best configurations: {best_configs}")

        table = tabulate(best_configs_metrics, headers="keys", tablefmt="orgtbl")
        logger.info(f"Metrics for best configurations:\n {table}")
        return best_configs

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
        # note that model can be mase_graph or nn.Module
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics
    
