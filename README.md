# Deconstructing What Makes a Good Optimizer for Language Models

Accompanying code for experiments in "Deconstructing What Makes a Good Optimizer For Language Models". Code is built on top of the AI2 OLMo repository; more specifically, this repository contains modifications to the base `olmo/` directory as well as the `scripts/` and `configs/` necessary to reproduce the main results.

## Installation

```
conda create -n opt-olmo python=3.10
conda activate opt-olmo
pip install -e .[all]
```

## Configuration Files
Config files used to generate plots of all main experiments are in `configs`. The base model configuration is given in `configs/base-c4-t5.yaml` where data paths (see parameter `data.paths` on line 93 in `configs/base-c4-t5.yaml`) to preprocessed `npy` files must be specified. The specific model configurations for different parameter sizes (100m, 300m, 600m, 1.2b) are given in `configs/models`.

Sweep configuration files given in `configs/sweeps` corresponding to the sweeps from all main experiments (eg. learning rate sweep, other 1-D hyperparameter sweeps, SGD+Adalayer, Frozen Adalayer). 

## How to Run Sweeps
The main script to run a sweep is `scripts/run_sweep.py`, which requires a `config` and a `sweep_config` string argument. This automatically parses the sweep config `yaml` file by flattening the nested arguments to create an `overrides` list and selecting one override according to the `slurm_task_id` argument. The `config` argument can be multiple paths to regular configuration files separated by a `+` sign; for instance, to run the main learning rate sweep on the 300m model, it suffices to run 

```
python scripts/run_sweep.py --config=configs/base-c4-t5.yaml+configs/models/300m.yaml --sweep_config=configs/sweeps/lr.yaml 
```

## Notes on Specific Configuration Files
**Base Hyperparameter Sweeps**: Configuration files are labeled by the hyperparameter being swept over when fixing the optimal learning rate for each optimizer (eg. `configs/sweeps/beta.yaml`, `configs/sweeps/momentum.yaml`). To run the same sweeps for an architecture without QK-LayerNorm or without z-loss, respectively set `model.attention_layer_norm` to `False` or `softmax_auxilary_loss` to `False` (for an example, see configs with the suffixes `no_qkz.yaml`).

**SGD + Adalayer/Adafactor**: See `configs/sweeps/adalayer+sgd.yaml` or `configs/sweeps/adafactor+sgd.yaml` for an example. The name of the optimizer to use is `optimizer.name: adalayerw_last` and `optimizer_name: adafactorw_last` respectively. The config parameters `update_norm` and `update_last` correspond to whether Adalayer/Adafactor is applied to the norm layers or the last layer respectively. The parameter `lr_last` sets the learning rate of the norm layers/last layer if `update_norm` or `update_last` are set to `True`.

**Frozen Adalayer**: See `configs/sweeps/adalayer_freeze.yaml` for an example. Set the learning rate scheduler parameter `scheduler.name` to `freeze_cosine_with_warmup`, which runs at zero learning rate for `optimizer.t_freeze` steps before doing the usual warmup and cosine decay. The `max_duration` parameter should be `optimizer.t_freeze` longer to account for this. As in SGD + Adalayer, set `update_norm` and `update_last` correspond to whether Adalayer continues to update the norm and last layers following the frozen scales at initialization.

## Citing

```bibtex
@article{zhao2024deconstructing,
  title={Deconstructing What Makes a Good Optimizer for Language Models},
  author={Zhao, Rosie and Morwani, Depen and Brandfonbrener, David and Vyas, Nikhil and Kakade, Sham},
  journal={arXiv preprint arXiv:2407.07972},
  year={2024}
}
```
