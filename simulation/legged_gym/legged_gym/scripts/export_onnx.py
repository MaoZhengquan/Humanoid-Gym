from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry
from datetime import datetime
import torch

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    models = [file for file in os.listdir(load_run)]
    models.sort(key=lambda m: "{0:0>15}".format(m))
    model = models[-1]

    load_path = os.path.join(load_run, model)
    return load_path

def export_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # load jit
    # log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_policies')
    log_root = args.load_run
    print("log_root", log_root)
    # model_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    model_path = args.load_run
    print("Load model from:", model_path)
    jit_model = torch.jit.load(model_path)
    jit_model.eval()

    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                             train_cfg.runner.experiment_name, 'exported_onnx',
                             current_date_time)

    os.makedirs(root_path, exist_ok=True)
    dir_name = args.task.split("_")[0] + "_policy.onnx"
    path = os.path.join(root_path, dir_name)
    example_input = torch.randn(1, env_cfg.env.num_observations)
    # export onnx model
    torch.onnx.export(jit_model,
                      example_input,
                      path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      )
    print("Export onnx model to: ", path)

if __name__ == "__main__":
    args = get_args()
    if args.load_run == -1:
        args.load_run = -1
    export_onnx(args)