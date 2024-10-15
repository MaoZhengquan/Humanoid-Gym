import torch
import onnx
# model = torch.jit.load('/home/mao/Github_Project/HIMLoco/legged_gym/logs/rough_gr1t1/exported/policies/policy.pt')
model = torch.jit.load('/home/mao/Github_Project/humanoid-gym/logs/1014EndWar/exported/policies/policy_1.pt')
model.eval()
dummy_input = torch.rand(1,615)
torch.onnx.export(
    model,                         # 你的模型
    dummy_input,                   # 示例输入
    "policy.onnx",               # 输出的 ONNX 文件名
    export_params=True,            # 保存模型权重
    opset_version=11,              # ONNX 的操作集版本（一般选择11或更新的版本）
    do_constant_folding=True,      # 是否执行常量折叠优化
    input_names=['input'],         # 输入的名称
    output_names=['output'],       # 输出的名称
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴，允许不同的batch size
)


onnx_model = onnx.load("policy.onnx")

try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")