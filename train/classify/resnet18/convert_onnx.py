import torch
import torchvision.models as models

input_path = './weights/sleep_20241219.pth'
output_path = 'sleep_classify_20241219_classify.onnx'


# 定义模型
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(in_features=512, out_features=2)
model.load_state_dict(torch.load(input_path))
model.eval()

# 准备示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 转换为 ONNX
torch.onnx.export(model, dummy_input, output_path, 
                  export_params=True,        # 是否导出参数
                  opset_version=12,          # ONNX 操作版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],     # 输入张量的名称
                  output_names=['output'])

print(f"模型已成功转换为 ONNX 格式并保存到 {output_path}")


