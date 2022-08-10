import torchvision.models as models
import torch
import onnx
import onnx.utils
import onnx.version_converter
from net import Unet_plus_plus

# 可生成onnx格式模型
# 定义数据+网络
data = torch.randn(1, 3, 80, 80)
net = Unet_plus_plus(deep_supervision=True, cut=False)
state_dict = torch.load('best_model.pth')
net.load_state_dict(state_dict, strict=False)

# 导出
torch.onnx.export(
    net,
    data,
    'model.onnx',
    export_params=True,
    opset_version=11,
)

# 增加维度信息
model_file = 'model.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
