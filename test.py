import torch
import timm


model = timm.create_model("resnet18", pretrained=True)
data = torch.randn((10, *model.pretrained_cfg["input_size"]))

print(model(data))

model = torch.compile(model)
print(model(data))
print(model(data))
