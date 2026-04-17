import timm

model = timm.create_model('resnet18', pretrained=True)
print(model)