import timm

model = timm.create_model('resnetv2_50x1_bit', pretrained=True)
print(model)