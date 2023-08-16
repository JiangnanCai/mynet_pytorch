import torch
from backbone.inceptionv3 import InceptionV3
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = Image.open(r'/home/cai/Documents/Inception/archive(1)/dataset/test/e30d13f7b0.jpg')
plt.imshow(img)

img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

with open(r'/home/cai/Documents/Inception/archive(1)/cat_to_name.json', 'r', encoding="utf-8") as r:
    json_file = r.read()
class_indict = json.loads(json_file)

model = InceptionV3(in_channels=3, num_classes=5, aux_logits=False)
model_weight = r'/home/cai/Documents/Inception/inception_v3/inception_v3.pth'
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight), strict=False)
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img)[0])
    print(output)
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy() + 1
print(class_indict[str(predict_cla)])
plt.show()