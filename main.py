import torch
import resnet.resnet as resnet

from torch import nn
from torchvision import transforms
from model import Net
from PIL import Image
from io import BytesIO
import re
import base64
from pythia_api import predictWithPythia
import warnings
# Cleaner demos : Don't do this normally...
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cpu')
state_path = './2017-08-04_00.55.19.pth'
saved_state = torch.load(state_path, map_location='cpu')
print(saved_state.keys())
qtoken_to_index = saved_state['vocab']['question']
answer_words = ['UNDEF'] * len(saved_state['vocab']['answer'])
print(len(answer_words))
for w, idx in (saved_state['vocab']['answer']).items():
    answer_words[idx] = w


class WrappedModel(nn.Module):

    def __init__(self, embedding_tokens):
        super().__init__()
        self.module = Net(embedding_tokens)

    def forward(self, v, q, q_len):
        return self.module.forward(v, q, q_len)


class ResNet152(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet.resnet152(pretrained=True)

        image_size = 448
        # output_features = 2048
        central_fraction = 0.875

        self.transform = get_transform(image_size, central_fraction)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

    def image_to_features(self, img):
        # img = Image.open(image_file).convert('RGB')
        img_transformed = self.transform(img)
        img_batch = img_transformed.unsqueeze(dim=0).to(device)
        return self.forward(img_batch)


def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])


def encode_question(question_str):
    tokens = question_str.lower().split(' ')
    vec = torch.zeros(len(tokens)).long()
    for i, token in enumerate(tokens):
        vec[i] = qtoken_to_index.get(token, 0)
    return vec.to(device), torch.tensor(len(tokens)).to(device)

# def get_model():

    # image_paths = [
    #     'C:/Users/mhala/Desktop/Projects/img/Black_pussy_-_panoramio.jpg',
    #     'C:/Users/mhala/Desktop/Projects/img/cat-black-jumping-off-wall_a-G-12469828-14258383.jpg',
    #     'C:/Users/mhala/Desktop/Projects/img/hqdefault.jpg',
    #     'C:/Users/mhala/Desktop/Projects/img/siamese5.jpg',
    #     'C:/Users/mhala/Desktop/Projects/img/cat_roof_home_architecture_building_roofs_animal_sit-536976.jpg',
    #     'C:/Users/mhala/Desktop/Projects/img/tabby-cat-colour-and-pattern-genetics-5516c44dbd383.jpg',
    # ]


    # print('-'*50)
    # for img_file in image_paths:
    #     v = resnet152.image_to_features(img_file)
    #     ans = model(v, q.unsqueeze(0), q_len.unsqueeze(0))
    #     _, answer_idx = ans.data.cpu().max(dim=1)
    #     print(answer_words[ answer_idx ])
    # print('-'*50)
model = WrappedModel(len(qtoken_to_index) + 1)
model.load_state_dict(saved_state['weights'])
model.to(device)
model.eval()

resnet152 = ResNet152().to(device)

tokens = len(saved_state['vocab']['question']) + 1


async def predict(text, img_base64, runInPythia=False):
    if runInPythia:
        return await predictWithPythia(text, img_base64)

    print("predect for Q:" + text)
    q, q_len = encode_question(text)
    img_base64 = re.sub('^data:image/.+;base64,', '', img_base64)
    img = Image.open(BytesIO(base64.b64decode(img_base64))).convert('RGB')
    v = resnet152.image_to_features(img)

    ans = model(v, q.unsqueeze(0), q_len.unsqueeze(0))
    _, answer_idx = ans.data.cpu().max(dim=1)

    print("most relevent answers:")
    all_ans_indx = ans.data.cpu()[0].sort(descending=True)[1][:5]
    for _ans in all_ans_indx:
        print(answer_words[_ans])

    return answer_words[answer_idx]

# if __name__ == '__main__':
#     main()




