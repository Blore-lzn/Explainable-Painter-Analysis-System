import argparse
import json

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

from model_init import *

num_classes = 79
resnet50 = model_init(num_classes)  # 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

json_file = open(json_path, "r")
class_indict = json.load(json_file)


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():  # resnet50没有.feature这个特征，直接删除用就可以。
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    def __init__(self, model, target_layers, use_cuda):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        self.cuda = use_cuda

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        if self.cuda:
            output = output.cpu()
            output = resnet50.fc(output).cuda()  # 这里就是为什么我们多加载一个resnet模型进来的原因，因为后面我们命名的model不包含fc层，但是这里又偏偏要使用。#
        else:
            output = resnet50.fc(output)  # 这里对应use-cuda上更正一些bug,不然用use-cuda的时候会导致类型对不上,这样保证既可以在cpu上运行,gpu上运行也不会出问题.
        return target_activations, output


def preprocess_image(img):
    data_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])

    # [N, C, H, W]
    img = data_transform(img)

    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    return img


def predict(input_img):
    resnet50.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(resnet50(input_img.to(device))).cpu()
        predict_ = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict_).numpy()

    print_res = "Author: {}   Prob: {:.3}".format(class_indict[str(predict_cla)],
                                                  predict_[predict_cla].numpy())

    return class_indict[str(predict_cla)], predict_[predict_cla].numpy(), print_res


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_cam_weights(type, grads_val, activations):
    if type == "grad_cam":
        return np.mean(grads_val, axis=(2, 3))[0, :]
    else:
        grads = grads_val
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)
        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))[0, :]
        return weights


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None, cam_type="grad_cam"):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        activations = target.cpu().data.numpy()
        target = target.cpu().data.numpy()[0, :]

        weights = get_cam_weights(cam_type, grads_val, activations)

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./train/',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def grad_cam_run(image):
    args = get_args()

    model = model_init()
    del model.fc

    grad_cam = GradCam(model,
                       target_layer_names=["layer4"],
                       use_cuda=args.use_cuda)

    img = np.array(image)
    rgb_img = img[:, :, (2, 1, 0)][:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    img = np.float32(cv2.resize(rgb_img, (224, 224))) / 255
    input = preprocess_image(img)
    input.required_grad = True

    target_index = None
    mask = grad_cam(input, target_index)
    cam_result = show_cam_on_image(img, mask)

    return cam_result


def cal_evaluation():
    pass


def explainInGradCAM(img_path):
    model = model_init(num_classes)
    del model.fc

    grad_cam = GradCam(model,
                       target_layer_names=["layer4"],
                       use_cuda=False)
    resize = transforms.Resize([224, 224])
    img = Image.open(img_path)
    img = resize(img)
    to_tensor = transforms.ToTensor()
    img_ori = to_tensor(img)

    img = preprocess_image(img)

    predict(img)

    tmp = cv2.imread(img_path, 1)
    print(tmp.shape)
    x = tmp.shape[0]
    y = tmp.shape[1]
    print([x, y])

    img = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))

    input_img = preprocess_image(img)

    input_img.required_grad = True
    target_index = None

    mask = grad_cam(input_img, target_index, "grad_cam++")
    img = np.float32(cv2.resize(tmp, (224, 224))) / 255
    result = cv2.resize(show_cam_on_image(img, mask), (y, x))
    print(result.shape)
    cv2.imwrite('static/grad_cam++.png', result)

    new = np.expand_dims(mask, 0).repeat(3, axis=0)
    img_ori = np.multiply(img_ori, new)
    to_PILimage = transforms.ToPILImage()
    img_restored = to_PILimage(img_ori)
    img_restored.save('static/mask.png')
