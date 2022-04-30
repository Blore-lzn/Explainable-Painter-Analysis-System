import os
import torch
from model_resnet import resnet50


def model_init(num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    model = resnet50(num_classes=num_classes).to(device)  # 需要修改

    # load model weights
    weights_path = "./resNet50.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


if __name__ == '__main__':
    model_init()
