import os
import torch
import torch.nn as nn
from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    model = resnet50(num_classes=50).to(device)

    # load model weights
    # weights_path = "./resNet50.pth"
    assert os.path.exists(model_weight_path), "file: '{}' dose not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    return model


if __name__ == '__main__':
    main()
