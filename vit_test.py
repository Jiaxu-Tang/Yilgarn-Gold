import json
import imagecodecs
import torch
from torchvision import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from vit_model import vit_base_patch16_224 as create_model
import numpy as np
def replace_inf_with_zero(tensor):
    # 使用torch.where将-inf替换为0
    return torch.where(torch.isinf(tensor), torch.tensor(0.0, device=tensor.device), tensor)
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5,0.5, 0.5, 0.5,0.5], [0.5, 0.5, 0.5,0.5, 0.5, 0.5,0.5])])
    # load image

    root = r"D:\大论文出图\深度学习\数据集\images"
    i = []
    for e in os.listdir(root):
        if e.endswith('tif'):
            e = os.path.join(root, e)
            #print(e)
            i.append(e)
            # assert os.path.exists(i), "file: '{}' dose not exist.".format(i)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "F:\VIT\Vision_transformer\weights\model-last1.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # 读取TIF图像数据
    for idx in range(len(i)):
        t = i[idx]

        # 解码TIF图像
        img = imagecodecs.imread(t)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            output = replace_inf_with_zero(output)
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print(predict[predict_cla].numpy())



    #print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     #predict[predict_cla].numpy())
    #print("预测：")
    #print(print_res)


if __name__ == '__main__':
    main()
