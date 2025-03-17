import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize
import tifffile
import os
import torch.nn.init as init
# 定义模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(7,16,3,1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.linear1 = nn.Linear(512, 2)
        self.softmax = nn.Softmax()
        self.LN=nn.BatchNorm2d(num_features=7)
        self.initialize_weights()

    def forward(self, x):
        #print("x0=",x)
        #x=self.LN(x)
        x = self.relu(self.conv1(x))
        #print("x=",x)
        x = self.maxpool1(x)
        #print("x1=", x)
        x = self.relu(self.conv2(x))
        #print("x2=", x)
        x = self.maxpool1(x)
        #print("x3=", x)
        x = self.flatten(x)
        #print("x4=", x)
        x = self.linear1(x)
        #print("x5=", x)
        #x = self.softmax(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

torch.set_printoptions(profile="full")

def replace_inf_with_zero(tensor):
    # 使用torch.where将-inf替换为0
    return torch.where(torch.isinf(tensor), torch.tensor(0.0, device=tensor.device), tensor)


if __name__ == '__main__':
    root_dir = r"D:\大论文出图\深度学习\数据集"
    file_paths = []
    labels = []
    for idx,class_name in enumerate(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('tif'):
                    file_path = os.path.join(class_dir, file_name)
                    file_paths.append(file_path)
                    labels.append(idx)

    test_file_paths = file_paths
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 记载训练好的模型
    model = Classifier().to(device)
    # load model weights
    model_weight_path =r"F:\2024年2月19日数据库\验证区\新\测试代码\model7cnnchanel.pt"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    for idx in range(len(file_paths)):
        file_paths = file_paths
        normalize = Normalize([0.5, 0.5, 0.5, 0.5,0.5,0.5,0.5], [0.5, 0.5, 0.5, 0.5,0.5,0.5,0.5])
        image_path = file_paths[idx]
        # 读取图像并将其转换为张量
        image = tifffile.imread(image_path)
        image = ToTensor()(image)
        image = normalize(image)
        image=image.unsqueeze(dim=0)
        model.eval()

        with torch.no_grad():
            image= image.to(device)
            image = replace_inf_with_zero(image)
            outputs = model(image)
            # #print("output=",outputs)
            # _, predicted = torch.max(outputs.data, dim=1)
            outputs=outputs.cpu().detach().numpy()
            #求概率
            softmax_scores = np.exp(outputs) / np.sum(np.exp(outputs))
        #print("文件：", file_paths[idx], "预测属于第1类的概率：", softmax_scores[0,1])
            print(softmax_scores[0, 1])
        #print( [idx])
