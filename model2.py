import torch
import torch.nn as nn
from torchsummary import summary


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(1,32, kernel_size=3, stride=1 ,padding=1,bias=False),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32,64, kernel_size=3, stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64,128, kernel_size=3, stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Linear(128*25,6)
    def forward(self, x):
        out = self.encode(x)
        # 预测
        out = out.view(out.size(0), -1)
        pre = self.fc(out)
        return pre

if __name__ == '__main__':
    model = classifier()
    #summary(model, (1, 200))
    input = torch.ones((1,1,200))
    output = model(input)
    output_s = torch.softmax(output / 2, dim=1)
    print(output)
    print(output_s)
