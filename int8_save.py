import torch
import torch.nn as nn
from basic import VGG
import torch.nn.functional as F


conv1_param = []
conv2_param = []
conv3_param = []
conv4_param = []
conv5_param = []
conv6_param = []
conv7_param = []
blobscale=[]

file= open('./calibration.txt')


line = file.readline().strip('\n')
conv1_param=line.split(" ")
line = file.readline().strip('\n')
conv2_param=line.split(" ")
line = file.readline().strip('\n')
conv3_param=line.split(" ")
line = file.readline().strip('\n')
conv4_param=line.split(" ")
line = file.readline().strip('\n')
conv5_param=line.split(" ")
line = file.readline().strip('\n')
conv6_param=line.split(" ")
line = file.readline().strip('\n')
conv7_param=line.split(" ")
line = file.readline().strip('\n')
blobscale = line.split(" ")
del conv1_param[0]
del conv2_param[0]
del conv3_param[0]
del conv4_param[0]
del conv5_param[0]
del conv6_param[0]
del conv7_param[0]
del blobscale[0]
del blobscale[0]
# print('conv1_param:',conv1_param)
# print('conv2_param:',conv2_param)
# print('conv3_param:',conv3_param)
# print('conv4_param:',conv4_param)
# print('conv5_param:',conv5_param)
# print('conv6_param:',conv6_param)
# print('conv7_param:',conv7_param)
# print('blobscale:',blobscale)



class int8Net(nn.Module):

    def __init__(self):
        super(int8Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1=nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
        nn.Linear(in_features=12544, out_features=2048, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256, out_features=10, bias=True)
    )


    def forward(self, x):


        #x= round(x * blobscale) →  input to int8
        x = x * float(blobscale[0])
        x = torch.round(x)
        # int8 conv due to dp4a-gpu  cudnn cublas support  we got int32 and transform to float32
        x = self.conv1(x)
        x = x - self.conv1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # output float32 /div weight scale(every channel)
        for i,scale in enumerate(conv1_param):
            x[:,i,:,:]/=float(scale)
        # output float32 /div blobscale(input scale)
        x = x / float(blobscale[0])
        # output = x +  conv's fp32 bias
        x = x + self.conv1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #relu
        x = F.relu(x)
        # x = torch.round(x)


        #next layer same as ↑
        x = x * float(blobscale[1])
        x = torch.round(x)
        x = self.conv2(x)
        x = x - self.conv2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, scale in enumerate(conv2_param):
            x[:, i, :, :] /= float(scale)
        x = x / float(blobscale[1])
        x = x + self.conv2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = self.pool(F.relu(x))
        # x = torch.round(x)


        x = x * float(blobscale[2])
        x = torch.round(x)
        x = self.conv3(x)
        x = x - self.conv3.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, scale in enumerate(conv3_param):
            x[:, i, :, :] /= float(scale)
        x = x / float(blobscale[2])
        x = x + self.conv3.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = F.relu(x)
        # x = torch.round(x)

        x = x *float(blobscale[3])
        x = torch.round(x)
        x = self.conv4(x)
        x = x - self.conv4.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, scale in enumerate(conv4_param):
            x[:, i, :, :] /= float(scale)
        x = x / float(blobscale[3])
        x = x + self.conv4.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.pool(F.relu(x))
        # x = torch.round(x)

        x = x * float(blobscale[4])
        x = torch.round(x)
        x = self.conv5(x)
        x = x - self.conv5.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, scale in enumerate(conv5_param):
            x[:, i, :, :] /= float(scale)
        x = x / float(blobscale[4])
        x = x + self.conv5.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = F.relu(x)
        # x = torch.round(x)

        x = x *float( blobscale[5])
        x = torch.round(x)
        x = self.conv6(x)
        x = x - self.conv6.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, scale in enumerate(conv6_param):
            x[:, i, :, :] /= float(scale)
        x = x / float(blobscale[5])
        x = x + self.conv6.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = F.relu(x)
        # x = torch.round(x)

        x = x * float(blobscale[6])
        x = torch.round(x)
        x = self.conv7(x)
        x = x - self.conv7.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, scale in enumerate(conv7_param):
            x[:, i, :, :] /= float(scale)
        x = x / float(blobscale[6])
        x = x + self.conv7.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.pool(F.relu(x))
        # x = torch.round(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def gen_int8_model():

    model=VGG()
    map=torch.load("./model/fp32.pth")
    model.load_state_dict(map['net'])


    d_map={}
    for name,i in model.named_parameters():
       print(name,i.shape)
       if name == "conv1.weight":
          for index, scale in enumerate(conv1_param):
              i[index, :, :, :] *= float(scale)
          i=torch.round(i)

       if name == "conv2.weight":
           for index, scale in enumerate(conv2_param):
               i[index, :, :, :] *= float(scale)
           i = torch.round(i)

       if name == "conv3.weight":
           for index, scale in enumerate(conv3_param):
               i[index, :, :, :] *= float(scale)
           i = torch.round(i)

       if name == "conv4.weight":
           for index, scale in enumerate(conv4_param):
               i[index, :, :, :] *= float(scale)
           i = torch.round(i)

       if name == "conv5.weight":
           for index, scale in enumerate(conv5_param):
               i[index, :, :, :] *= float(scale)
           i = torch.round(i)

       if name == "conv6.weight":
           for index, scale in enumerate(conv6_param):
               i[index, :, :, :] *= float(scale)
           i = torch.round(i)

       if name == "conv7.weight":
           for index, scale in enumerate(conv7_param):
               i[index, :, :, :] *= float(scale)
           i = torch.round(i)


       d_map[name]=i
       torch.save(d_map,"./model/int8.pth")





if __name__ == "__main__":
    gen_int8_model()
    model = VGG()

    model.load_state_dict(torch.load("./model/fp32.pth")['net'])
    for name, i in model.named_parameters():
        if name == "conv1.weight":
            print(i)


    model.load_state_dict(torch.load("./model/int8.pth"))
    for name, i in model.named_parameters():
        if name == "conv1.weight":
            print(i)
