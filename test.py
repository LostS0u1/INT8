import torch
from basic import TestDataset
from torch.utils.data import DataLoader
from int8_save import int8Net



def test(testmodel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = testmodel(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc

if __name__ == "__main__":

    accbefore=torch.load('./model/fp32.pth')['acc']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    int8model = int8Net().to(device)
    int8model.load_state_dict(torch.load('./model/int8.pth'))
    int8net=int8model.cuda()

    test_cifar10 = TestDataset()
    test_loader = DataLoader(dataset=test_cifar10, batch_size=64, shuffle=False)


    int8net.eval()
    accint8=test(int8net)

    print("量化前精度: ",accbefore ,"%")
    print("量化后精度: ",accint8 , "%")


