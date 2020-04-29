import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from basic import VGG,TrainDataset,TestDataset

#训练fp32

lr = 0.01
momentum = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=VGG().to(device)
#model.load_state_dict(torch.load('./model/fp32.pth')['net'])

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = torch.nn.CrossEntropyLoss()
print(model)

traindata = TrainDataset()
testdata=  TestDataset()
train_loader = DataLoader(dataset=traindata,batch_size=128,shuffle=True)
test_loader = DataLoader(dataset=testdata,batch_size=128,shuffle=False)

bestacc=torch.load('./model/fp32.pth')['acc']
# if(bestacc<70.): bestacc=70.

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def test(epoch):
    global bestacc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print("Accuracy: ", acc)
    if acc > bestacc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('model'):
            os.mkdir('model')
        torch.save(state, './model/fp32.pth')
        bestacc = acc

if __name__ == "__main__":
    for epoch in range(0, 30):
        train(epoch)
        test(epoch)