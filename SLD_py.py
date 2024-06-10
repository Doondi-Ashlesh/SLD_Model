import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#Loading pre-trained ResNet-50 model
model=torchvision.models.resnet50(pretrained=True)

#Freeze Base Model Layers
for param in model.parameters():
    param.requires_grad=False

#Replace Final Classification Layer
num_classes=len(train_dataset.classes)
model.fc=nn.Linear(model.fc.in_features, num_classes)

#Define  data transforms and datasets
changes = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='C:\\Users\\THUNDER\\Desktop\\Projects\\SLD\\train', transform=changes)
test_dataset = datasets.ImageFolder(root='C:\\Users\\THUNDER\\Desktop\\Projects\\SLD\\test', transform=changes)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


#Loss Function and Optimizer 
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=0.005)


#Setting the Model
epochs = 20
losses=[]
nofepochs=[]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device,)

#Training the Model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    nofepochs.append(epoch+1)
    losses.append(epoch_loss)
    

#Visualization
plt.plot(nofepochs, losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

#Saving the current model
#Assuming `model` is your trained model
#torch.save(model.state_dict(), 'SLD1.pth')


#Evaluation of the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')