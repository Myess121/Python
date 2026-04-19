import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from model_def import get_model

def train_model():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_ratio=0.1
    val_size=int(len(train_dataset)*val_ratio)
    train_size=len(train_dataset)-val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    batch_size=64
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    model=get_model(10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs=30
    best_acc=0.0

    for epoch in range(1,epochs+1):
        model.train()
        train_loss=0
        train_acc=0
        train_total=0
        train_correct=0
        train_bar=tqdm(train_loader,desc=f'train epoch {epoch}/{epochs}')
        for images,labels in train_bar:
            images,labels=images.to(device),labels.to(device)
            output=model(images)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _,predicted=output.max(1)
            train_total+=labels.size(0)
            train_correct += (predicted.eq(labels).sum().item())
            train_bar.set_postfix({'Loss': f'{train_loss /(train_bar.n+1):.3f}','Acc': f'{100. * train_correct / train_total:.2f}%'})
        train_loss=train_loss/len(train_loader)
        train_acc=100.0*train_correct/train_total

        model.eval()
        val_loss=0
        val_correct=0
        val_total=0
        with torch.no_grad():
            val_bar=tqdm(val_loader,desc=f'validate epoch {epoch}/{epochs}')
            for images,labels in val_bar:
                images,labels=images.to(device),labels.to(device)
                output=model(images)
                loss=criterion(output,labels)
                val_loss+=loss.item()
                _,predicted=output.max(1)
                val_total+=labels.size(0)
                val_correct+=(predicted.eq(labels).sum().item())
                val_bar.set_postfix({'Loss': f'{val_loss /(val_bar.n+1):.3f}','Acc': f'{100. * val_correct / val_total:.2f}%'})
            val_loss=val_loss/len(val_loader)
            val_acc=100.0*val_correct/val_total
            current_lr=optimizer.param_groups[0]['lr']
            scheduler.step()
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:6.2f}% | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:6.2f}% | Learning Rate: {current_lr:.6f}")
            if val_acc>best_acc:
                best_acc=val_acc
                os.makedirs('models_new', exist_ok=True)
                torch.save(model.state_dict(), 'models_new/best_model.pth')
    test_model(model,device)


def test_model(model, device):
    batch_size=64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    test_correct=0
    test_total=0
    with torch.no_grad():
        for images, labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predicted=outputs.max(1)
            test_correct += (predicted.eq(labels).sum().item())
            test_total+=labels.size(0)
    test_acc=100.0*test_correct/test_total
    print(f"test_accuracy: {test_acc:.2f}%")

if __name__=='__main__':
    train_model()


