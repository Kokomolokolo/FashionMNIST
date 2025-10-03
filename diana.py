import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# config

def get_config():
    return {
        "num_epochs": 2,
        "batch_size": 32,
        "lr": 0.01,
        "path": "./models/diana.pth"
    }

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # 50 % Chance das sich der Bild dreht
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # pytroch erwartet tupels

    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return train_transform, test_transform

def load_dataset(train_transform, test_transform):
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        transform=train_transform,
        train= True,
        download=True
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", 
        train=False, 
        transform=test_transform, 
        download=True,
    )

    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, config):
    dataloader = torch.utils.data.DataLoader
    train_dataloader = dataloader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    test_dataloader = dataloader(
        test_dataset, shuffle=False, batch_size=config["batch_size"]
    )

    return train_dataloader, test_dataloader

classes = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
def imshow(imgs):
    imgs = imgs / 2 + 0.5
    npimgs = imgs.numpy()
    
    # Check ob Grayscale oder RGB
    if npimgs.shape[0] == 1:  # Grayscale
        plt.imshow(npimgs.squeeze(), cmap='gray')
    else:  # RGB
        plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()
def show_sample_data(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
    imshow(img_grid)

# show_sample_data(get_data_loaders(
#     load_dataset(get_transforms()[0], get_transforms()[1])[0],
#     load_dataset(get_transforms()[0], get_transforms()[1])[1], 
#     get_config()
# )[0])

class Diana(nn.Module):
    def __init__(self):
        super(Diana, self).__init__()
        # pool halbiert die menge, conv2d macht es je nach padding kleiner. Da kein padding: um 2 kleiner, da auf jeder seite eins verloren geht
        self.conv1 = nn.Conv2d(1, 32, 3) # 3 colo channels, output, kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3) # in und out sind die anzahl der feature maps. Diese bestehen aus verschiedenen Kernels, die das moldel selber lenrt
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*3*3, 64) # steht für fully connected, 3x3 ist die bildgröße an diesem Punkt
        self.fc2 = nn.Linear(64, 10) # 10 wegen 10 output classes

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = torch.flatten(x, 1) #Vorher:  (Batch=32, Channels=64, Height=3, Width=3, Nachher: (Batch=32, Features=576)
        # FCs
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Erstellt ein model, wow
def create_model(device):
    model = Diana().to(device)
    return model

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0 # der verlust während einer epoche
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forwardprob
        outputs = model(images) # fragt das model nach den images
        loss = criterion(outputs, labels) # erechnet den unterschied zwischen den outputs und den labels

        #backprob
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # run_loss
        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f'Loss: {loss.item():.4f}')

    end_time = time.time() - start
    avg_loss = running_loss / len(train_loader)
    return end_time, avg_loss

def train_model(model, train_loader, optimizer, criterion, device, config, scheduler):
    print("Started Training")
    print("-" * 100)

    start_time = time.time()

    for e in range(config["num_epochs"]):
        epoch_time, loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch Time: {epoch_time:.3f}s, Running Loss {loss:.3f}, LR {current_lr:.6f}")

    end_time = time.time() - start_time

    print("-" * 100)
    print(f"Training Complete")
    print(f"Total Training Time: {end_time:.2f}s ({end_time/60:.2f} min")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = get_config()

    train_transform, test_transform = get_transforms()
    train_dataset, test_dataset = load_dataset(train_transform, test_transform)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, config)

    model = create_model(device)

    optimizer = optim.Adam(params=model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = nn.CrossEntropyLoss()

    train_model(
        model, train_loader, optimizer, criterion, device, config, scheduler
    )

    # save model
    print(f"Saving model to {config['path']}")
    torch.save(model.state_dict(), config["path"])

    # Eval the model on a test dataset
    eval_model(model, test_loader, device)



def eval_model(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        n_corr = 0 # number of correct classifications
        n_test = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # max wert der zeile, dim 1
            n_corr += (predicted == labels).sum().item()

        acc = 100.0 * n_corr / n_test

        print(f"Das trainierte Model hat eine acc von {acc}")


if __name__ == "__main__":
    main()