import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
import torchvision.datasets as datasets


def train():
    
    print("Training starts...")
    B = 256
    E = 2
    EPOCHS = int(1e5)
    lr = 1e-5
    weights_decay = 0
    path = os.path.join("./","logs", f"B_{B}_E_{E}_lr_{str(lr)}_EPOCHS_{EPOCHS}_wd_{weights_decay}")
    print(f"Path logs training {path}")
    writer = SummaryWriter(path)
    
    # EPOCHS_UPDATE = 100
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device used:{device}")
    net = nn.Sequential(
        nn.Linear(2, 1, bias=False),
        # nn.ReLU(),
        # nn.Linear(2,1)
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(),
                                lr=lr,weight_decay=weights_decay)
    
    for epoch in tqdm(range(1,EPOCHS), desc="Epoch training"):
        optimizer.zero_grad()
        
        # generate batch
        x = torch.randint(0,10, size=(B, E), device=device, dtype=torch.float32)
        # generate res
        y_true = x.sum(dim=-1, keepdim=True)
        # predict
        y_pred = net(x)
        # loss computation
        loss = criterion(y_pred, y_true)
        # weights update
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss, epoch)
            

     

if __name__ == "__main__":
    print("train_sum starts...")
    train()

    