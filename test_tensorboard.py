from torch.utils.tensorboard import SummaryWriter
import os
import random

if __name__ == "__main__":
    
    for 
    
    random_number = random.random()
    path = os.path.join("./","logs", str(random_number))
    print(path)
    writer = SummaryWriter(path)
    for i in range(1,19):
        writer.add_scalar("Loss/train", i, i)
