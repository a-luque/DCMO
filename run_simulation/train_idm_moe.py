import os
import io
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append("../src")
sys.path.append('..')
sys.path.append('./')
from src.idm_moe import BasicMOE

def get_dataloaders(data_file="./train_idm_moe.npy", batch_size=32, num_workers=4):
    train_data = np.load(data_file, allow_pickle=True)

    split_test_size = 0.1

    X_train, X_test = train_test_split(train_data, test_size=split_test_size, random_state=42)

    train_dataset = torch.from_numpy(X_train).float()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = torch.from_numpy(X_test).float()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def test_moe_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 10000

    model = BasicMOE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader, test_loader, n_train, n_test  = get_dataloaders(batch_size=batch_size)


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            x = batch[:,:-1]
            x.to(device)
            
            target = batch[:,-1]
            target.to(device)
            
            # forward pass
            output = model(x)
            # mse loss for prediction
            loss = F.mse_loss(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        scheduler.step()
        epoch_train_loss = running_loss / n_train

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[:,:-1]
                x.to(device)
                target = batch[:,-1]
                target.to(device)
                output = model(x)
                test_loss = F.mse_loss(output, target)
                running_test_loss += test_loss.item() * x.size(0)
        epoch_test_loss = running_test_loss / n_test
        if epoch % 10 == 0:
            log_line = f"Epoch {epoch}, Loss: {epoch_train_loss:.4f}" \
                f"(Test loss: {epoch_test_loss:.4f})"
            if epoch == 0:
                with open("./MoE_results/log.txt", "w") as f:
                    f.write(log_line)
            else:
                with open("./MoE_results/log.txt", "a") as f:
                    f.write("\n" + log_line)
            print(log_line)
            torch.save(model.state_dict(), f"./MoE_results/moe_model_epoch_{epoch}.pt")

if __name__ == "__main__":
    test_moe_training()