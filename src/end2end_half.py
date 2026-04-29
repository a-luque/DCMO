import pandas as pd
import numpy as np
import os
import torch
import torchvision
import cv2
from tqdm import tqdm
from enum import Enum
import re
import shutil

class Weather(Enum):
    ClearNoon = [5.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    CloudyNoon = [60.0, 0.0, 0.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetNoon = [5.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudyNoon = [60.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainyNoon = [60.0, 60.0, 60.0, 60.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainNoon = [100.0, 100.0, 90.0, 100.0, -1.0, 45.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    SoftRainNoon = [20.0, 30.0, 50.0, 30.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    ClearSunset = [5.0, 0.0, 0.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    CloudySunset = [60.0, 0.0, 0.0, 10.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetSunset = [5.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudySunset = [60.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainSunset = [60.0, 60.0, 60.0, 60.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] 
    HardRainSunset = [100.0, 100.0, 90.0, 100.0, -1.0, 15.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    SoftRainSunset = [20.0, 30.0, 50.0, 30.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]


class End2End(torch.nn.Module):
    def __init__(self, context_dim,
                 num_experts, num_objectives, dropout_rate=0.1):
        super(End2End, self).__init__()

        # Layers
        self.layer1 = torch.nn.Linear(context_dim + num_experts, 128)
        self.leaky_relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(dropout_rate)

        self.layer2 = torch.nn.Linear(128, 256)
        self.leaky_relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(dropout_rate)

        self.layer3 = torch.nn.Linear(256, 128)
        self.leaky_relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(dropout_rate)

        self.layer4 = torch.nn.Linear(128, num_objectives)
        # print(num_experts)

    def forward(self, 
        controller,  # (batch, 9)
        context      # (batch, 16)
    ):
        assert controller.size()[1] == 9
        assert context.size()[1] == 16
        x = torch.cat([controller, context], axis=1)
        x = self.layer1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)

        # x = x.unsqueeze(0)
        x = torch.softmax(self.layer4(x), dim=1)
        return x


def train_E2E(
    train_dataset,
    val_dataset,
    save_path,
    batch_size=16,
    n_epochs=500,
    lr=0.0005,
    device="cuda",
    model_name="end2end",
):
    """
    Train a CNN on the given data
    """
    model = End2End(context_dim=16, num_experts=9, num_objectives=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=10
    )
    criterion = torch.nn.MSELoss()




    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    best_val_loss = np.inf
    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {params}", flush=True)
    with torch.no_grad():
        test_loss = 0
        for controller_batch, context_batch, y_batch in test_loader:
            y_batch = y_batch.to(device)
            y_pred = model(controller_batch.to(device), context_batch.to(device) )
            test_loss += criterion(y_pred, y_batch).item()
        test_loss /= len(test_loader)
    print(f"Epoch {-1}: val loss = {test_loss}", flush=True)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for loaded_batch in train_loader:
            controller_batch, context_batch, y_batch = loaded_batch
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_pred = model(controller_batch.to(device), context_batch.to(device) )
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for controller_batch, context_batch, y_batch in test_loader:
                y_batch = y_batch.to(device)
                y_pred = model(controller_batch.to(device), context_batch.to(device) )
                test_loss += criterion(y_pred, y_batch).item()
            test_loss /= len(test_loader)
        scheduler.step(test_loss)
        print(
            f"Epoch {epoch}: train loss = {total_loss / len(train_loader)} val loss = {test_loss}", flush=True
        )
        # save model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_{epoch}.pth"))
        if epoch % 50 == 0 or epoch == n_epochs-1:
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_{epoch}.pth"))

    return model


from torch.utils.data import Dataset
from torchvision.io import read_image

def load_empirical_validation_df(
    split_ratio=0.1,
    seed=42,
):
    # Load only 4 validation result_df
    dfs = []
    for i in range(1,5):
        df = pd.read_csv(f"/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/empirical_validation_{i}/results.csv", index_col=0)
        dfs += [df.copy()]
    df = pd.concat(dfs)

    # Get dummies from controllers with correct order
    controllers = list(df["controller"].unique())
    controllers.sort()
    cont_dict = {e: controllers.index(e) for e in controllers}
    df["controller"] = df["controller"].apply(lambda x: cont_dict[x])
    df = pd.get_dummies(df, columns=["controller"],dtype=float)

    # Get weather features
    df_weather = pd.DataFrame(df["weather"].apply(lambda x: Weather[x].value).tolist(), index=df.index, columns=[f"weather_feature_{i}" for i in range(14)]) 
    df = df.drop(["weather"], axis=1)
    data = pd.concat([df, df_weather], axis=1)


    # Randomize order
    data_size = len(data)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_sample = int(data_size * (1 - split_ratio))
    training_data = data.loc[:train_sample]
    val_data = data.loc[train_sample:]

    train_dataset = E2E_Dataset(
        dataframe=training_data
    )
    val_dataset = E2E_Dataset(
        dataframe=val_data
    )

    return train_dataset, val_dataset

class E2E_Dataset(Dataset):
    def __init__(self, dataframe):
        self.controller = dataframe[[f"controller_{i}" for i in range(9)]]
        self.context = dataframe[[f"weather_feature_{i}" for i in range(14)] + ["dist", "speed"]]
        
        self.labels = dataframe[["rew_sta", "rew_eff", "rew_safe"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        controller = self.controller.iloc[idx]
        context = self.context.iloc[idx]
        labels = self.labels.iloc[idx]

        return torch.FloatTensor(controller), torch.FloatTensor(context), torch.FloatTensor(labels)



if __name__ == "__main__":

    RESULTS_DIR = "/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/training_end2end_half"
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)

    print("-- Loading data", flush=True)
    train_dataset, val_dataset = load_empirical_validation_df()
    print("-- Training", flush=True)
    model = train_E2E(train_dataset, val_dataset, RESULTS_DIR)
    torch.save(model.state_dict(), os.path.join(save_path, "final.pth"))
