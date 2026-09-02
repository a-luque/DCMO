import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(data_dir="./data", batch_size=64, num_workers=4):
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  )

  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
  ])

  val_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
  ])

  train_dataset = datasets.Food101(
      root=data_dir, split="train", download=True, transform=train_transform
  )
  val_dataset = datasets.Food101(
      root=data_dir, split="test", download=True, transform=val_transform
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
      drop_last=True,
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_loader, val_loader, len(train_dataset), len(val_dataset)


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  epochs = 10000
  batch_size = 32
  base_lr = 0.05

  train_loader, val_loader, n_train, n_val = get_dataloaders(
      batch_size=batch_size
  )

  model = models.resnet101(weights=None)
  model.fc = nn.Linear(model.fc.in_features, 101)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = optim.SGD(
      model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4
  )
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

  for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(train_loader):
      images, targets = images.to(device, non_blocking=True), targets.to(
          device, non_blocking=True
      )

      optimizer.zero_grad()
      with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(images)
        loss = criterion(outputs, targets)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      running_loss += loss.item() * images.size(0)

    scheduler.step()
    epoch_train_loss = running_loss / n_train


    model.eval()
    top1_correct, top5_correct = 0, 0

    with torch.no_grad():
      for images, targets in val_loader:
        images, targets = images.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        outputs = model(images)
        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1_correct += correct[0].view(-1).float().sum().item()
        top5_correct += correct[:5].reshape(-1).float().sum().item()

    top1_acc = 100.0 * top1_correct / n_val
    top5_acc = 100.0 * top5_correct / n_val


if __name__ == "__main__":
  main()