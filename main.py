from Resized import resize_images
from trainer import train
from DataLoaderSegmentation import DataLoaderSegmentation
import torch
from torch import nn
from unet import UNET
import torch.utils.data as data


path_list = ["path_to_train", "path_to_train_masks", "path_to_val", "path_to_val_masks"]
for path in path_list:
	resize_images(path, f"{path}_resize", size=512)

training_dataset = DataLoaderSegmentation('path', 'train_resize', 'masks_resize')
training_dataloader = data.DataLoader(dataset=training_dataset, batch_size=6, shuffle=True)

validation_dataset = DataLoaderSegmentation('path', 'val_resize', 'masks_val_resize')
validation_dataloader = data.DataLoader(dataset=validation_dataset, batch_size=6, shuffle=True)

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(unet.parameters(), lr=0.001)

#Two class - sand and background
unet = UNET(3,2)
train_loss, valid_loss = train(unet, training_dataloader, dataloader_validation, loss, opt, acc_metric, epochs=50)
