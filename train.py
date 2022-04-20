import torch
from dataset import CatDogDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train_fn(disc_D, disc_C, gen_C, gen_D, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)
    for idx, (cat, dog) in enumerate(loop):
        cat = cat.to(config.DEVICE)
        dog = dog.to(config.DEVICE)

        # train discriminator
        with torch.cuda.amp.autocast():
            fake_dog = gen_D(dog)
            D_D_real = disc_D(dog)
            D_D_fake = disc_D(fake_dog.detach())
            D_D_real_loss = mse(D_D_real, torch.ones_like(D_D_real))
            D_D_fake_loss = mse(D_D_fake, torch.ones_like(D_D_fake))
            D_D_loss = D_D_real_loss + D_D_fake_loss

            fake_cat = gen_C(cat)
            D_C_real = disc_C(cat)
            D_C_fake = disc_C(fake_cat.detach())
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.ones_like(D_C_fake))
            D_C_loss = D_C_real_loss + D_C_fake_loss

            D_loss = (D_D_loss + D_C_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generator
        with torch.cuda.amp.autocast():
            D_D_fake = disc_D(fake_dog)
            D_C_fake = disc_C(fake_cat)
            loss_G_D = mse(D_D_fake, torch.ones_like(D_D_fake))
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))

            # cycle loss
            cycle_cat = gen_C(fake_cat)
            cycle_dog = gen_D(fake_dog)
            cycle_cat_loss = l1(cat, cycle_cat)
            cycle_dog_loss = l1(dog, cycle_dog)

            # identity loss
            identity_cat = gen_C(cat)
            identity_dog = gen_D(dog)
            identity_cat_loss = l1(cat, identity_cat)
            identity_dog_loss = l1(dog, identity_dog)

            # add all together
            G_loss = (
                loss_G_C +
                loss_G_D +
                cycle_cat_loss * config.LAMBDA_CYCLE +
                cycle_dog_loss * config.LAMBDA_CYCLE +
                identity_cat_loss * config.LAMBDA_IDENTITY +
                identity_dog_loss * config.LAMBDA_IDENTITY
            )
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_dog * 0.5 + 0.5, f'saved_images/dog_{idx}.png')
            save_image(fake_cat * 0.5 + 0.5, f'saved_images/cat_{idx}.png')

def main():
    disc_C = Discriminator(in_channels=3).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_C.parameters()) + list(disc_D.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_D.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_D, gen_D, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_CRITIC_D, disc_D, opt_disc, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_CRITIC_C, disc_C, opt_disc, config.LEARNING_RATE)
    dataset = CatDogDataset(root_cat=config.TRAIN_DIR + "/cat", root_dog=config.TRAIN_DIR + "/dog", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    g_scale = torch.cuda.amp.GradScaler()
    d_scale = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_D, disc_C, gen_C, gen_D, loader, opt_disc, opt_gen, L1, mse, d_scale, g_scale)
        if config.SAVE_MODEL:
            save_checkpoint(gen_D, opt_gen, filename=config.CHECKPOINT_GEN_D)
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C)
            save_checkpoint(disc_D, opt_disc, filename=config.CHECKPOINT_CRITIC_D)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_CRITIC_C)

if __name__ == '__main__':
    main()
