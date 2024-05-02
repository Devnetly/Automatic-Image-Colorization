import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ddcolor_arch import DDColor
from losses.losses import L1Loss, PerceptualLoss, GANLoss, ColorfulnessLoss
from custom_data_loader import ColorizationDataset
from metrics.colorfulness_metric import calculate_cf
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
from save_model_checkpoints import save_model




def train_step(device, model, optimizer, train_dataloader, pixel_loss_fn, gan_loss_fn, colorfulness_loss_fn, perceptual_loss):
    model.train() 
    model.to(device)

    train_loss = 0.0


    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training')
    for  batch, (rgb_image, orig_l, orig_ab, img_gray_rgb) in progress_bar:
        rgb_image = rgb_image.to(device)
        orig_l = orig_l.to(device)
        orig_ab = orig_ab.to(device)
        img_gray_rgb = img_gray_rgb.to(device)


        # Forward pass
        output_ab = model(img_gray_rgb)
        output_ab = output_ab.permute(0, 3, 2, 1)

        # convert to lab
        output_lab = torch.cat((orig_l, output_ab), dim=-1)
        
        # convert to rgb 
        output_rgb_batch = []
        for i in range(output_lab.shape[0]):
            output_lab_i = output_lab[i].detach().cpu().numpy() 
            output_rgb_i = cv2.cvtColor(output_lab_i, cv2.COLOR_LAB2RGB)
            output_rgb_batch.append(output_rgb_i)
            
            
        
        # predicted image tensor in rgb format
        output_rgb = np.stack(output_rgb_batch, axis=0)
        output_rgb = torch.tensor(output_rgb)

        # Calculate losses
        pixel_loss = pixel_loss_fn(output_ab.permute(0,3,2,1), orig_ab.permute(0,3,2,1))
        colorfulness_loss = colorfulness_loss_fn(output_rgb.permute(0,3,2,1))
        percep_loss, _ = perceptual_loss(output_rgb.permute(0,3,2,1), rgb_image.permute(0,3,2,1))
        gan_loss = gan_loss_fn(output_rgb.permute(0,3,2,1),target_is_real=True)

        total_loss = pixel_loss + colorfulness_loss + percep_loss + gan_loss
        train_loss += total_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    
    train_loss = train_loss / len(train_dataloader)
        

    return train_loss







def validation_step(device, model, val_dataloader, pixel_loss_fn, gan_loss_fn, colorfulness_loss_fn, perceptual_loss):
    model.eval()  
    val_loss = 0.0
    total_colorfulness = 0.0  

    progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Validation')
    with torch.no_grad():
        for batch, (rgb_image, orig_l, orig_ab, img_gray_rgb) in progress_bar:

            rgb_image = rgb_image.to(device)
            orig_l = orig_l.to(device)
            orig_ab = orig_ab.to(device)
            img_gray_rgb = img_gray_rgb.to(device)


            # Forward pass
            output_ab = model(img_gray_rgb)
            output_ab = output_ab.permute(0, 3, 2, 1)

            # convert to lab
            output_lab = torch.cat((orig_l, output_ab), dim=-1)
        
            # convert to rgb 
            output_rgb_batch = []
            for i in range(output_lab.shape[0]):
                output_lab_i = output_lab[i].detach().cpu().numpy() 
                output_rgb_i = cv2.cvtColor(output_lab_i, cv2.COLOR_LAB2RGB)
                output_rgb_batch.append(output_rgb_i)
                
           
        
            # predicted image tensor in rgb format
            output_rgb = np.stack(output_rgb_batch, axis=0)
            output_rgb = torch.tensor(output_rgb)

            # Calculate losses
            pixel_loss = pixel_loss_fn(output_ab.permute(0,3,2,1), orig_ab.permute(0,3,2,1))
            colorfulness_loss = colorfulness_loss_fn(output_rgb.permute(0,3,2,1))
            percep_loss, _ = perceptual_loss(output_rgb.permute(0,3,2,1), rgb_image.permute(0,3,2,1))
            gan_loss = gan_loss_fn(output_rgb.permute(0,3,2,1),target_is_real=True)

            total_loss = pixel_loss + colorfulness_loss + percep_loss + gan_loss
            val_loss += total_loss

            

            # calculate colorfulness 
            colorfulness = calculate_cf(output_rgb.permute(0,3,2,1))
            total_colorfulness += colorfulness
            
                   


        val_loss = val_loss / len(val_dataloader)
        val_colorfulness = total_colorfulness / len(val_dataloader)
        

        return val_loss, val_colorfulness









def train(device, model, optimizer, train_dataloader, val_dataloader, num_epochs, pixel_loss_fn, gan_loss_fn, colorfulness_loss_fn, perceptual_loss):

    
  results = {"train_loss": [], "val_loss": [], "val_colorfulness": []} 
  
  for epoch in range(num_epochs):
       
      train_loss = train_step(device, model, optimizer, train_dataloader, pixel_loss_fn, gan_loss_fn, colorfulness_loss_fn, perceptual_loss)

      val_loss, val_colorfulness = validation_step(device, model, val_dataloader, pixel_loss_fn, gan_loss_fn, colorfulness_loss_fn, perceptual_loss)

         
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          
          f"val_loss: {val_loss:.4f} | ",
           f"val_colorfulness: {val_colorfulness:.4f} | "
           
        )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["val_loss"].append(val_loss)
      results["val_colorfulness"].append(val_colorfulness)
     
  
  return results




def main():

    # Define hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.99
    weight_decay = 0.01

    # Dataloader
    train_dataset = ColorizationDataset(data_dir='/home/meriem-mk/Downloads/DDColor/test-data', input_size=256)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ColorizationDataset(data_dir='/home/meriem-mk/Downloads/DDColor/test-data', input_size=256)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = DDColor(
        encoder_name='convnext-t', 
        decoder_name='MultiScaleColorDecoder',  
        num_input_channels=3,  
        input_size=(256, 256),
        nf=512,
        num_output_channels=2,  
        last_norm='Weight',
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
        encoder_from_pretrain=True
    )

    model.to(device)

    # Define losses
    pixel_loss_fn = L1Loss(loss_weight=0.1)
    gan_loss_fn = GANLoss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0)
    colorfulness_loss_fn = ColorfulnessLoss(loss_weight=0.5)

    layer_weights = {'conv1_1': 1.0, 'conv2_1': 1.0, 'conv3_1': 1.0, 'conv4_1': 1.0, 'conv5_1': 1.0}
    perceptual_loss = PerceptualLoss(layer_weights=layer_weights, vgg_type='vgg19', perceptual_weight=5.0, style_weight=0.0)

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    # Train the model
    results = train(device, model, optimizer, train_dataloader, val_dataloader, num_epochs, pixel_loss_fn, gan_loss_fn, colorfulness_loss_fn, perceptual_loss)

    save_model(model=model,
                 target_dir="ddcolor_model_checkpoints",
                 model_name="model.pth")
    



if __name__ == "__main__":
    main()











