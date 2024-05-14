import streamlit as st
from PIL import Image
import numpy as np
import torch
import numpy as np
import PIL
from torchvision import transforms
from skimage.color import lab2rgb
import sys
sys.path.append('../GANs/src')
from model import MainModel
from PIL import Image
import sys
sys.path.append('../GANs/src')
import cv2
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
import warnings
warnings.filterwarnings("ignore")

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def build_res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(DEVICE)
    return net_G

# Load the pre-trained cGAN model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("res18-unet.pt", map_location=DEVICE))
model = MainModel(net_G=net_G)
model.load_state_dict(
    torch.load(
        "final_model_weights.pt",
        map_location=DEVICE
    )
)
model.eval()

def colorize_image(image):
    # Preprocess the input image
    image = transforms.ToTensor()(image)[:1] * 2. - 1.
    with torch.no_grad():
        preds = model.net_G(image.unsqueeze(0).to(DEVICE))
    colorized = lab_to_rgb(image.unsqueeze(0), preds.cpu())[0]
    return colorized

# Streamlit app
st.set_page_config(page_title="Automatic Image Colorization", page_icon="🎨", layout="wide")
st.title("Automatic Image Colorization")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))  # Resize the input image to 256x256

    # Take a center crop and resize it to 256x256
    width, height = image.size
    left = (width - height) / 2
    top = (height - width) / 2
    right = (width + height) / 2
    bottom = (height + width) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((256, 256), resample=Image.BICUBIC)

    # Create a column to center the images
    center_column, _ = st.columns([1, 1])

    with center_column:
        # Create two columns to display the images side by side
        image_col1, image_col2 = st.columns(2)

        with image_col1:
            st.image(image, caption="Input Image", width=256, use_column_width=True)

        if st.button("Colorize"):
            colorized_image = colorize_image(image)

            with image_col2:
                st.image(colorized_image, caption="Colorized Image", width=256, use_column_width=True)