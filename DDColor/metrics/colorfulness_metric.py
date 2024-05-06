import torch

def calculate_cf(tensor):
    """Calculate Colorfulness for a tensor of images.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Colorfulness value for each image in the batch.
    """
    total_colorfulness = 0.0
    for i in range(tensor.shape[0]):
        R, G, B = tensor[i][0], tensor[i][1], tensor[i][2]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        rb_mean = torch.mean(rg)
        rb_std = torch.std(rg)
        yb_mean = torch.mean(yb)
        yb_std = torch.std(yb)
        std_root = torch.sqrt((rb_std ** 2) + (yb_std ** 2))
        mean_root = torch.sqrt((rb_mean ** 2) + (yb_mean ** 2))
        colorfulness = std_root + (0.3 * mean_root)
        total_colorfulness += colorfulness
    
    return total_colorfulness




