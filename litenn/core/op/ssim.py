import numpy as np
import litenn as nn
import litenn.core as nc

def ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """
    ssim operator.
    
    Computes per-channel structural similarity.
    """
    filter_size = int(filter_size)
    
    N1, CH1, H1, W1 = img1.shape
    N2, CH2, H2, W2 = img2.shape

    if N1 != N2:
        raise ValueError('Images batch must match.')
    if CH1 != CH2:
        raise ValueError('Images channels must match.')

    kernel_key = (ssim, CH1, filter_size, filter_sigma)
    kernel = nc.Cacheton.get_var(kernel_key)
    if kernel is None:
        kernel = np.arange(0, filter_size, dtype=np.float32)
        kernel -= (filter_size - 1 ) / 2.0
        kernel = kernel**2
        kernel *= ( -0.5 / (filter_sigma**2) )
        kernel = np.reshape (kernel, (1,-1)) + np.reshape(kernel, (-1,1) )        
        kernel_exp = np.exp(kernel)
        kernel = kernel_exp / kernel_exp.sum()
        kernel = kernel[None,...]
        kernel = np.tile (kernel, (CH1,1,1))
        nc.Cacheton.set_var(kernel_key, kernel)
        
    kernel_t = nn.Tensor_from_value(kernel)
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = nn.depthwise_conv2D(img1, kernel_t, stride=1, padding='valid')
    mean1 = nn.depthwise_conv2D(img2, kernel_t, stride=1, padding='valid')
    num0 = mean0 * mean1 * 2.0

    den0 = mean0 * mean0 + mean1 * mean1

    luminance = (num0 + c1) / (den0 + c1)
    
    num1 = nn.depthwise_conv2D( img1*img2, kernel_t, stride=1, padding='valid')* 2.0
    den1 = nn.depthwise_conv2D( img1*img1 + img2*img2, kernel_t, stride=1, padding='valid')

    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    return nn.reduce_mean(luminance * cs, axes=(-2,-1) )
    
def dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """
    dssim operator
    
    Computes per-channel structural dissimilarity.
    """
    ssim_val = ssim(img1, img2, max_val, filter_size, filter_sigma, k1, k2)
    return (1.0 - ssim_val ) / 2.0
    
def ssim_test():
    img1 = nn.Tensor( (4,3,64,64) )
    img2 = nn.Tensor( (4,3,64,64) )
    
    val = ssim(img1, img2, 1.0)
    val.backward(grad_for_non_trainables=True)
    
def dssim_test():
    img1 = nn.Tensor( (4,3,64,64) )
    img2 = nn.Tensor( (4,3,64,64) )
    
    val = dssim(img1, img2, 1.0)
    val.backward(grad_for_non_trainables=True)
    