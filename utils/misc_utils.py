import torch
import torchvision as tv
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imsave, imread



# def imshow2D(img, blocking=False):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img[0, 0, ...].float().detach().cpu().numpy())
#     if blocking:
#         plt.show()


# def imshow3D(vol, blocking=False):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(volume_2_projections(vol.permute(0, 2, 3, 1).unsqueeze(1), normalize=True)[
#                    0, 0, ...].float().detach().cpu().numpy())
#     if blocking:
#         plt.show()


# def imshowComplex(vol, blocking=False):
#     plt.figure(figsize=(10, 10))
#     plt.subplot(1, 2, 1)
#     plt.imshow(volume_2_projections(torch.real(vol).permute(0, 2, 3, 1).unsqueeze(1))[
#                    0, 0, ...].float().detach().cpu().numpy())
#     plt.subplot(1, 2, 2)
#     plt.imshow(volume_2_projections(torch.imag(vol).permute(0, 2, 3, 1).unsqueeze(1))[
#                    0, 0, ...].float().detach().cpu().numpy())
#     if blocking:
#         plt.show()


def save_image(tensor, path='output.png'):
    if 'tif' in path:
        imsave(path, tensor[0, ...].cpu().numpy().astype(np.float32))
        return
    if tensor.shape[1] == 1:
        imshow2D(tensor)
    else:
        imshow3D(tensor)
    plt.savefig(path)


# # Aid functions for shiftfft2
# def roll_n(X, axis, n):
#     f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
#     b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
#     front = X[f_idx]
#     back = X[b_idx]
#     return torch.cat([back, front], axis)


# def batch_fftshift2d_real(x):
#     out = x
#     for dim in range(2, len(out.size())):
#         n_shift = x.size(dim) // 2
#         if x.size(dim) % 2 != 0:
#             n_shift += 1  # for odd-sized images
#         out = roll_n(out, axis=dim, n=n_shift)
#     return out


# # FFT convolution, the kernel fft can be precomputed
# def fft_conv(A, B, fullSize, Bshape=[], B_precomputed=False):
#     import torch.fft
#     nDims = A.ndim - 2
#     # fullSize = torch.tensor(A.shape[2:]) + Bshape
#     # fullSize = torch.pow(2, torch.ceil(torch.log(fullSize.float())/torch.log(torch.tensor(2.0)))-1)
#     padSizeA = (fullSize - torch.tensor(A.shape[2:]))
#     padSizesA = torch.zeros(2 * nDims, dtype=int)
#     padSizesA[0::2] = torch.floor(padSizeA / 2.0)
#     padSizesA[1::2] = torch.ceil(padSizeA / 2.0)
#     padSizesA = list(padSizesA.numpy()[::-1])

#     A_padded = F.pad(A, padSizesA)
#     Afft = torch.fft.rfft2(A_padded)
#     if B_precomputed:
#         return batch_fftshift2d_real(torch.fft.irfft2(Afft * B.detach()))
#     else:
#         padSizeB = (fullSize - torch.tensor(B.shape[2:]))
#         padSizesB = torch.zeros(2 * nDims, dtype=int)
#         padSizesB[0::2] = torch.floor(padSizeB / 2.0)
#         padSizesB[1::2] = torch.ceil(padSizeB / 2.0)
#         padSizesB = list(padSizesB.numpy()[::-1])
#         B_padded = F.pad(B, padSizesB)
#         Bfft = torch.fft.rfft2(B_padded)
#         return batch_fftshift2d_real(torch.fft.irfft2(Afft * Bfft.detach())), Bfft.detach()


# def reprojection_loss_camera(gt_imgs, prediction, PSF, camera, dataset, device="cpu"):
#     out_type = gt_imgs.type()
#     camera = camera.to(device)
#     reprojection = camera(prediction.to(device), PSF.to(device))
#     reprojection_views = dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0, 0, ...]
#     loss = F.mse_loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

#     return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)


# def reprojection_loss(gt_imgs, prediction, OTF, psf_shape, dataset, n_split=20, device="cpu", loss=F.mse_loss):
#     out_type = gt_imgs.type()
#     batch_size = prediction.shape[0]
#     reprojection = fft_conv_split(prediction[0, ...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True,
#                                   device=device)

#     reprojection_views = torch.zeros_like(gt_imgs)
#     reprojection_views[0, ...] = dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[
#         0, 0, ...]

#     # full_reprojection = reprojection.detach()
#     # reprojection_views = reprojection_views.unsqueeze(0).repeat(batch_size,1,1,1)
#     for nSample in range(1, batch_size):
#         reprojection = fft_conv_split(prediction[nSample, ...].unsqueeze(0), OTF, psf_shape, n_split,
#                                       B_precomputed=True, device=device)
#         reprojection_views[nSample, ...] = \
#             dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0, 0, ...]
#         # full_reprojection += reprojection.detach()

#     # gt_imgs /= gt_imgs.float().max()
#     # reprojection_views /= reprojection_views.float().max()
#     # loss = F.mse_loss(gt_imgs[gt_imgs!=0].to(device), reprojection_views[gt_imgs!=0])
#     # loss = (1-gt_imgs[reprojection_views!=0]/reprojection_views[reprojection_views!=0]).abs().mean()
#     loss = loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

#     return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)


# # Split an fft convolution into batches containing different depths
# def fft_conv_split(A, B, psf_shape, n_split, B_precomputed=False, device="cpu"):
#     n_depths = A.shape[1]

#     split_conv = n_depths // n_split
#     depths = list(range(n_depths))
#     depths = [depths[i:i + split_conv] for i in range(0, n_depths, split_conv)]

#     fullSize = torch.tensor(A.shape[2:]) + psf_shape

#     crop_pad = [(psf_shape[i] - fullSize[i]) // 2 for i in range(0, 2)]
#     crop_pad = (crop_pad[1], (psf_shape[-1] - fullSize[-1]) - crop_pad[1], crop_pad[0],
#                 (psf_shape[-2] - fullSize[-2]) - crop_pad[0])
#     # Crop convolved image to match size of PSF
#     img_new = torch.zeros(A.shape[0], 1, psf_shape[0], psf_shape[1], device=device)
#     if B_precomputed == False:
#         OTF_out = torch.zeros(1, n_depths, fullSize[0], fullSize[1] // 2 + 1, requires_grad=False,
#                               dtype=torch.complex64, device=device)
#     for n in range(n_split):
#         # print(n)
#         curr_psf = B[:, depths[n], ...].to(device)
#         img_curr = fft_conv(A[:, depths[n], ...].to(device), curr_psf, fullSize, psf_shape, B_precomputed)
#         if B_precomputed == False:
#             OTF_out[:, depths[n], ...] = img_curr[1]
#             img_curr = img_curr[0]
#         img_curr = F.pad(img_curr, crop_pad)
#         img_new += img_curr[:, :, :psf_shape[0], :psf_shape[1]].sum(1).unsqueeze(1).abs()

#     if B_precomputed == False:
#         return img_new, OTF_out
#     return img_new


# def imadjust(x, a, b, c, d, gamma=1):
#     # Similar to imadjust in MATLAB.
#     # Converts an image range from [a,b] to [c,d].
#     # The Equation of a line can be used for this transformation:
#     #   y=((d-c)/(b-a))*(x-a)+c
#     # However, it is better to use a more generalized equation:
#     #   y=((x-a)/(b-a))^gamma*(d-c)+c
#     # If gamma is equal to 1, then the line equation is used.
#     # When gamma is not equal to 1, then the transformation is not linear.

#     y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
#     mask = (y > 0).float()
#     y = torch.mul(y, mask)
#     return y


# Apply different normalizations to images
def normalize_type(LF_views, id=0, mean_imgs=0, std_imgs=1, max_imgs=1, inverse=False):
    if inverse:
        if id == -1:  # No normalization
            return LF_views,
        if id == 0:  # baseline normalization
            return (LF_views) * (2 * std_imgs)
        if id == 1:  # Standardization of images and normalization
            return LF_views * std_imgs + mean_imgs
        if id == 2:  # normalization
            return LF_views * max_imgs
        if id == 3:  # normalization
            return LF_views * std_imgs
    else:
        if id == -1:  # No normalization
            return LF_views
        if id == 0:  # baseline normalization
            return LF_views / (2 * std_imgs)
        if id == 1:  # Standardization of images normalization
            return (LF_views - mean_imgs) / std_imgs
        if id == 2:  # normalization
            return LF_views / max_imgs
        if id == 3:  # normalization
            return LF_views / std_imgs


# def plot_param_grads(writer, net, curr_it, prefix=""):
#     for tag, parm in net.named_parameters():
#         if parm.grad is not None:
#             writer.add_histogram(prefix + tag, parm.grad.data.cpu().numpy(), curr_it)
#             assert not torch.isnan(parm.grad.sum()), print("NAN in: " + str(tag) + "\t\t")


# def compute_histograms(gt, pred, input_img, n_bins=1000):
#     volGTHist = torch.histc(gt, bins=n_bins, max=gt.max().item())
#     volPredHist = torch.histc(pred, bins=n_bins, max=pred.max().item())
#     inputHist = torch.histc(input_img, bins=n_bins, max=input_img.max().item())
#     return volGTHist, volPredHist, inputHist


# def match_histogram(source, reference):
#     isTorch = False
#     source = source / source.max() * reference.max()
#     if isinstance(source, torch.Tensor):
#         source = source.cpu().numpy()
#         isTorch = True
#     if isinstance(reference, torch.Tensor):
#         reference = reference[:source.shape[0], ...].cpu().numpy()

#     matched = match_histograms(source, reference, multichannel=False)
#     if isTorch:
#         matched = torch.from_numpy(matched)
#     return matched


# def get_number_of_frames(name):
#     n_frames = re.match(r"^.*_(\d*)timeF", name)
#     if n_frames is not None:
#         n_frames = int(n_frames.groups()[0])
#     else:
#         n_frames = 1
#     return n_frames


# def net_get_params(net):
#     if hasattr(net, 'module'):
#         return net.module
#     else:
#         return net


# def center_crop(layer, target_size, pad=0):
#     _, _, layer_height, layer_width = layer.size()
#     diff_y = (layer_height - target_size[0]) // 2
#     diff_x = (layer_width - target_size[1]) // 2
#     return layer[
#            :, :, (diff_y - pad): (diff_y + target_size[0] - pad), (diff_x - pad): (diff_x + target_size[1] - pad)
#            ]
