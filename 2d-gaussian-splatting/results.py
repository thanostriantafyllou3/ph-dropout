import torch 
import numpy as np
import os
from PIL import Image
from argparse import ArgumentParser, Namespace
from torchmetrics.functional import structural_similarity_index_measure as ssim
from scipy.stats import pearsonr, spearmanr, norm
import matplotlib.pyplot as plt
from tqdm import tqdm


####################################################################################################
#####                                        DATA LOADING                                      #####
####################################################################################################

def get_images_from_dir(images_dir, device, normalize=True, max_samples=float('inf')):
    if not os.path.exists(images_dir):
        return torch.Tensor([]).to(device)
    images = []
    images_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    for file in images_files:
        if len(images) >= max_samples:
            break
        img_path = os.path.join(images_dir, file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.Tensor(np.array(img)).to(device)
        images.append(img_tensor)
    images = torch.stack(images).to(device)

    # Normalize images between 0 and 1
    if normalize and images.max() > 1:
        images = images / 255.0
    
    return images


def get_images_from_dir_of_dirs(images_dir, device, normalize=True, max_samples=float('inf')):
    if not os.path.exists(images_dir):
        return torch.Tensor([]).to(device)
    mean_images = []
    std_images = []
    images_dirs = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
    for d in tqdm(images_dirs):
        image_samples = get_images_from_dir(os.path.join(images_dir, d), device, normalize, max_samples)
        mean_images.append(torch.mean(image_samples, dim=0))
        std_images.append(torch.std(image_samples, dim=0))

        # Clear cache after processing each directory
        torch.cuda.empty_cache()

    mean_images = torch.stack(mean_images).to(device)
    std_images = torch.stack(std_images).to(device)

    return mean_images, std_images


def get_dir_paths_gs(model_dir, num_iterations, ssim_decr_margin=None, dropout_ratio=None):
    if ssim_decr_margin is not None and dropout_ratio is not None:
        raise ValueError("Both dropout_ratio and ssim_decr_margin cannot be set at the same time.")
    if ssim_decr_margin is not None:
        train_dir = model_dir + f"/train/ours_{num_iterations}_ssim_decr_{ssim_decr_margin}/"
        test_dir = model_dir + f"/test/ours_{num_iterations}_ssim_decr_{ssim_decr_margin}/"
        post_fix = f"ssim_decr_margin_{ssim_decr_margin}"
        
        # Read dropout ratio used for rendering
        if os.path.exists(train_dir + "dropout_ratio.txt"):
            with open(train_dir + "dropout_ratio.txt", "r") as f:
                dropout_ratio = float(f.read())
        elif os.path.exists(test_dir + "dropout_ratio.txt"):
            with open(test_dir + "dropout_ratio.txt", "r") as f:
                dropout_ratio = float(f.read())
        else:
            raise ValueError("Dropout ratio file not found.")

    elif dropout_ratio is not None:
        train_dir = model_dir + f"/train/ours_{num_iterations}_dropout_{dropout_ratio}/"
        test_dir = model_dir + f"/test/ours_{num_iterations}_dropout_{dropout_ratio}/"
        post_fix = f"dropout_ratio_{dropout_ratio}"
    else:
        raise ValueError("Either dropout_ratio or ssim_decr_margin must be set.")
    return train_dir, test_dir, dropout_ratio, post_fix


def get_dir_paths_nerf(model_dir, num_iterations, ssim_decr_margin=None, dropout_ratio=None):
    if ssim_decr_margin is not None and dropout_ratio is not None:
        raise ValueError("Both dropout_ratio and ssim_decr_margin cannot be set at the same time.")
    if ssim_decr_margin is not None:
        test_dir = model_dir + f"/renderonly_test_0{num_iterations-1}_ssim_decr_margin_{ssim_decr_margin}/"
        post_fix = f"ssim_decr_margin_{ssim_decr_margin}"
        
        # Read dropout ratio used for rendering
        if os.path.exists(test_dir):
            with open(test_dir + "dropout_ratio.txt", "r") as f:
                # Read three lines in the format
                lines = f.readlines()
                dropout_ratio = float(lines[0].split('=')[-1].strip())
                # NOTE: These are computed for only 1 trial per view
                avg_train_ssim_no_dropout = float(lines[1].split('=')[-1].strip())
                avg_train_ssim = float(lines[2].split('=')[-1].strip())
        else:
            raise ValueError("Dropout ratio file not found.")
    elif dropout_ratio is not None:
        test_dir = model_dir + f"/renderonly_test_0{num_iterations-1}_dropoutratio_{dropout_ratio}/"
        post_fix = f"dropout_ratio_{dropout_ratio}"
    else:
        raise ValueError("Either dropout_ratio or ssim_decr_margin must be set.")
    return test_dir, dropout_ratio, avg_train_ssim_no_dropout, avg_train_ssim, post_fix


def get_images_gs(images_dir, device, normalize=True, max_samples=float('inf'), gt_only=False):
    # Get all the ground truth images
    gt_dir = os.path.join(images_dir, 'gt')
    gt_images = get_images_from_dir(gt_dir, device) # [num_images, height, width, channels]
    if gt_only:
        return gt_images, None, None

    # Get all the rendered images
    renders_dir = os.path.join(images_dir, 'renders')
    mean_images, std_images = get_images_from_dir_of_dirs(renders_dir, device, normalize=normalize, max_samples=max_samples) # [num_images, num_samples, height, width, channels]
    
    return gt_images, mean_images, std_images


def get_images_nerf(renders_dir, device, normalize=True, max_samples=float('inf'), gt_only=False):
    # Get all the ground truth images
    if "train" in renders_dir:
        gt_dir = os.path.join(renders_dir, '../trainset')
    elif "test" in renders_dir:
        gt_dir = os.path.join(renders_dir, '../testset')
    else:
        raise NotImplementedError("Only train and test directories are supported.")
    gt_images = get_images_from_dir(gt_dir, device) # [num_images, height, width, channels]
    if gt_only:
        return gt_images, None, None

    # Get all the rendered images
    mean_images, std_images = get_images_from_dir_of_dirs(renders_dir, device, normalize=normalize, max_samples=max_samples) # [num_images, num_samples, height, width, channels]
    
    return gt_images, mean_images, std_images


####################################################################################################
#####                                     PREDICTION ERROR                                     #####
####################################################################################################

def get_rmse_images(gt_images, pred_images):
    mse_images = (pred_images - gt_images) ** 2
    return torch.sqrt(mse_images)

####################################################################################################
#####                                         METRICS                                          #####
####################################################################################################


def compute_ssim(gt_image, pred_image):
    # SSIM expects the input to be in the format [batch, channels, height, width]
    gt_image = gt_image.permute(2, 0, 1).unsqueeze(0) # [1, channels, height, width]
    pred_image = pred_image.permute(2, 0, 1).unsqueeze(0)
    data_range = gt_image.max() - gt_image.min()
    return ssim(pred_image, gt_image, data_range=data_range).item()


def compute_ssims(gt_images, pred_images):
    return torch.Tensor([compute_ssim(gt_image, pred_image) for gt_image, pred_image in zip(gt_images, pred_images)])


def compute_psnr(gt_image, pred_image):
    mse = torch.mean((gt_image - pred_image) ** 2)
    max_ = 1 if gt_image.max() <= 1 else 255
    return 20 * torch.log10(torch.tensor(max_).to(gt_image.device)) - 10 * torch.log10(mse)


def compute_psnrs(gt_images, pred_images):
    return torch.Tensor([compute_psnr(gt_image, pred_image) for gt_image, pred_image in zip(gt_images, pred_images)])


def compute_correration(error_image, std_image, type='pearson', error_only=False):
    error_flat = error_image.flatten().cpu().numpy()
    std_flat = std_image.flatten().cpu().numpy()
    idx = np.where(error_flat != 0) if error_only else np.arange(len(error_flat))
    return pearsonr(error_flat[idx], std_flat[idx])[0] if type == 'pearson' else spearmanr(error_flat[idx], std_flat[idx])[0]
    
def compute_correlations(error_images, std_images, type='pearson', error_only=False):
    return torch.Tensor([compute_correration(error_image, std_image, type, error_only) for error_image, std_image in zip(error_images, std_images)])


def compute_ause(error_image, std_image):
    """
    Ref. to https://arxiv.org/pdf/2203.10192 (Section 5, Metrics):
    Concretely, given an error metric (e.g. RMSE), 
    we obtain two lists by sorting the values of all the pixels according to their uncertainty and the error computed from the ground-truth. 
    By removing the top t%(t = 1 ∼ 100) of the errors in each vector and repeatedly computing the average of the last subset, 
    we can obtain the sparsification curve and the oracle curve respectively. 
    The area between them is the AUSE, which evaluates how much the uncertainty is correlated with the predicted error.
    """
    errors, uncertainties = error_image.flatten(), std_image.flatten()
    sort_idx_uncertainty, sort_idx_error = torch.argsort(uncertainties), torch.argsort(errors)
    sparsification_curve = torch.Tensor([torch.mean(errors[sort_idx_uncertainty][:int(len(errors)*(1-t/100))]) for t in range(1, 100)])
    oracle_curve = torch.Tensor([torch.mean(errors[sort_idx_error][:int(len(errors)*(1-t/100))]) for t in range(1, 100)])
    ause = torch.trapz(sparsification_curve) - torch.trapz(oracle_curve)
    return ause.item()

def compute_auses(error_images, std_images):
    return torch.Tensor([compute_ause(error_image, std_image) for error_image, std_image in zip(error_images, std_images)])


def compute_ece(error_image, std_image, num_bins=10):
    errors, uncertainties = error_image.flatten(), std_image.flatten()
    bin_edges = torch.linspace(0, uncertainties.max(), num_bins + 1, device=uncertainties.device)
    total_points = errors.numel()
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_edges[:-1], bin_edges[1:]):
        bin_idx = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
        if torch.any(bin_idx):
            avg_uncertainty = torch.mean(uncertainties[bin_idx])
            avg_error = torch.mean(errors[bin_idx])
            ece += torch.abs(avg_uncertainty - avg_error) * bin_idx.sum().float() / total_points
    return ece

def compute_eces(error_images, std_images, num_bins=10):
    return torch.Tensor([compute_ece(error_image, std_image, num_bins) for error_image, std_image in zip(error_images, std_images)])


def compute_nll(gt_image, mean_image, std_image, eps=1e-2):
    std_image = torch.clamp(std_image, min=eps)
    return torch.mean(0.5 * torch.log(2 * torch.pi * std_image ** 2) + ((gt_image - mean_image) ** 2) / (2 * std_image ** 2))

def compute_nlls(gt_images, mean_images, std_images):
    return torch.Tensor([compute_nll(gt, mean, std) for gt, mean, std in zip(gt_images, mean_images, std_images)])


def compute_sharpnesses(std_images):
    return torch.Tensor([torch.mean(std_image) for std_image in std_images])


def compute_prediction_interval(gt_image, mean_image, std_image, confidence=0.95):
    z = norm.ppf((1 + confidence) / 2)
    lower_bound, upper_bound = mean_image - z * std_image, mean_image + z * std_image
    return torch.mean(((gt_image >= lower_bound) & (gt_image <= upper_bound)).float())

def compute_prediction_intervals(gt_images, mean_images, std_images, confidence=0.95):
    return torch.Tensor([compute_prediction_interval(gt, mean, std, confidence) for gt, mean, std in zip(gt_images, mean_images, std_images)])

def compute_max_stds(std_images):
    return torch.Tensor([torch.max(std_image) for std_image in std_images])


####################################################################################################
#####                                           MAIN                                           #####
####################################################################################################

def _results(gt_images, mean_images, std_images, output_dir, args, train_test='test'):
    rmse_images = get_rmse_images(gt_images, mean_images)

    if args.verbose:
        print(f"Processed images:   Mean:  {mean_images.shape},   STD:  {std_images.shape},   RMSE:  {rmse_images.shape}")
    
    if args.save_images:
        if args.verbose:
            print(f"Saving images to {output_dir}")
        os.makedirs(os.path.join(output_dir, f'gt_{train_test}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'mean_{train_test}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'std_{train_test}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'rmse_{train_test}'), exist_ok=True)
        for i, (gt, mean, rmse, std) in enumerate(zip(gt_images, mean_images, rmse_images, std_images)):
            Image.fromarray((gt.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(output_dir, f'gt_{train_test}', f"{i}.png"))
            Image.fromarray((mean.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(output_dir, f'mean_{train_test}', f"{i}.png"))
            # Save std and error image with hot colormap
            colormap = plt.get_cmap('hot')
            std = torch.mean(std, dim=-1) # [H, W, RGB] -> [H, W]
            std = std.cpu().numpy()
            std = (std - std.min()) / (std.max() - std.min())
            std = (colormap(std) * 255).astype(np.uint8)
            Image.fromarray(std).save(os.path.join(output_dir, f'std_{train_test}', f"{i}.png"))
            rmse = torch.mean(rmse, dim=-1) # [H, W, RGB] -> [H, W]
            rmse = rmse.cpu().numpy()
            rmse = (rmse - rmse.min()) / (rmse.max() - rmse.min())
            rmse = (colormap(rmse) * 255).astype(np.uint8)
            Image.fromarray(rmse).save(os.path.join(output_dir, f'rmse_{train_test}', f"{i}.png"))

    psnr = compute_psnrs(gt_images, mean_images)
    ssim = compute_ssims(gt_images, mean_images)
    spearman = compute_correlations(rmse_images, std_images, type='spearman')
    pearson = compute_correlations(rmse_images, std_images, type='pearson')
    ause = compute_auses(rmse_images, std_images)
    ece = compute_eces(rmse_images, std_images)
    nll = compute_nlls(gt_images, mean_images, std_images)
    sharpness = compute_sharpnesses(std_images)
    pred_interval = compute_prediction_intervals(gt_images, mean_images, std_images)
    max_std = compute_max_stds(std_images)

    # Store average metrics in a dictionary
    metrics = {
        "psnr": torch.mean(psnr).item(),
        "ssim": torch.mean(ssim).item(),
        "spearman": torch.mean(spearman).item(),
        "pearson": torch.mean(pearson).item(),
        "ause": torch.mean(ause).item(),
        "ece": torch.mean(ece).item(),
        "nll": torch.mean(nll).item(),
        "sharpness": torch.mean(sharpness).item(),
        "pred_interval": torch.mean(pred_interval).item(),
        "avg_max_std": torch.mean(max_std).item()
    }

    if args.verbose:
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value}")
    
    # Save results to 'metrics'.txt
    results_file = os.path.join(output_dir, "metrics.txt")
    with open(results_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()} = {value}\n")
    
    return metrics


def results(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.verbose:
        print(f"Model directory: {args.model_dir}")
        print(f"Number of iterations: {args.num_iterations}")
        print(f"Using device: {device}")

    # Check if model directory corresponds to GS or NeRF
    if os.path.exists(args.model_dir + "/point_cloud") and os.path.exists(args.model_dir + "/train") and os.path.exists(args.model_dir + "/test"):
        if args.num_iterations == -1:
            args.num_iterations = 30000 # defaulf for GS

        train_dir, test_dir, dropout_ratio, post_fix = get_dir_paths_gs(args.model_dir, args.num_iterations, args.ssim_decr_margin, args.dropout_ratio)
        
        if args.verbose:
            print(f"Train directory: {train_dir}")
            print(f"Test directory: {test_dir}")
            print(f"Dropout ratio: {dropout_ratio}")
            print(f"SSIM decrease margin: {args.ssim_decr_margin}")
            print(f"Parsing as GS model...")
            print("Loading GT and rendered (1 sample) train images...")

        # gt_train_images, _, _ = get_images_gs(train_dir, device, gt_only=True)
        # train_renders_dir = os.path.join(train_dir, 'renders')
        # train_images_one_sample = get_images_from_dir_of_dirs(train_renders_dir, device, max_samples=1)[0].squeeze(1)
        # if args.ssim_decr_margin is None:
        #     train_images_one_sample_no_dropout = get_images_from_dir_of_dirs(train_renders_dir.replace(f"dropout_{dropout_ratio}", "dropout_0.0"), device, max_samples=1)[0].squeeze(1)
        # else:
        #     train_images_one_sample_no_dropout = get_images_from_dir_of_dirs(train_renders_dir.replace(f"ssim_decr_{args.ssim_decr_margin}", "dropout_0.0"), device, max_samples=1)[0].squeeze(1)
        
        if args.verbose:
            print("Loading GT and rendered (all samples) test images...")
        gt_test_images, mean_test_images, std_test_images = get_images_gs(test_dir, device)

        # Compute SSIM w/ and w/o dropout (similar to NeRF)
        avg_train_ssim_no_dropout = 0 # torch.mean(compute_ssims(gt_train_images, train_images_one_sample_no_dropout)).item()
        avg_train_ssim = 0 # torch.mean(compute_ssims(gt_train_images, train_images_one_sample)).item()
    elif os.path.exists(args.model_dir + "/testset") and os.path.exists(args.model_dir + "/args.txt"):
        if args.num_iterations == -1:
            args.num_iterations = 50000 # default for NeRF
        test_dir, dropout_ratio, avg_train_ssim_no_dropout, avg_train_ssim, post_fix = get_dir_paths_nerf(args.model_dir, args.num_iterations, args.ssim_decr_margin, args.dropout_ratio)
        
        if args.verbose:
            # print(f"Train directory: {train_dir}")
            print(f"Test directory: {test_dir}")
            print(f"Dropout ratio: {dropout_ratio}")
            print(f"SSIM decrease margin: {args.ssim_decr_margin}")
            print(f"Parsing as NeRF model...")
            print("Loading GT and rendered (all samples) test images...")
       
        gt_test_images, mean_test_images, std_test_images = get_images_nerf(test_dir, device)
    else:
        raise ValueError(f"Model directory does not correspond to GS or NeRF ({args.model_dir})")
    
    if args.verbose:
        print(f"Test images:   GT:  {gt_test_images.shape},   Mean:  {mean_test_images.shape},   STD:  {std_test_images.shape}")

    expr_name = args.model_dir.split('/')[-1]
    output_dir = os.path.join(args.output_dir, expr_name, post_fix) # e.g. phd_results/chair_8v/ssim_decr_margin_0.01
    os.makedirs(output_dir, exist_ok=True)
    
    gt_test_images = gt_test_images.to(device)
    _results(gt_test_images, mean_test_images, std_test_images, output_dir, args)

    with open(os.path.join(output_dir, "metrics.txt"), "a") as f:
        f.write(f"dropout_ratio = {dropout_ratio}\n")
        f.write(f"avg_train_ssim_no_dropout = {avg_train_ssim_no_dropout}\n")
        f.write(f"avg_train_ssim = {avg_train_ssim}\n")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='phd_results')
    parser.add_argument('--num_iterations', type=int, default=-1, help="-1 for default (30000 for GS, 50000 for NeRF)")
    parser.add_argument('--ssim_decr_margin', type=float, default=None)
    parser.add_argument('--dropout_ratio', type=float, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_images', action='store_true', help="Save images (gt, mean pred., pred. error, uncertainty) as .png.")
    args = parser.parse_args()

    # Input validation
    if args.ssim_decr_margin is not None and args.dropout_ratio is not None:
        raise ValueError("Both dropout_ratio and ssim_decr_margin cannot be set at the same time.")
    if args.ssim_decr_margin is None and args.dropout_ratio is None:
        raise ValueError("Either dropout_ratio or ssim_decr_margin must be set.")
    
    with torch.no_grad():
        results(args)