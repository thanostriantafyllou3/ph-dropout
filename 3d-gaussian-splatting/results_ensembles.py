import torch 
import numpy as np
import os
from PIL import Image
from argparse import ArgumentParser, Namespace
from torchmetrics.functional import structural_similarity_index_measure as ssim
from scipy.stats import pearsonr, spearmanr, norm
import matplotlib.pyplot as plt
from tqdm import tqdm

device = None

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
    return torch.Tensor([torch.max(std_image) for std_image in std_images]) # from [num_images, height, width, channels] to [num_images]


####################################################################################################
#####                                           MAIN                                           #####
####################################################################################################

def _results_ensembles(gt_images1, mean_images1, std_images1, gt_images2, mean_images2, std_images2, output_dir, args):
    global device
    # Compute SSIM and PSNR for both models
    psnr1 = compute_psnrs(gt_images1, mean_images1).to(device) # [num_images]
    ssim1 = compute_ssims(gt_images1, mean_images1).to(device) # [num_images]

    psnr2 = compute_psnrs(gt_images2, mean_images2).to(device) # [num_images]
    ssim2 = compute_ssims(gt_images2, mean_images2).to(device) # [num_images]

    # Get the "selected" and best images
    avg_std1 = torch.mean(std_images1, dim=(1, 2, 3)) # [num_images]
    avg_std2 = torch.mean(std_images2, dim=(1, 2, 3)) # [num_images]

    min_std_mask = avg_std1 < avg_std2
    ensemble_mean_images = []
    for i in range(len(min_std_mask)):
        ensemble_mean_images.append(mean_images1[i] if min_std_mask[i] else mean_images2[i])
    ensemble_mean_images = torch.stack(ensemble_mean_images).to(device)

    max_ssim_mask = ssim1 > ssim2
    best_ssim_mean_images = []
    for i in range(len(max_ssim_mask)):
        best_ssim_mean_images.append(mean_images1[i] if max_ssim_mask[i] else mean_images2[i])
    best_ssim_mean_images = torch.stack(best_ssim_mean_images).to(device)

    # Compute metrics for the ensemble
    psnr_selected = compute_psnrs(gt_images1, ensemble_mean_images)
    ssim_selected = compute_ssims(gt_images1, ensemble_mean_images) # [num_images]
    ssim_best = compute_ssims(gt_images1, best_ssim_mean_images) # [num_images]

    e_em = torch.mean(ssim_selected / ssim_best) # mean[ssim_selected_i / ssim_best_i for all i in [num_images]]

    # Store average metrics in a dictionary
    metrics = {
        "psnr1": torch.mean(psnr1).item(),
        "ssim1": torch.mean(ssim1).item(),
        "psnr2": torch.mean(psnr2).item(),
        "ssim2": torch.mean(ssim2).item(),
        "psnr_selected": torch.mean(psnr_selected).item(),
        "ssim_selected": torch.mean(ssim_selected).item(),
        "ssim_best": torch.mean(ssim_best).item(),
        "e_em": e_em.item()
    }

    if args.verbose:
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value}")

    # Save results to 'metrics'.txt
    results_file = os.path.join(output_dir, "metrics.txt")
    with open(results_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()} = {value}\n")

    print(f"Results saved to {results_file}")


def results_ensembles(args):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.verbose:
        print(f"Model 1 directory: {args.model1_dir}")
        print(f"Model 2 directory: {args.model2_dir}")
        print(f"Number of iterations: {args.num_iterations}")
        print(f"Using device: {device}")

    # Check if model directory corresponds to GS or NeRF
    if os.path.exists(args.model1_dir + "/point_cloud") and os.path.exists(args.model2_dir + "/point_cloud"):
        train1_dir, test1_dir, dropout_ratio1, post_fix1 = get_dir_paths_gs(args.model1_dir, args.num_iterations, args.ssim_decr_margin, args.dropout_ratio)
        train2_dir, test2_dir, dropout_ratio2, post_fix2 = get_dir_paths_gs(args.model2_dir, args.num_iterations, args.ssim_decr_margin, args.dropout_ratio)

        assert post_fix1 == post_fix2, f"Post-fixes do not match: {post_fix1} != {post_fix2}"

        if args.verbose:
            print("Loading GT and rendered test images for model 1...")
        gt_test1_images, mean_test1_images, std_test1_images = get_images_gs(test1_dir, device)
        if args.verbose:
            print("Loading GT and rendered test images for model 2...")
        gt_test2_images, mean_test2_images, std_test2_images = get_images_gs(test2_dir, device)

        assert gt_test1_images.shape == gt_test2_images.shape, f"GT images do not match: {gt_test1_images.shape} != {gt_test2_images.shape}"
    else:
        raise ValueError(f"Models' directories do not correspond to GS ({args.model1_dir}, {args.model2_dir})")
    
    if args.verbose:
        print(f"Test images for model 1:   GT:  {gt_test1_images.shape},   Mean:  {mean_test1_images.shape},   STD:  {std_test1_images.shape}")
        print(f"Test images for model 2:   GT:  {gt_test2_images.shape},   Mean:  {mean_test2_images.shape},   STD:  {std_test2_images.shape}")

    expr_name = args.model1_dir.split('/')[-1].split('_')[0] # e.g. chair_8v -> chair

    output_dir = os.path.join(args.output_dir, expr_name, post_fix1) # e.g. phd_results/chair_8v/ssim_decr_margin_0.01
    os.makedirs(output_dir, exist_ok=True)
    
    gt_test1_images, gt_test2_images = gt_test1_images.to(device), gt_test2_images.to(device)
    mean_test1_images, mean_test2_images = mean_test1_images.to(device), mean_test2_images.to(device)
    std_test1_images, std_test2_images = std_test1_images.to(device), std_test2_images.to(device)
    _results_ensembles(gt_test1_images, mean_test1_images, std_test1_images, gt_test2_images, mean_test2_images, std_test2_images, output_dir, args)
    
    with open(os.path.join(output_dir, "metrics.txt"), "a") as f:
        f.write(f"dropout_ratio1 = {dropout_ratio1}\n")
        f.write(f"dropout_ratio2 = {dropout_ratio2}\n")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model1_dir', type=str, required=True)
    parser.add_argument('--model2_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='ensembles_results')
    parser.add_argument('--num_iterations', type=int, default=30000)
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
        results_ensembles(args)