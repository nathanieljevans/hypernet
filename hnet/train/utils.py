import torch

def expected_calibration_error(y_pred, y_true, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE) of a multiclass classifier.

    Args:
        y_pred (torch.Tensor): Predicted logits of shape (n_samples, n_classes)
        y_true (torch.Tensor): True labels of shape (n_samples, 1) or (n_samples,)
        n_bins (int): Number of bins to use for ECE calculation

    Returns:
        ece (float): Expected Calibration Error
    """

    # Ensure y_pred and y_true are on the same device
    device = y_pred.device
    y_true = y_true.to(device)

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(y_pred, dim=1)  # Shape: (n_samples, n_classes)

    # Get predicted confidences and predicted classes
    confidences, predictions = torch.max(probabilities, dim=1)  # Shapes: (n_samples,), (n_samples,)

    # Flatten y_true if necessary
    if y_true.dim() > 1:
        y_true = y_true.squeeze()

    # Ensure y_true is of shape (n_samples,)
    accuracies = predictions.eq(y_true)  # Shape: (n_samples,)

    # Initialize bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)  # Bin edges: [0.0, 0.1, ..., 1.0]
    bin_lowers = bin_boundaries[:-1]  # Lower edges
    bin_uppers = bin_boundaries[1:]   # Upper edges

    ece = torch.zeros(1, device=device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples that fall into the bin
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())  # Boolean mask
        prop_in_bin = in_bin.float().mean()  # Proportion of samples in bin

        if prop_in_bin.item() > 0:
            # Compute average confidence and average accuracy in the bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].float().mean()

            # Compute absolute difference and add to ECE
            ece += torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin

    return ece.item()