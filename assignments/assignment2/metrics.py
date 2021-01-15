def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    true_values = (prediction == ground_truth).sum()

    # TODO: Implement computing accuracy
#     raise Exception("Not implemented!")

    return true_values/len(prediction)
