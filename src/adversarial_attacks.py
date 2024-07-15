import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.path_utils import (load_hyperparameters_cub, load_hyperparameters_shapes, load_model_cub,
                                   load_model_shapes, save_adversarial_hyperparameters, save_model_cub,
                                   save_model_shapes)
from src.common.utils import get_logger, load_single_model, load_single_model_cub, seed_everything
from src.datasets.datasets_cub import denormalize_cub, load_data_cub
from src.datasets.datasets_shapes import denormalize_shapes, load_data_shapes
from src.plotting import plot_perturbed, plot_perturbed_images
from src.train import train_cbm

logger = get_logger(__name__)


def run_iterative_class_attack(model, input_image, label, target=None, logits=False, least_likely=False,
                               epsilon=0.5, alpha=0.005, max_steps=100, extra_steps=1, random_start=None,
                               mean=0.5, std=0.5, device="cpu", non_blocking=False):
    """
    Run a iterative adversarial attack. Many possible methods.
    The attack is on the class, with no evaluation of what is going on with the concepts.
    In other words, "normal" adversary attack.

    Setting epsilon < 1 means projected gradient method (PGM), which projects the perturbed images back onto a ball
    of radius `epsilon` around the original input image.
    Setting `target` will make the attack targeted to class number `target`, and setting `least_likely` to `True` will
    make the target the least likely class.
    Setting `logits` to `True` will make the attack change the gradient to the logits of the class (up or down if
    targeted or not), and setting it to `False` will use the signed cross entropy loss.


    Args:
        model (pytorch model): The model to do attacks based on. Must be a concept model.
        input_image (Tensor): Original input image.
        label (Tensor): The label of the image.
        target (int): If not `None`, will do target attack against this class.
        logits (bool): If `True`, will do attacks based on the gradients of the logits.
            If `False`, will do based on the signed cross entropy loss.
        least_likely (bool): If `True`, will do targeted attack towards the least likely predicted class of
            the original image.
        epsilon (float): The maximum perturbation allowed. Uses L2 projection down on it.
        alpha (float): Step size per iteration.
        max_steps (int): Maximum number of iterations.
        extra_steps (int): How many steps are run after the perturbed image turnes adversarial.
        random_start (float): If not None, will make starting image in a random start within the original input image.
            Will be the uniform noise added to each pixel in the random start.
        mean (float or Tensor): Mean of the normalization done in the dataloader.
        std (float or Tensor): Standard deviation of the normalization done in the dataloader.
        device (str): Use "cpu" for cpu training and "cuda:0" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.

    Returns:
        Tensor: Perturbed image.
        int: 1 or 0 depending on if the image succesfully turned adversarial.
        int: The number of iterations ran.
    """
    if isinstance(target, int):
        target = torch.tensor([target]).to(device, non_blocking=non_blocking)

    perturbed_image = input_image
    if random_start is not None:  # Make image slightly away from the input image
        noise = torch.empty_like(input_image).uniform_(-random_start, random_start)
        perturbed_image = perturbed_image + noise
        perturbed_image = torch.clamp(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, -mean / std, (1 - mean) / std)

    success = 0
    for i in range(max_steps):
        perturbed_image = perturbed_image.detach().clone()
        perturbed_image.requires_grad = True

        class_outputs, attr_outputs = model(perturbed_image)
        _, prediction = torch.max(class_outputs, 1)

        if least_likely and target is None:  # Set least-likely class in first iteration
            least_likely_class = class_outputs.argmin(dim=1)
            target = least_likely_class

        # Check if we have reached our goal
        if target is not None:
            if prediction == target:
                extra_steps -= 1
                success = 1
        else:
            if prediction != label:
                extra_steps -= 1
                success = 1
        if extra_steps <= 0:  # Sufficient steps taken after changing class prediction
            success = 1
            break

        model.zero_grad()
        if logits:  # Alter logit values
            if target is not None:
                logit = class_outputs[0][target][0]  # Batch size is 1.
                sign = 1
            else:
                logit = class_outputs[0][label][0]
                sign = -1
            logit.backward(retain_graph=True)
        else:  # Alter Signed loss
            if target is not None:
                loss = F.cross_entropy(class_outputs, target)
                sign = -1
            else:
                loss = F.cross_entropy(class_outputs, label)
                sign = 1
            loss.backward(retain_graph=True)

        # Apply perturbation
        perturbed_image = perturbed_image + sign * alpha * perturbed_image.grad.data.sign()
        if epsilon is not None:
            perturbed_image = torch.clamp(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, -mean / std, (1 - mean) / std)

    return perturbed_image, success, i


def run_iterative_concept_attack(
        model, input_image, class_label, concept_labels, target=None, logits=True, least_likely=False, epsilon=0.5,
        alpha=0.005, concept_threshold=0.1, grad_weight=-0.3, max_steps=100, extra_steps=1, random_start=None, mean=0.5,
        std=0.5, device=None, non_blocking=False):
    """
    Run a iterative adversarial attack. Many possible methods.
    Tries to change the class without changing the concept-predictions.
    This is done by comparing the concept predictions to the concept predictions from the first iteration, and breaking
    if they are different. Additionally, this scenario is attempted to be avoided. If one or more concept prediction
    logit is close to 0 (by `concept_threshold`), we calculate the gradients to this concepts with respect to the
    pixels in the input image. Then, the pixels which are supposed to move in a direction that makes the concept
    logits close to 0 are zeroed out.
    This needs some tuning, since a high `alpha` will make the concept-prediction change over 0 to fast, and
    a high `concept_threshold` will make the gradients zero out too much and therefore make no progress.

    Setting epsilon < 1 means projected gradient method (PGM), which projects the perturbed images back onto a ball
    of radius `epsilon` around the original input image.
    Setting `target` will make the attack targeted to class number `target`, and setting `least_likely` to `True` will
    make the target the least likely class.
    Setting `logits` to `True` will make the attack change the gradient to the logits of the class (up or down if
    targeted or not), and setting it to `False` will use the signed cross entropy loss.

    Args:
        model (pytorch model): The model to do attacks based on. Must be a concept model.
        input_image (Tensor): Original input image.
        class_label (Tensor): The class label of the image.
        concept_labels (Tensor): The concept label vector for the image.
        target (int): If not `None`, will do target attack against this class.
        logits (bool): If `True`, will do attacks based on the gradients of the logits.
            If `False`, will do based on the signed cross entropy loss.
        least_likely (bool): If `True`, will do targeted attack towards the least likely predicted class of
            the original image.
        epsilon (float): The maximum perturbation allowed. Uses L2 projection down on it.
        alpha (float): Step size per iteration.
        concept_threshold (float): The threshold of how close the concept logits need to be to zero before zeroing
            out the gradients.
        grad_weight (float): The weight to zero out or cancel the gradient to the pixels that will change sensitive
            concepts. Should probably be in [-1, 0].
        max_steps (int): Maximum number of iterations.
        extra_steps (int): How many steps are run after the perturbed image turnes adversarial.
        random_start (float): If not None, will make starting image in a random start within the original input image.
            Will be the uniform noise added to each pixel in the random start.
        mean (float or Tensor): Mean of the normalization done in the dataloader.
        std (float or Tensor): Standard deviation of the normalization done in the dataloader.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.

    Returns:
        Tensor: Perturbed image.
        int: 1 or 0 depending on if the image succesfully turned adversarial.
        int: The number of iterations ran.
        int: Marks if the mask got turned into zeros.
        int: Marks if the concepts changed.
    """
    if isinstance(target, int):
        target = torch.tensor([target]).to(device, non_blocking=non_blocking)

    perturbed_image = input_image
    if random_start is not None:  # Make image slightly away from the input image
        noise = torch.empty_like(input_image).uniform_(-random_start, random_start)
        perturbed_image = perturbed_image + noise
        perturbed_image = torch.clamp(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, -mean / std, (1 - mean) / std)

    success = 0
    overall_accuracy = 0
    zero_mask = 0
    changed_concepts = 0
    previous_image = perturbed_image.clone()  # Save last iterations image
    for i in range(max_steps + extra_steps):
        if success == 0 and i >= max_steps:
            break

        perturbed_image = perturbed_image.detach().clone()
        perturbed_image.requires_grad = True

        class_outputs, attr_outputs = model(perturbed_image)
        _, prediction = torch.max(class_outputs, 1)

        if i == 0:  # Save original attribute predictions
            original_attr_probabilities = torch.sigmoid(attr_outputs)
            original_predictions = (original_attr_probabilities > 0.5).float()
            num_concepts = attr_outputs.shape[1]

        if least_likely and target is None:  # Set least-likely class in first iteration
            least_likely_class = class_outputs.argmin(dim=1)
            target = least_likely_class

        attr_probabilities = torch.sigmoid(attr_outputs)
        predictions = (attr_probabilities > 0.5).float()
        correct_predictions = (predictions == original_predictions).float()
        overall_accuracy = correct_predictions.mean()

        if overall_accuracy < 1:  # Concept predictions have changed
            message = f"Aborting attack due to changed concepts. \n     Iteration {i}, Success {success}. "
            message += f"Concept-Accuracy {overall_accuracy:.4f}. "
            logger.debug(message)
            if success == 1:  # Reset last image with correct predictions
                perturbed_image = previous_image
            changed_concepts = 1
            break

        # Check if we have reached our goal
        if target is not None:
            if prediction == target:
                extra_steps -= 1
                success = 1
            else:
                success = 0
        else:
            if prediction != class_label:
                extra_steps -= 1
                success = 1
            else:
                success = 0
        if extra_steps <= 0:  # Sufficient steps taken after changing class prediction
            success = 1
            attr_probabilities = torch.sigmoid(attr_outputs)
            predictions = (attr_probabilities > 0).float()
            correct_predictions = (predictions == original_predictions).float()
            overall_accuracy = correct_predictions.mean()
            break

        model.zero_grad()
        # Calculate gradient based on class-target:
        if logits:  # Alter logit values
            if target is not None:
                logit = class_outputs[0][target][0]  # Batch size is 1.
                sign = 1
            else:
                logit = class_outputs[0][class_label][0]
                sign = -1
            logit.backward(retain_graph=True)
        else:  # Alter Signed loss
            if target is not None:
                loss = F.cross_entropy(class_outputs, target)
                sign = -1
            else:
                loss = F.cross_entropy(class_outputs, class_label)
                sign = 1
            loss.backward(retain_graph=True)
        # Save the gradient from the class and zero the gradient
        class_adversarial_gradient = perturbed_image.grad.data.sign().clone()
        perturbed_image.grad.data.zero_()

        # Examine gradients for concepts that are close to 0 (and therefore close to changing):
        concept_distances = attr_outputs.abs()
        sensitive_concepts = concept_distances < concept_threshold

        concept_gradients = []
        for j in range(num_concepts):  # Calculate the gradients to the sensitive concepts
            if not sensitive_concepts[0][j]:
                concept_gradients.append(None)
                continue
            model.zero_grad()
            attr_outputs[:, j].backward(retain_graph=True)
            concept_gradients.append(perturbed_image.grad.data.sign().clone())
            perturbed_image.grad.data.zero_()

        final_mask = torch.ones_like(perturbed_image.grad.data)
        for j in range(num_concepts):  # Make a zero-mask for sensitive concepts that move close towards 0
            if not sensitive_concepts[0][j]:
                continue
            concept_grad_sign = concept_gradients[j]
            concept_sign = attr_outputs[:, j].sign()
            # Zero out gradients that would increase positive concepts or decrease negative concepts
            total_grad_sign = concept_grad_sign * class_adversarial_gradient  # Direction of change
            mask = torch.where((total_grad_sign != concept_sign), grad_weight * torch.ones_like(final_mask), final_mask)
            final_mask = torch.min(final_mask, mask)

        final_mask_float = final_mask.to(torch.float32)
        grad_weight_tensor = torch.tensor(grad_weight, dtype=torch.float32)
        if torch.all(torch.isclose(final_mask_float, grad_weight_tensor, atol=0.000001)):
            message = f"Aborting attack due all-beta mask \n     Iteration {i}, Success {success}. "
            message += f"Concept-Accuracy {overall_accuracy:.4f}. "
            logger.debug(message)
            zero_mask = 1
            break
        # Apply perturbation
        modified_class_gradient = class_adversarial_gradient * final_mask
        previous_image = perturbed_image.clone()  # Copy last image
        perturbed_image = perturbed_image + sign * alpha * modified_class_gradient
        if epsilon is not None:  # Projected gradient in L2
            perturbed_image = torch.clamp(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, -mean / std, (1 - mean) / std)

    return perturbed_image, success, i, zero_mask, changed_concepts


def run_adversarial_attacks(
        model, test_loader, target=None, logits=True, least_likely=False, epsilon=1, alpha=0.001, concept_threshold=0.1,
        grad_weight=-0.3, max_steps=100, extra_steps=3, max_images=100, random_start=None, denorm_func=None, mean=0.5,
        std=0.5, device="cpu", non_blocking=False):
    """
    Runs adversarial attacks on images from the `test_loader`.

    Args:
        model (pytorch model): The model to do attacks based on
        test_loader (dataloader): Dataloader to run attacks on. Should have batch-size 1.
        target (int): If not `None`, will do target attack against this class.
        logits (bool): If `True`, will do attacks based on the gradients of the logits.
            If `False`, will do based on the signed cross entropy loss.
        least_likely (bool): If `True`, will do targeted attack towards the least likely predicted class of
            the original image.
        epsilon (float): The maximum perturbation allowed. Uses L2 projection down on it.
        alpha (float): Step size per iteration.
        concept_threshold (float): The threshold of how close the concept logits need to be to zero before zeroing
            out the gradients.
        grad_weight (float): The weight to zero out or cancel the gradient to the pixels that will change sensitive
            concepts. Should probably be in [-1, 0].
        max_steps (int): Maximum number of iterations.
        extra_steps (int): How many steps are run after the perturbed image turnes adversarial.
        max_images (int): Maximum number of images from the test-loader to run.
        random_start (float): If not None, will make starting image in a random start within the original input image.
            Will be the uniform noise added to each pixel in the random start.
        denorm_func (callable): Function for de-normalising images.
        mean (float or Tensor): Mean of the normalization done in the dataloader.
        std (float or Tensor): Standard deviation of the normalization done in the dataloader.
        device (str): Use "cpu" for cpu training and "cuda:0" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.

    Returns:
        dict: Dictionary of lists of the images that were successfully ran adversarial attacks on. The keys are:
            perturbed_images, original_images, original_predictions, new_predictions, iterations_list
            and success_rate, which is the ratio of images successfully ran on.
    """
    original_predictions = []
    new_predictions = []
    original_images = []
    perturbed_images = []
    iterations_list = []
    path_list = []
    attr_predictions_list = []
    attr_accuracy_list = []
    attr_precision_list = []
    attr_recall_list = []

    if denorm_func is None:
        denorm_func = denormalize_shapes  # Shapes denormalisation

    counter = 0
    correct_counter = 0
    zero_mask_counter = 0
    changed_concepts_counter = 0
    for data, label, attr_labels, path in test_loader:
        data = data.to(device, non_blocking=non_blocking)
        label = label.to(device, non_blocking=non_blocking)
        attr_labels = attr_labels.to(device, non_blocking=non_blocking)
        class_outputs, attr_outputs = model(data)
        _, prediction = torch.max(class_outputs, 1)

        if prediction.item() != label.item():  # Original prediction was wrong, iterate
            continue

        if target is not None and prediction.item() == target:  # Already predicting target class
            continue

        counter += 1
        if counter % 10 == 0:
            logger.debug(f"On iteration [{counter} / {max_images}] ")

        # Run the attack
        perturbed_image, success, iterations_ran, zero_mask, changed_concepts = run_iterative_concept_attack(
            model=model, input_image=data, class_label=label, concept_labels=attr_labels, target=target, logits=logits,
            least_likely=least_likely, epsilon=epsilon, alpha=alpha, concept_threshold=concept_threshold,
            grad_weight=grad_weight, max_steps=max_steps, extra_steps=extra_steps, random_start=random_start,
            mean=mean, std=std, device=device, non_blocking=non_blocking)

        logger.debug(f"Image number {counter}, Succes: {success}, iterations ran: {iterations_ran}")

        if success:  # Save stuff
            new_class_output, new_attr_outputs = model(perturbed_image)
            new_attr_predictions = (torch.sigmoid(new_attr_outputs) > 0.5).float()
            old_attr_predictions = (torch.sigmoid(attr_outputs) > 0.5).float()
            if not torch.all(torch.eq(old_attr_predictions, new_attr_predictions)):  # Double check right concepts
                logger.warn(f"Concept predictions changed. {old_attr_predictions=} {new_attr_predictions=}")
            else:  # Successful adversarial concept attack, add a bunch of stats
                correct_counter += 1
                attr_predictions_list.append(new_attr_predictions)
                attr_accuracy = (new_attr_predictions == attr_labels).sum().item() / len(attr_labels.squeeze()) * 100
                attr_accuracy_list.append(attr_accuracy)
                true_attr_positives = torch.sum((new_attr_predictions == 1) & (attr_labels == 1))
                false_attr_positives = torch.sum((new_attr_predictions == 1) & (attr_labels == 0))
                false_attr_negatives = torch.sum((new_attr_predictions == 0) & (attr_labels == 1))
                if true_attr_positives + false_attr_positives == 0:
                    attr_precision = 0
                else:
                    attr_precision = true_attr_positives.float() / (true_attr_positives + false_attr_positives) * 100
                if true_attr_positives + false_attr_negatives == 0:
                    attr_recall = 0
                else:
                    attr_recall = true_attr_positives.float() / (true_attr_positives + false_attr_negatives) * 100
                attr_precision_list.append(attr_precision)
                attr_recall_list.append(attr_recall)
                _, new_prediction = torch.max(new_class_output, 1)
                image = denorm_func(data)
                perturbed_image = denorm_func(perturbed_image)
                original_images.append(image)
                perturbed_images.append(perturbed_image)
                original_predictions.append(prediction.item())
                new_predictions.append(new_prediction.item())
                iterations_list.append(iterations_ran)
                path_list.append(path)

        zero_mask_counter += zero_mask
        changed_concepts_counter += changed_concepts

        if counter >= max_images:
            break

    success_rate = correct_counter / counter
    zero_mask_rate = zero_mask_counter / counter
    changed_concepts_rate = changed_concepts_counter / counter
    output = {
        "perturbed_images": perturbed_images, "original_images": original_images,
        "original_predictions": original_predictions, "new_predictions": new_predictions,
        "iterations_list": iterations_list, "paths": path_list, "attr_predictions": attr_predictions_list,
        "attr_accuracy": attr_accuracy_list, "attr_precision": attr_precision_list, "attr_recall": attr_recall_list,
        "success_rate": success_rate, "zero_mask_rate": zero_mask_rate, "changed_concepts_rate": changed_concepts_rate}
    logger.info(f"Adversarial attack results: {success_rate=}, {zero_mask_rate=}, {changed_concepts_rate=}")
    return output


def load_model_and_run_attacks_shapes(
        n_classes, n_attr, signal_strength, train_model=False, target=None, logits=False, least_likely=False,
        epsilon=1, alpha=0.001, concept_threshold=0.1, grad_weight=-0.3, max_steps=100, extra_steps=3, max_images=100,
        random_start=None, batch_size=16, plot=True, device=None, num_workers=0, pin_memory=False,
        persistent_workers=False, non_blocking=False, seed=57):
    """
    Loads model and tes-dataloader, and runs adversarial attacks.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        signal_strength (int): The signal-strength used for creating the dataset.
        train_model (bool): Wether to train the model or not.
        target (int): If not `None`, will do target attack against this class.
        logits (bool): If `True`, will do attacks based on the gradients of the logits.
            If `False`, will do based on the signed cross entropy loss.
        least_likely (bool): If `True`, will do targeted attack towards the least likely predicted class of
            the original image.
        epsilon (float): The maximum perturbation allowed. Uses L2 projection down on it.
        alpha (float): Step size per iteration.
        concept_threshold (float): The threshold of how close the concept logits need to be to zero before zeroing
            out the gradients.
        grad_weight (float): The weight to zero out or cancel the gradient to the pixels that will change sensitive
            concepts. Should probably be in [-1, 0].
        max_steps (int): Maximum number of iterations.
        extra_steps (int): How many steps are run after the perturbed image turnes adversarial.
        random_start (float): If not None, will make starting image in a random start within the original input image.
            Will be the uniform noise added to each pixel in the random start.
        max_images (int): Maximum number of images from the test-loader to run.
        batch_size (int): Batch-size for training the model.
        plot (bool): If `True`, will plot some of the adversarial images.
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.
        seed (int, optional): Seed, used in case subdataset needs to be created. Defaults to 57.
    """
    seed_everything(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_subset = 250
    test_loader = load_data_shapes(n_classes, n_attr, signal_strength,
                                   mode="test", shuffle=True, batch_size=1)
    hp = load_hyperparameters_shapes(n_classes, n_attr, signal_strength, n_subset, hard_bottleneck=False)["cbm"]
    model = load_single_model("cbm", n_classes, n_attr, hyperparameters=hp)
    model = model.to(device, non_blocking=non_blocking)
    if train_model:  # Train model
        train_loader, val_loader = load_data_shapes(
            n_classes=n_classes, n_attr=n_attr, signal_strength=signal_strength, n_subset=n_subset,
            mode="train-val", batch_size=batch_size, drop_last=True, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=persistent_workers)

        criterion = nn.CrossEntropyLoss()
        attr_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp["learning_rate"])
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=hp["gamma"])
        _, state_dicts = train_cbm(
            model, criterion, attr_criterion, optimizer, train_loader, val_loader, n_epochs=50,
            attr_weight=hp["attr_weight"], scheduler=exp_lr_scheduler, device=device,
            non_blocking=non_blocking, verbose=2)
        state_dict = state_dicts["best_model_loss_state_dict"]
        save_model_shapes(
            n_classes, n_attr, signal_strength, n_subset, state_dict, "cbm", metric="loss",
            hard_bottleneck=False, adversarial=True)

    state_dict = load_model_shapes(n_classes, n_attr, signal_strength, n_subset, "cbm", hard_bottleneck=False,
                                   device=device, adversarial=True)
    model.load_state_dict(state_dict)
    model.eval()

    output = run_adversarial_attacks(
        model, test_loader, target=target, logits=logits, least_likely=least_likely, epsilon=epsilon, alpha=alpha,
        concept_threshold=concept_threshold, grad_weight=grad_weight, max_steps=max_steps, extra_steps=extra_steps,
        max_images=max_images, random_start=random_start, denorm_func=denormalize_shapes, mean=0.5, std=0.5,
        device=device, non_blocking=non_blocking)
    if plot:
        plot_perturbed_images(output["perturbed_images"], output["original_images"], output["original_predictions"],
                              output["new_predictions"], output["iterations_list"],
                              adversarial_filename="adversarial_image_shapes.png", max_rows=7)
        plot_perturbed(
            output, alpha=alpha, concept_threshold=concept_threshold, dataset_name="shapes", n_classes=n_classes)
    return output


def load_model_and_run_attacks_cub(
        train_model=False, target=None, logits=False, least_likely=False, epsilon=1, alpha=0.001, concept_threshold=0.1,
        grad_weight=-0.3, max_steps=100, extra_steps=3, max_images=100, random_start=None, batch_size=16, plot=True,
        device=None, num_workers=0, pin_memory=False, persistent_workers=False, non_blocking=False, seed=57):
    """
    Loads model and tes-dataloader, and runs adversarial attacks.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        signal_strength (int): The signal-strength used for creating the dataset.
        train_model (bool): Wether to train the model or not.
        target (int): If not `None`, will do target attack against this class.
        logits (bool): If `True`, will do attacks based on the gradients of the logits.
            If `False`, will do based on the signed cross entropy loss.
        least_likely (bool): If `True`, will do targeted attack towards the least likely predicted class of
            the original image.
        epsilon (float): The maximum perturbation allowed. Uses L2 projection down on it.
        alpha (float): Step size per iteration.
        concept_threshold (float): The threshold of how close the concept logits need to be to zero before zeroing
            out the gradients.
        grad_weight (float): The weight to zero out or cancel the gradient to the pixels that will change sensitive
            concepts. Should probably be in [-1, 0].
        max_steps (int): Maximum number of iterations.
        extra_steps (int): How many steps are run after the perturbed image turnes adversarial.
        max_images (int): Maximum number of images from the test-loader to run.
        random_start (float): If not None, will make starting image in a random start within the original input image.
            Will be the uniform noise added to each pixel in the random start.
        batch_size (int): Batch-size for training the model.
        plot (bool): If `True`, will plot some of the adversarial images.
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.
        seed (int, optional): Seed, used in case subdataset needs to be created. Defaults to 57.
    """
    seed_everything(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = load_data_cub(mode="test", shuffle=True, batch_size=1)
    n_subset = None

    hp = load_hyperparameters_cub(n_subset, hard_bottleneck=False)["cbm"]
    model = load_single_model_cub("cbm", hyperparameters=hp)
    model = model.to(device, non_blocking=non_blocking)
    if train_model:  # Train model
        train_loader, val_loader = load_data_cub(
            mode="train-val", batch_size=batch_size, drop_last=True, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=persistent_workers)

        criterion = nn.CrossEntropyLoss()
        attr_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp["learning_rate"])
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=hp["gamma"])
        _, state_dicts = train_cbm(
            model, criterion, attr_criterion, optimizer, train_loader, val_loader, n_epochs=50,
            attr_weight=hp["attr_weight"], scheduler=exp_lr_scheduler, device=device,
            non_blocking=non_blocking, verbose=2)
        state_dict = state_dicts["best_model_loss_state_dict"]
        save_model_cub(n_subset, state_dict, "cbm", metric="loss", hard_bottleneck=False, adversarial=True)

    state_dict = load_model_cub(n_subset, "cbm", hard_bottleneck=False, device=device, adversarial=True)
    model.load_state_dict(state_dict)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Imagenet normalisation
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    mean = mean.to(device, non_blocking=non_blocking)
    std = std.to(device, non_blocking=non_blocking)
    output = run_adversarial_attacks(
        model, test_loader, target=target, logits=logits, least_likely=least_likely, epsilon=epsilon, alpha=alpha,
        concept_threshold=concept_threshold, grad_weight=grad_weight, max_steps=max_steps, extra_steps=extra_steps,
        max_images=max_images, random_start=random_start, denorm_func=denormalize_cub, mean=mean, std=std,
        device=device, non_blocking=non_blocking)
    if plot:
        plot_perturbed_images(output["perturbed_images"], output["original_images"], output["original_predictions"],
                              output["new_predictions"], output["iterations_list"],
                              adversarial_filename="adversarial_image_cub.png")
        plot_perturbed(output, alpha=alpha, concept_threshold=concept_threshold, dataset_name="cub")
    return output


def adversarial_grid_search(
        dataset, n_classes=None, n_attr=None, signal_strength=None, train_model=False, logits=False, epsilon=1,
        max_steps=100, extra_steps=3, max_images=100, random_start=None, end_name="", device=None, num_workers=0,
        pin_memory=False, persistent_workers=False, non_blocking=False, seed=57):
    """
    Runs a grid-search on the hyperparameters for the adversarial-concept-attack.

    Args:
        dataset (str): Which dataset to use. Either "Shapes" or "CUB".
        n_classes (int): If Shapes, Amount of classes.
        n_attr (int): If Shapes, Amount of attribues.
        signal_strength (int): If Shapes, The signal-strength used for creating the dataset.
        train_model (bool): Wether to train the model or not.
        logits (bool): If `True`, will do attacks based on the gradients of the logits.
            If `False`, will do based on the signed cross entropy loss.
        epsilon (float): The maximum perturbation allowed. Uses L2 projection down on it.
        max_steps (int): Maximum number of iterations.
        extra_steps (int): How many steps are run after the perturbed image turnes adversarial.
        max_images (int): Maximum number of images from the test-loader to run.
        random_start (float): If not None, will make starting image in a random start within the original input image.
            Will be the uniform noise added to each pixel in the random start.
        batch_size (int): Batch-size for training the model.
        end_name (str): Name suffix for the saved hyperparameters.
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.
        seed (int, optional): Seed, used in case subdataset needs to be created. Defaults to 57.

    Returns:
        dict: Best hyperparameters
    """
    dataset = dataset.strip().lower()
    if dataset not in ["shapes", "cub"]:
        raise ValueError(f"Argument `dataset` must be either \"shapes\" or \"cub\". Was {dataset}. ")
    if dataset == "shapes" and (n_classes is None or n_attr is None or signal_strength is None):
        message = "Arguments `n_classes`, `n_attr` and `signal_strength` must be provided when `dataset` == "
        message += f"\"shapes\". Was {n_classes=}, {n_attr=}, {signal_strength=}. "
        raise ValueError(message)

    if train_model:
        if dataset == "shapes":
            load_model_and_run_attacks_shapes(
                n_classes, n_attr, signal_strength, train_model=True, target=None, logits=logits, max_steps=1,
                extra_steps=1, max_images=1, batch_size=16, plot=False, device=device, num_workers=num_workers,
                pin_memory=pin_memory, persistent_workers=persistent_workers, non_blocking=non_blocking, seed=seed)
        elif dataset == "cub":
            load_model_and_run_attacks_cub(
                train_model=True, target=None, logits=False, max_steps=1, extra_steps=1, max_images=1, batch_size=16,
                plot=False, device=device, num_workers=num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, non_blocking=non_blocking, seed=seed)

    if dataset == "shapes":
        gradient_weights = [-0.1]
        alphas = [0.003, 0.001, 0.00075]
        sensitivity_thresholds = [0.1, 0.05, 0.01]
    elif dataset == "cub":
        gradient_weights = [-0.1]
        alphas = [0.0001, 0.000075, 0.00005]
        sensitivity_thresholds = [0.1, 0.075, 0.05, 0.02]

    best_success = 0
    best_hyperparameters = None
    results = []
    counter = 0
    total_runs = len(gradient_weights) * len(alphas) * len(sensitivity_thresholds)
    for gradient_weight in gradient_weights:
        for alpha in alphas:
            for sensitivity_threshold in sensitivity_thresholds:
                counter += 1
                logger.info(f"On search [{counter} / {total_runs}]")
                if dataset == "shapes":
                    output = load_model_and_run_attacks_shapes(
                        n_classes, n_attr, signal_strength, train_model=False, target=None, logits=logits,
                        least_likely=False, epsilon=epsilon, alpha=alpha, concept_threshold=sensitivity_threshold,
                        grad_weight=gradient_weight, max_steps=max_steps, extra_steps=extra_steps,
                        max_images=max_images, random_start=random_start, batch_size=16, plot=False, device=device,
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
                        non_blocking=non_blocking, seed=seed)
                elif dataset == "cub":
                    output = load_model_and_run_attacks_cub(
                        train_model=False, target=None, logits=logits,
                        least_likely=False, epsilon=epsilon, alpha=alpha, concept_threshold=sensitivity_threshold,
                        grad_weight=gradient_weight, max_steps=max_steps, extra_steps=extra_steps,
                        max_images=max_images, random_start=random_start, batch_size=16, plot=False, device=device,
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
                        non_blocking=non_blocking, seed=seed)
                success_rate = output["success_rate"]
                result_dict = {
                    "gradient_weight": gradient_weight, "alpha": alpha, "sensitivity_threshold": sensitivity_threshold,
                    "epsilon": epsilon, "max_steps": max_steps, "extra_steps": extra_steps, "max_images": max_images,
                    "logits": logits, "zero_mask_rate": output["zero_mask_rate"],
                    "changed_concept_rate": output["changed_concepts_rate"], "success_rate": success_rate, }
                results.append(result_dict)
                logger.info(f"Results: {result_dict} \n")
                if success_rate >= best_success:
                    best_success = success_rate
                    best_hyperparameters = result_dict

    save_adversarial_hyperparameters(dataset=dataset, hyperparameters=best_hyperparameters, end_name=end_name)
    logger.info(f"Beset hyperparameters found: {best_hyperparameters}")
    return best_hyperparameters
