from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from detection_model.util import ensemble_predict_single_image, get_model, draw_boxes, get_ground_truth_boxes, calculate_iou, get_prediction_boxes


def evaluate_and_print_individual_metrics(pred_boxes, gt_boxes, image_size):
    iou_threshold = 0.5
    all_points = image_size[0] * image_size[1]  # Total number of pixels in the image
    overall_iou_list = []

    # Initialize counters for overall metrics
    overall_TP = overall_FP = overall_FN = 0

    # Calculate and print individual metrics for each predicted box
    for i, pred_box in enumerate(pred_boxes):
        ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_boxes])
        max_iou = np.max(ious) if ious.size > 0 else 0
        overall_iou_list.append(max_iou)  # Add the IoU to the overall list
        max_iou_index = np.argmax(ious) if ious.size > 0 else -1

        if max_iou > iou_threshold:
            TP = 1
            overall_TP += 1
        else:
            TP = 0
            overall_FP += 1

        FP = 1 - TP
        FN = 1 - TP  # FN is 1 if there was no match (since it's a binary situation for each box)
        TN = all_points - (TP + FP + FN)  # Simplified assumption, not typically used

        # Calculate Precision, Recall, and Accuracy for the individual prediction
        Precision = TP / (TP + FP) if TP + FP > 0 else 0
        Recall = TP / (TP + FN) if TP + FN > 0 else 0
        Accuracy = (TP + TN) / (TP + FP + FN + TN) if all_points > 0 else 0

        print(f"Box {i} (Prediction): IoU: {max_iou}, Matched with Ground Truth Box: {max_iou_index}")
        print(f"Box {i} Metrics: Precision: {Precision}, Recall: {Recall}, Accuracy: {Accuracy}\n")

    # Calculate overall IoU as the average of individual IoUs
    overall_IoU = np.mean(overall_iou_list) if overall_iou_list else 0

    # Calculate FN for overall metrics by checking ground truth boxes that weren't matched
    matched_gt_boxes = set()
    for iou, gt_box in zip(ious, gt_boxes):
        if iou > iou_threshold:
            matched_gt_boxes.add(tuple(gt_box))  # Add matched ground truth box as a tuple to the set

    overall_FN = len(gt_boxes) - len(matched_gt_boxes)

    # Calculate overall Precision, Recall, and Accuracy
    overall_Precision = overall_TP / (overall_TP + overall_FP) if overall_TP + overall_FP > 0 else 0
    overall_Recall = overall_TP / (overall_TP + overall_FN) if overall_TP + overall_FN > 0 else 0
    overall_Accuracy = (overall_TP + (all_points - (overall_TP + overall_FP + overall_FN))) / all_points if all_points > 0 else 0

    return overall_Precision, overall_Recall, overall_Accuracy, overall_IoU


def detection(image_path):

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_folds = 4
    models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold in range(num_folds):
        model = get_model(num_classes=3).to(device)
        model_state_dict = torch.load(f'./detection_model/Result/best_model_fold_{fold}.pth', map_location='cpu')
        model.load_state_dict(model_state_dict)
        model.eval()
        models.append(model)
    
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).to(device)  # The image is already transformed here
    iou_threshold = 0.5  # Set a suitable IoU threshold value
    ensemble_prediction = ensemble_predict_single_image(models, transformed_image, device, iou_threshold)
    draw_boxes(transformed_image, ensemble_prediction, title='Prediction')

    # Process ground truth data from JSON
    json_path = image_path.replace('.png', '.json')
    original_size = (512, 512)  
    transformed_size = (256, 256)  
    gt_boxes, gt_labels = get_ground_truth_boxes(json_path, original_size, transformed_size)
    pred_boxes, pred_labels = get_prediction_boxes(transformed_image, ensemble_prediction)

    # Convert labels to numeric values
    label_mapping = {'right normal': 1, 'left normal': 2}
    numeric_labels = [label_mapping.get(label, 0) for label in gt_labels]

    # Convert boxes and labels to the format expected by draw_boxes
    ground_truth = {'boxes': torch.tensor(gt_boxes), 'labels': torch.tensor(numeric_labels)}

    # Draw prediction and ground truth boxes on the image
    draw_boxes(transformed_image.clone(), ground_truth, title='Ground Truth')

    image_size = (256, 256)  # Replace with your actual image size
    overall_Precision, overall_Recall, overall_Accuracy, overall_IoU = evaluate_and_print_individual_metrics(pred_boxes, gt_boxes, image_size)
    return overall_Precision, overall_Recall, overall_Accuracy, overall_IoU
