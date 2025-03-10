import tensorflow as tf
import numpy as np
import json
import os
import cv2

green = ['Food waste']
blue = ['Aluminium foil', 'Bottle', 'Bottle cap', 'Can', 'Carton', 'Cup', 'Glass jar', 'Lid', 'Paper', 'Paper bag', 'Plastic container', 'Plastic utensils', 'Pop tab', 'Scrap metal', 'Squeezable tube', 'Straw', 'Plastic gloves', 'Plastic bag & wrapper', 'Other plastic', 'Unlabeled litter','Rope & strings', 'Cigarette']
gray = ['Battery', 'Blister pack', 'Shoe', 'Styrofoam piece']

def calculate_iou(box1, box2):
    """Calculates IoU between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def load_annotations(json_path, img_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    categories = data['categories']
    
    category_to_supercategory = {cat['id']: cat['supercategory'] for cat in categories}
    supercategory_to_index = {supercat: idx for idx, supercat in enumerate(set(cat['supercategory'] for cat in categories))}
    
    index_to_supercategory = {idx: supercat for idx, supercat in enumerate(set(cat['supercategory'] for cat in categories))}

    image_ids = list(images.keys())

    valid_image_ids = []
    for img_id in image_ids:
        img_path = os.path.join(img_dir, images[img_id]['file_name'])
        if os.path.exists(img_path):
            valid_image_ids.append(img_id)

    return images, annotations, category_to_supercategory, supercategory_to_index, index_to_supercategory, valid_image_ids

def preprocess_image_and_boxes(img_path, image_info, annotations, category_to_supercategory, supercategory_to_index):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  

    img_anns = [ann for ann in annotations if ann['image_id'] == image_info['id']]
    boxes = []
    labels = []

    for ann in img_anns:
        x, y, w, h = ann['bbox']
        x = x / image_info['width']
        y = y / image_info['height']
        w = w / image_info['width']
        h = h / image_info['height']

        boxes.append([x, y, w, h])
        supercategory = category_to_supercategory[ann['category_id']]
        
        supercategory_idx = supercategory_to_index[supercategory]
        labels.append(supercategory_idx)

    boxes = np.array(boxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return img, boxes, labels

def get_supercategory(label):
    """Return the supercategory group (green, blue, or gray) for a given label."""
    if label in green:
        return 'green'
    elif label in blue:
        return 'blue'
    elif label in gray:
        return 'gray'
    else:
        return None

def calculate_accuracy(ground_truth_labels, predicted_labels):
    """Calculate the accuracy based on supercategories (green, blue, or gray)."""
    total_accuracy = 0
    total_samples = 0

    for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
        gt_supercategory = get_supercategory(gt_label)
        pred_supercategory = get_supercategory(pred_label)

        if gt_supercategory and pred_supercategory and gt_supercategory == pred_supercategory:
            total_accuracy += 1

        total_samples += 1

    return total_accuracy / total_samples if total_samples > 0 else 0

def evaluate_model(tflite_interpreter, test_dir, annotations_file, batch_size=8):
    images, annotations, category_to_supercategory, supercategory_to_index, index_to_supercategory, valid_image_ids = load_annotations(annotations_file, test_dir)

    total_iou = 0
    total_class_accuracy = 0
    total_samples = 0  # Initialize the counter for samples

    for img_id in valid_image_ids:
        image_info = images[img_id]
        img_path = os.path.join(test_dir, image_info['file_name'])

        img, ground_truth_boxes, ground_truth_labels = preprocess_image_and_boxes(
            img_path, image_info, annotations, category_to_supercategory, supercategory_to_index
        )

        if img is None or len(ground_truth_boxes) == 0:
            continue

        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0  

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        tflite_interpreter.set_tensor(input_details[0]['index'], img)
        tflite_interpreter.invoke()

        class_preds = tflite_interpreter.get_tensor(output_details[0]['index'])
        box_preds = tflite_interpreter.get_tensor(output_details[1]['index'])

        for gt_box, gt_label, pred_box, pred_class in zip(ground_truth_boxes, ground_truth_labels, box_preds[0], class_preds[0]):
            iou = calculate_iou(gt_box, pred_box)
            total_iou += iou

            pred_label = np.argmax(pred_class)

            gt_supercategory = get_supercategory(index_to_supercategory[gt_label])
            pred_supercategory = get_supercategory(index_to_supercategory[pred_label])

            if gt_supercategory == pred_supercategory:
                total_class_accuracy += 1

            total_samples += 1 

    avg_iou = total_iou / total_samples if total_samples > 0 else 0
    avg_class_accuracy = total_class_accuracy / total_samples if total_samples > 0 else 0

    return avg_iou, avg_class_accuracy

tflite_model_path = '../models/trash_detection_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Directories and annotation file paths
test_dir = '../data'
annotations_file = '../data/split/test_annotations.json'

# Evaluate model performance
avg_iou, avg_class_accuracy = evaluate_model(interpreter, test_dir, annotations_file)

# Output results
print(f'Average IoU: {avg_iou:.4f}')
print(f'Average Class Accuracy: {avg_class_accuracy:.4f}')