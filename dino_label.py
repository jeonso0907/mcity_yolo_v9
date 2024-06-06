import os
import cv2
import supervision as sv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import time
import shutil
import random

# Define your mappings and model
definitions = {
    "pedestrian with blue shirt": "pedestrian",
    "grey car": "car",
    "brown deer which looks like a fake deer": "deer",
    "round orange and white striped barrel. it has 3 orange stripes and 2 white stripes": "barrel",
    "non static color striped type III barricade": "barricade",
    "traffic light": "traffic light"
}
label_map = {
    0: "pedestrian",
    1: "car",
    2: "deer",
    3: "barrel",
    4: "barricade",
    5: "traffic light"
}
base_model = GroundingDINO(ontology=CaptionOntology(definitions))

# Define directories
base_dir = "mcity_data/Day1_DATASET/dataset_2024_06_03_Construction_Reverse/camera"
image_dir = "mcity_data/data/images"
labeled_image_dir = "mcity_data/data/labeled_images"
label_dir = "mcity_data/data/labels"

# Create directories if not exist
for d in ['train', 'val', 'test']:
    os.makedirs(os.path.join(image_dir, d), exist_ok=True)
    os.makedirs(os.path.join(label_dir, d), exist_ok=True)

# Function to get the next file number
def get_next_file_number(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    if not files:
        return 1
    files.sort()
    last_file = files[-1]
    return int(last_file.split('.')[0]) + 1

# Split ratio
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Get the last file numbers
train_num = get_next_file_number(os.path.join(image_dir, 'train'))
val_num = get_next_file_number(os.path.join(image_dir, 'val'))
test_num = get_next_file_number(os.path.join(image_dir, 'test'))

# Process images
for file in os.listdir(base_dir):
    image_path = os.path.join(base_dir, file)
    if not file.endswith('.png'):
        continue

    start = time.time()
    predictions = base_model.predict(image_path)
    print(time.time() - start)

    image = cv2.imread(image_path)

    labels = [
        label_map[class_id]
        for class_id in predictions.class_id
    ]

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=predictions)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions, labels=labels)

    # Determine where to save (train, val, or test)
    rand = random.random()
    if rand < train_ratio:
        subset = 'train'
        num = train_num
        train_num += 1
    elif rand < train_ratio + val_ratio:
        subset = 'val'
        num = val_num
        val_num += 1
    else:
        subset = 'test'
        num = test_num
        test_num += 1

    # Save the annotated image
    save_image_path = os.path.join(image_dir, subset, f'{num:06}.png')
    cv2.imwrite(save_image_path, image)
    save_labeled_image_path = os.path.join(labeled_image_dir, f'{num:06}.png')
    cv2.imwrite(save_labeled_image_path, annotated_image)

    # Save the label file in YOLO format
    save_label_path = os.path.join(label_dir, subset, f'{num:06}.txt')
    with open(save_label_path, 'w') as f:
        for class_id, bbox in zip(predictions.class_id, predictions.xyxy):
            x_center = (bbox[0] + bbox[2]) / 2 / image.shape[1]
            y_center = (bbox[1] + bbox[3]) / 2 / image.shape[0]
            width = (bbox[2] - bbox[0]) / image.shape[1]
            height = (bbox[3] - bbox[1]) / image.shape[0]
            f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

print("Processing complete.")
