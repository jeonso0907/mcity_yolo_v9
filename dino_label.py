import os
import cv2
import supervision as sv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import time

# definitions = {
#     "construction barrel": "barrel",
#     "stop sign": "stop",
#     "pedestrian": "pedestrian",
#     "pedestrian crossing sign": "pedestrian crossing sign",
#     "speed limit sign": "speed limit",
#     "yield sign": "yield"
# }
definitions = {
    "blue pedestrian": "pedestrian",
    "brown deer": "deer",
    "round red and white striped barrel": "barrel",
    "non static color striped type III barricade": "barricade",
    "traffic light": "traffic light"
}
# label_map = {
#     0: "barrel",
#     1: "stop",
#     2: "pedestrian",
#     3: "pedestrian crossing sign",
#     4: "speed limit",
#     5: "yield"
# }
label_map = {
    0: "pedestrian",
    1: "deer",
    2: "barrel",
    3: "barricade",
    4: "traffic light"
}
base_model = GroundingDINO(ontology=CaptionOntology(definitions))

base_dir = "Mcity"
for file in os.listdir(base_dir):
    image_path = f"{base_dir}/{file}"
    if "label" in image_path:
        continue

    start = time.time()
    predictions = base_model.predict(image_path)
    print(time.time() - start)
    print(predictions)

    image = cv2.imread(image_path)

    labels = [
        label_map[class_id]
        for class_id
        in predictions.class_id
    ]

    annotator = sv.BoxAnnotator()
    annotated_image = annotator.annotate(scene=image, detections=predictions)
    annotated_image = annotator.annotate(scene=annotated_image, detections=predictions, labels=labels)

    cv2.imwrite(f"{image_path[:-3]}_label.png", annotated_image)