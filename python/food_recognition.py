import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision import transforms as torchtrans
from PIL import Image

import transforms as T
"""First model shoulf downloaded from 
https://drive.google.com/file/d/1TiF4E_aFoLkuL76gmO9qZUILTTjAblZH/view?usp=sharing"""

def load_model(model_path, num_classes):
    model = torch.load(model_path)
    model.eval()
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def visualize_predictions(image_path, predictions, shrink_percentage):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in predictions[0]["boxes"]:
        x1, y1, x2, y2 = box.tolist()
        shrink_x = (x2 - x1) * shrink_percentage
        shrink_y = (y2 - y1) * shrink_percentage
        x1 += shrink_x
        x2 -= shrink_x
        y1 += shrink_y
        y2 -= shrink_y
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    plt.show()


def merge_overlapping_boxes(predictions, overlap_thresh):
    boxes = predictions[0]['boxes'].cpu().numpy()
    merged_indices = []
    for i in range(len(boxes)):
        if i in merged_indices:
            continue

        for j in range(i + 1, len(boxes)):
            if j in merged_indices:
                continue
            x1_i, y1_i, x2_i, y2_i = boxes[i]
            x1_j, y1_j, x2_j, y2_j = boxes[j]
            x_overlap = max(0, min(x2_i, x2_j) - max(x1_i, x1_j))
            y_overlap = max(0, min(y2_i, y2_j) - max(y1_i, y1_j))
            intersection_area = x_overlap * y_overlap
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            area_j = (x2_j - x1_j) * (y2_j - y1_j)
            min_area = min(area_i, area_j)
            if intersection_area > overlap_thresh * min_area:
                merged_box = [min(x1_i, x1_j), min(y1_i, y1_j), max(x2_i, x2_j), max(y2_i, y2_j)]
                boxes[i] = merged_box
                merged_indices.append(j)

    filtered_boxes = [box for i, box in enumerate(boxes) if i not in merged_indices]
    predictions[0]['boxes'] = torch.tensor(filtered_boxes, dtype=torch.float32)

    return predictions


def apply_nms(orig_prediction, iou_thresh):
    keep = torchvision.ops.nms(orig_prediction[0]['boxes'], orig_prediction[0]['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction[0]['boxes'] = final_prediction[0]['boxes'][keep]
    final_prediction[0]['scores'] = final_prediction[0]['scores'][keep]
    final_prediction[0]['labels'] = final_prediction[0]['labels'][keep]

    return final_prediction


def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')


def test_images(image_path):
    model_path = 'model.pth'
    model = load_model(model_path, 101)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_image = Image.open(image_path).convert("RGB")
    transform = get_transform(train=False)
    test_image, _ = transform(test_image, None)
    test_image = test_image.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(test_image)
    nms_prediction = apply_nms(predictions)

    merged_prediction = merge_overlapping_boxes(nms_prediction, overlap_thresh=0.5)
    visualize_predictions(image_path, merged_prediction, shrink_percentage=0.060)
