from prepDataset import convert_dataset, merge_info, load_categories
import os
import os, torch, torchvision
from PIL import Image
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

old_dir="UECFOOD100"
root="UECFOOD100merged"
convert_dataset(old_dir,root)

bbox_info = merge_info(old_dir)
bbox_info.to_csv(os.path.join(root,'bbox.csv'), index = False)

class foodDataset(object):
    def __init__(self, root, transforms, bbox_info):
        self.root = root
        self.transforms = transforms
        self.bbox_info = bbox_info
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        obj_ids = self.bbox_info.loc[(
                self.bbox_info['img'] == int(self.imgs[idx][:-4]))]
        num_objs = len(obj_ids)
        boxes = []
        for _, bbox in obj_ids.iterrows():
            boxes.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids.category.tolist(), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_object_detection(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    root = "UECFOOD100merged"
    img_dir = "Images"

    bbox_info = pd.read_csv(os.path.join(root, 'bbox.csv'))
    categ_labels = load_categories(root)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 100 food classes + background
    num_classes = len(categ_labels) + 1

    img_path = os.path.join(root, img_dir)

    # use our dataset and defined transformations
    dataset = foodDataset(img_path, get_transform(train=True), bbox_info)
    dataset_test = foodDataset(img_path, get_transform(train=False), bbox_info)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_object_detection(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    torch.save(model,"model.pth")