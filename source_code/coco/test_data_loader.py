import torch.utils.data as data
import os
import torch
from PIL import Image
from torchvision.transforms import transforms


class COCO_MLC(data.Dataset):

    def __init__(self, path, transform):
        assert os.path.exists(path)

        self.img_files = []
        self.transform = transform

        for root, dirs, files in os.walk(path):
            for file in files:
                self.img_files.append(os.path.join(root, file))

    def __getitem__(self, index):
        # TODO return ground truth and result.
        img_path = self.img_files[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, os.path.basename(img_path)

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    input_size = 448
    val_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])
    cocomlc = COCO_MLC("../coco_img/bg2_0/", val_transform)

    test_loader = torch.utils.data.DataLoader(cocomlc, batch_size=16,
                                              shuffle=False, num_workers=4,
                                              pin_memory=True)
    # for f in enumerate(cocomlc)
    for batch_idx, (imgs, paths) in enumerate(test_loader):
        print(batch_idx)
        print(imgs.shape)
        print(len(paths))
        print(paths[0])

