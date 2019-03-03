import unet3d.utils.args_utils as get_args
from projects.drive.config import config
import unet3d.utils.path_utils as path_utils
import unet3d.utils.print_utils as print_utils
import os
import glob
import shutil
from PIL import Image
from random import shuffle
import numpy as np
np.random.seed(1988)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(
    CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])
RAW_DIR = os.path.join(PROJECT_DIR, "raw/IDRID")


def clean_name(name):

    case = name.split('_')
    return case[0]


def get_manual_mask_name(name):
    haemorrhages = "haemorrhages/{}_HE.tif".format(name)
    hard_exudates = "hard_exudates/{}_EX.tif".format(name)
    soft_exudates = "soft_exudates/{}_SE.tif".format(name)
    microaneurysms = "microaneurysms/{}_MA.tif".format(name)
    return [microaneurysms, haemorrhages, soft_exudates, hard_exudates]


def read_resize_set_label(path, label, shape=(512,512), mode="L"):
    if os.path.exists(path):
        desired_size = 4288
        img = Image.open(path)

        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new(mode, (desired_size, desired_size))
        new_im.paste(img, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))

        # new_im.show()

        img = new_im.resize(shape, Image.ANTIALIAS)

        # img.show()

        img = np.array(img)
        
    else:
        img = np.zeros(shape).astype(np.uint8)

    if label is not None:
        return img*label
    else:
        return img


def build_seg(seg_paths, shape=(512,512)):
    microaneurysms = read_resize_set_label(seg_paths[0], 1, shape)
    haemorrhages = read_resize_set_label(seg_paths[1], 2, shape)
    soft_exudates = read_resize_set_label(seg_paths[2], 3, shape)
    hard_exudates = read_resize_set_label(seg_paths[3], 4, shape)

    seg = microaneurysms + haemorrhages + soft_exudates + hard_exudates
    seg[seg > 4] = 0

    seg_show = np.copy(seg)
    seg_show[seg_show == 1] = 50
    seg_show[seg_show == 2] = 120
    seg_show[seg_show == 3] = 180
    seg_show[seg_show == 4] = 255

    seg = Image.fromarray(np.uint8(seg))
    seg_show = Image.fromarray(np.uint8(seg_show))

    return seg, seg_show


def move_organize_training(dir_in, dir_out, dataset="train"):
    train_dir = os.path.join(dir_out, "data_train/original")
    valid_dir = os.path.join(dir_out, "data_valid/original")
    test_dir = os.path.join(dir_out, "data_test/original")

    images_dir = "{}/images/{}/".format(dir_in, dataset)
    gt_dir = "{}/groundtruth/{}/".format(dir_in, dataset)

    img_dirs = glob.glob(os.path.join(images_dir, "*"))

    if dataset == "train":
        shuffle(img_dirs)

    num_train = 38
    count = 0

    for img_dir in img_dirs:
        print(">> processing", img_dir)
        img_name = path_utils.get_filename_without_extension(img_dir)
        img_full_name = path_utils.get_filename(img_dir)

        img = read_resize_set_label(img_dir, None, shape=(512,512), mode="RGB")
        img = Image.fromarray(np.uint8(img))
        # img.show()

        seg_names = get_manual_mask_name(img_name)
        seg_paths = [gt_dir + s for s in seg_names]
        seg, seg_show = build_seg(seg_paths, shape=(512,512))    
        # seg.show()
        # seg_show.show()

        # img = Image.open(img_dir)
        # img = img.resize((512,256), Image.ANTIALIAS)
        # img.save('sompic.jpg')


        if dataset == "train" and count < num_train:
            img_path_dst = os.path.join(train_dir, img_name, "img.png")
            seg_path_dst = os.path.join(train_dir, img_name, "gt.png")
            
            if not os.path.exists(os.path.join(train_dir, img_name)):
                os.makedirs(os.path.join(train_dir, img_name))
            
            img.save(img_path_dst)
            seg.save(seg_path_dst)
            count = count + 1
        elif dataset == "train" and count >= num_train:
            img_path_dst = os.path.join(valid_dir, img_name, "img.png")
            seg_path_dst = os.path.join(valid_dir, img_name, "gt.png")
            
            if not os.path.exists(os.path.join(valid_dir, img_name)):
                os.makedirs(os.path.join(valid_dir, img_name))
            
            img.save(img_path_dst)
            seg.save(seg_path_dst)
            count = count + 1
        else:
            img_path_dst = os.path.join(test_dir, img_name, "img.png")
            seg_path_dst = os.path.join(test_dir, img_name, "gt.png")
            
            if not os.path.exists(os.path.join(test_dir, img_name)):
                os.makedirs(os.path.join(test_dir, img_name))
            
            img.save(img_path_dst)
            seg.save(seg_path_dst)
            count = count + 1          



def main():
    dir_in = RAW_DIR
    dir_out = os.path.join(PROJECT_DIR, "projects/idrid/database")

    move_organize_training(dir_in, dir_out, dataset="train")
    move_organize_training(dir_in, dir_out, dataset="test")


if __name__ == "__main__":
    main()
