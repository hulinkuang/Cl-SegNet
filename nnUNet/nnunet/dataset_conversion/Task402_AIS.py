#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from multiprocessing import Pool
import shutil
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
import numpy as np

test_ids = ['0073410', '0072723', '0226290', '0537908', '0538058', '0091415', '0538780', '0073540', '0226188',
            '0226258', '0226314', '0091507',
            '0226298', '0538975', '0226257', '0226142', '0072681', '0091538', '0538983', '0537961', '0091646',
            '0072765', '0226137', '0091621',
            '0091458', '0021822', '0538319', '0226133', '0091657', '0537925', '0073489', '0538502', '0091476',
            '0226136', '0538532', '0073312',
            '0539025', '0226309', '0226307', '0091383', '0021092', '0537990', '0226299', '0073060', '0538505',
            '0073424', '0091534', '0226125',
            '0072691', '0538425', '0226199', '0226261']


def get_subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = []
    for home, _, files in os.walk(folder):
        for filename in files:
            if os.path.isfile(os.path.join(home, filename)) \
                    and (prefix is None or filename.startswith(prefix)) \
                    and (suffix is None or filename.endswith(suffix)):
                res.append(l(home, filename))

    if sort:
        res.sort()
    return res


def zscore_norm(image):
    """ Normalise the image intensity by the mean and standard deviation """
    val_l = 0  # 像素下限
    val_h = 60
    roi = np.where((image >= val_l) & (image <= val_h))
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    image2 = np.copy(image).astype(np.float32)
    image2[image < val_l] = val_l  # val_l
    image2[image > val_h] = val_h

    eps = 1e-6
    image2 = (image2 - mu) / (sigma + eps)
    return image2


if __name__ == "__main__":
    train_dir = "/home/wyh/Datasets/AISD1/"

    output_folder = "/home/wyh/Codes/Coformer/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task402_AIS"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        assert data_file.split('/')[-2] == seg_file.split('/')[-2]
        pat_id = data_file.split('/')[-2]
        print(pat_id)

        shutil.copy(data_file, join(img_dir, pat_id + "_0000.nii.gz"))
        shutil.copy(seg_file, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id


    nii_files_data = sorted(get_subfiles(train_dir, True, "CT.nii.gz", "nii.gz", True))
    nii_files_seg = sorted(get_subfiles(train_dir, True, "GT_hard", "nii.gz", True))

    p = Pool(8)
    all_ids = p.map(load_save_train, zip(nii_files_data, nii_files_seg))
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "AISD"
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "0",
        "1": "1"
    }

    json_dict['numTraining'] = len(all_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             all_ids]
    json_dict['test'] = []

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    # create a dummy split (patients need to be separated)
    splits = [OrderedDict()]
    train = list(set(all_ids) - set(test_ids))
    train_ids, val_ids = train_test_split(train, test_size=0.118, shuffle=True, random_state=99)
    # train, val = train_test_split(train, test_size=0.125, shuffle=True, random_state=99)
    splits[-1]['train'] = train_ids
    splits[-1]['val'] = val_ids
    splits[-1]['test'] = test_ids
    splits[-1]['all'] = all_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
