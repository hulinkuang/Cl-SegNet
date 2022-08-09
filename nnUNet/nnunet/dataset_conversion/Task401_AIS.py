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

import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
import numpy as np


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
    train_dir = "/home/wyh/Codes/dataset/AIS"

    output_folder = "/home/wyh/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task401_AIS"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file, f_id = args
        description = data_file.split('/')[-2]
        pat_id = f_id[description]
        pat_id = "AIS" + description + "_" + str(pat_id)

        img_itk_ = sitk.ReadImage(data_file)
        img_arr = sitk.GetArrayFromImage(img_itk_)
        img_arr = zscore_norm(img_arr)
        img_itk = sitk.GetImageFromArray(img_arr)
        img_itk.SetSpacing(img_itk_.GetSpacing())
        img_itk.SetOrigin(img_itk_.GetOrigin())
        img_itk.SetDirection(img_itk_.GetDirection())
        sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))

        seg_itk = sitk.ReadImage(seg_file)
        seg_arr = sitk.GetArrayFromImage(seg_itk)
        seg_arr[seg_arr == 4] = 0
        seg_arr[seg_arr > 0] = 1
        itk = sitk.GetImageFromArray(seg_arr)
        itk.SetSpacing(seg_itk.GetSpacing())
        itk.SetOrigin(seg_itk.GetOrigin())
        itk.SetDirection(seg_itk.GetDirection())
        sitk.WriteImage(itk, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id


    def load_save_test(args):
        data_file, f_id = args
        description = data_file.split('/')[-2]
        pat_id = f_id[description]
        pat_id = "KSR" + description + "_" + str(pat_id)

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
        return pat_id

    def get_pat_id(args):
        data_file, f_id = args
        description = data_file.split('/')[-2]
        pat_id = f_id[description]
        pat_id = "KSR" + description + "_" + str(pat_id)

        return pat_id


    nii_files_data = get_subfiles(train_dir, True, "CT", "nii.gz", True)
    nii_files_seg = get_subfiles(train_dir, True, "GT", "nii.gz", True)
    file_id = OrderedDict()
    for k, v in zip(nii_files_data, range(1, len(nii_files_data) + 1)):
        file_id[k.split('/')[-2]] = v

    p = Pool(default_num_threads)
    all_ids = p.map(load_save_train, zip(nii_files_data, nii_files_seg, [file_id] * len(nii_files_data)))
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "KSR"
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
    train, test = train_test_split(all_ids, test_size=0.203, shuffle=True, random_state=99)
    train, val = train_test_split(train, test_size=0.125, shuffle=True, random_state=99)
    splits[-1]['train'] = train
    splits[-1]['val'] = val
    splits[-1]['test'] = test
    splits[-1]['all'] = all_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
