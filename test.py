import os
import time
import os.path as osp
import cv2
import torch
from datasets.change_convert import Change_Convert
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import time
import shutil
from PIL import ImageFile


def get_same_img(img_dir, img_name):
    result = {}
    for idx, name in enumerate(img_name):
        temp_name = ''
        for idx2, item in enumerate(name.split('_')[:-4]):
            if idx2 == 0:
                temp_name = temp_name + item
            else:
                temp_name = temp_name + '_' + item

        if temp_name in result:
            result[temp_name].append(img_dir[idx])
        else:
            result[temp_name] = []
            result[temp_name].append(img_dir[idx])
    return result


def get_file_names(data_dir, file_type='tif'):
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = maindir + '/' + filename
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name


def combine(data_dir, w_list, h_list, c, out_dir, out_type='tif', file_type='tif'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_dir, img_name = get_file_names(data_dir, file_type)
    print('Combine begining for ', str(len(img_dir)), ' images.....')
    dir_dict = get_same_img(img_dir, img_name)
    count = 0
    for key in dir_dict.keys():
        if c == 3:
            temp_label = np.zeros(shape=(w_list[count], h_list[count], 3), dtype=np.uint8)
        else:
            temp_label = np.zeros(shape=(w_list[count], h_list[count]), dtype=np.uint8)
        dir_list = dir_dict[key]
        for item in dir_list:
            name_split = item.split('_')
            x_start = int(name_split[-4])
            x_end = int(name_split[-3])
            y_start = int(name_split[-2])
            y_end = int(name_split[-1].split('.')[0])
            img = Image.open(item)
            img = np.array(img)

            #  在不改变数据内容情况下，改变shape
            # img = np.reshape(img,img.shape+(1,))
            temp_label[x_start:x_end, y_start:y_end] = img

        img_name = key + '.' + out_type
        new_out_dir = out_dir + '/' + img_name

        label = Image.fromarray(temp_label)
        label.save(new_out_dir)
        # src_path = '.1/AOI.tif'  # 带地理坐标影像
        # assign_spatial_reference_byfile(src_path, new_out_dir)
        count += 1
        print('End of ' + str(count) + '/' + str(len(dir_dict)) + '...')
    print('Combine Finsh!')

    return 0


def cut(in_dir, out_dir, file_type='tif', out_type='tif', out_size=512):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_dir_list, _ = get_file_names(in_dir, file_type)
    count = 0
    h_list = []
    w_list = []
    print('Cut begining for ', str(len(data_dir_list)), ' images.....')
    for each_dir in data_dir_list:
        time_start = time.time()
        # image = np.array(io.imread(each_dir))
        image = np.array(Image.open(each_dir))
        h = image.shape[1]
        w = image.shape[0]
        h_list.append(h)
        w_list.append(w)
        print(image.shape)

        cut_factor_row = int(np.ceil(image.shape[0] / out_size))
        cut_factor_clo = int(np.ceil(image.shape[1] / out_size))
        for i in range(cut_factor_row):
            for j in range(cut_factor_clo):

                if i == cut_factor_row - 1:
                    i = image.shape[0] / out_size - 1
                else:
                    pass

                    if j == cut_factor_clo - 1:
                        j = image.shape[1] / out_size - 1
                    else:
                        pass

                start_x = int(np.rint(i * out_size))
                start_y = int(np.rint(j * out_size))
                end_x = int(np.rint((i + 1) * out_size))
                end_y = int(np.rint((j + 1) * out_size))

                # temp_image = image[start_x:end_x, start_y:end_y, :]
                temp_image = image[start_x:end_x, start_y:end_y]
                # print('temp_image:', temp_image.shape)

                out_dir_images = out_dir + '/' + each_dir.split('/')[-1].split('.')[0] \
                                 + '_' + str(start_x) + '_' + str(end_x) + '_' + str(start_y) + '_' + str(
                    end_y) + '.' + out_type

                out_image = Image.fromarray(temp_image)
                out_image.save(out_dir_images)

                # src_path = './cut/geo.tif'  # 带地理影像
                # assign_spatial_reference_byfile(src_path, out_dir_images)

        count += 1
        print('End of ' + str(count) + '/' + str(len(data_dir_list)) + '...')
        time_end = time.time()
        print('Time cost: ', time_end - time_start)
    print('Cut Finsh!')
    return h_list, w_list, count


def img_cut(test_dir='test_DATA', test_dir_cut='tmp/cut'):
    test_dir_A = os.path.join(test_dir, 'A')
    test_dir_B = os.path.join(test_dir, 'B')
    test_dir_Acut = os.path.join(test_dir_cut, 'A')
    test_dir_Bcut = os.path.join(test_dir_cut, 'B')
    if not os.path.exists(test_dir_cut):
        os.makedirs(test_dir_cut)
    if not os.path.exists(test_dir_Acut):
        os.makedirs(test_dir_Acut)
    if not os.path.exists(test_dir_Bcut):
        os.makedirs(test_dir_Bcut)
    h_list, w_list, countcut = cut(test_dir_A, test_dir_Acut, 'tif', 'tif', 512)
    h_list, w_list, countcut = cut(test_dir_B, test_dir_Bcut, 'tif', 'tif', 512)
    return h_list, w_list, countcut


def predict_img(DATA_DIR_via="", save_dir="", model_paths=None):
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    test_dataset = Change_Convert(DATA_DIR_via,
                                   sub_dir_1='A',
                                   sub_dir_2='B',
                                   img_suffix='.tif',
                                   ann_dir=None,
                                   size=512,
                                   debug=False,
                                   test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        best_model = torch.load(model_paths[0])
        best_model.eval()

        best_model_1 = torch.load(model_paths[1])
        best_model_1.eval()

        for (x1, filename) in tqdm(test_loader):
            x1 = x1.float()
            x1 = x1.to(DEVICE)

            y_pred = torch.zeros((1, 1, 512, 512)).cuda()

            with torch.no_grad():
                y_pred_0 = best_model.forward(x1)
                y_pred_1 = best_model_1.forward(x1)
                y_pred = (y_pred_0 + y_pred_1)/2

                y_pred[y_pred < 0.5] = 0
                y_pred[y_pred > 0.5] = 1
                y_pred = y_pred.squeeze().cpu().numpy().round()
                filename = filename[0].split('.')[0] + '.png'
                cv2.imwrite(osp.join(save_dir, filename), y_pred)

def main(input_path='/input_path', output_path='/output_path'):
    start = time.time()
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    os.makedirs('tmp')
    h_list, w_list, countcut = img_cut(test_dir=input_path, test_dir_cut='tmp/cut')
    predict_img(DATA_DIR_via='tmp/cut', save_dir="tmp/res", model_paths=['1225HR_netmodel/hrnet18,drp0.5_dice/k0.pth', '1225HR_netmodel/hrnet18,drp0.5_dice+bice/k0.pth'])
    combine('tmp/res', w_list, h_list, 1, output_path, 'png', 'png')
    end = time.time()
    print('time: ', end - start)


if __name__ == '__main__':
    main()
