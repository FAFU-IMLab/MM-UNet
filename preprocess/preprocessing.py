# -*- coding: utf-8 -*-
import numpy as np
import os
import SimpleITK as sitk



def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp


def crop_ceter(img,croph,cropw):   
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape 
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)        
    return img[:,starth:starth+croph,startw:startw+cropw]


def file_name_path(file_dir, dir=True, file=False):

#"    get root path,sub_dirs,all_sub_files"
#"    :param file_dir:"
#"    :return: dir or file"

    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:/", dirs)
            return dirs
        if len(files) and file:
            print("files:/", files)
            return files

def datareader(txtdir):
    file = open(txtdir,'r')
    a = file.readlines()
    for i in range(0,len(a)):
        a[i] = a[i].strip('\n')
    return a
#输出train和label的路径
outputImg_path = r'/media/twh/软件1/czs/2d/valImage'
outputMask_path = r'/media/twh/软件1/czs/2d/valMask'

if not os.path.exists(outputImg_path):
    os.makedirs(outputImg_path)
if not os.path.exists(outputMask_path):
    os.makedirs(outputMask_path)

#输入train的数据集
bratslgg_path = r'/media/twh/软件1/czs/2d/Val'
#--------
#bratshgg_path = r'/media/twh/F6CCEC1FCCEBD7BF/czs/medical/2d/HGG'
#bratslgg_path = r'/media/twh/F6CCEC1FCCEBD7BF/czs/medical/2d/LGG'
#--------
#train_img_paths = datareader(r'/home/sophia/桌面/BraTS/BraTS2020/itrain_811.txt')

# pathlgg_list 是 HGG文件夹底下所有子文件夹的名字
pathlgg_list = []
#------
#pathgg_list = file_name_path(bratshgg_path)
#pathlgg_list = file_name_path(bratslgg_path)
#------
for roots, dirs, files in os.walk(bratslgg_path):
    for i in dirs:
        pathlgg_list.append(i)


flair_name = str('_flair.nii.gz')
t1_name = str('_t1.nii.gz')
t1ce_name = str('_t1ce.nii.gz')
t2_name = str('_t2.nii.gz')
mask_name = str('_seg.nii.gz')

'''
每个病例有四个模态(flair t1  t1ce t2),需要割3个部分:WT ET TC
flair t1  t1ce t2 为MRI的四个不同纬度信息,每个序列的图像shape为(155,240,240) C H W
目标是分割出三个label,对应医学中的三个不同肿瘤类型
BraTs数据集类型为XX.nii.gz, 分别对应flair t1  t1ce t2 seg,其中seg是分割图像
'''
for subsetindex in range(len(pathlgg_list)):
    # imagename =  train_img_paths[subsetindex].split('/')[-1]
    # print(imagename)
    # train_paths = train_img_paths[subsetindex] + '/' + imagename

    brats_subset_path = bratslgg_path + '/' + str(pathlgg_list[subsetindex]) + '/'
    print('brats_subset_path:' +brats_subset_path)
	#获取每个病例的四个模态及Mask的路径,
    flair_image = brats_subset_path + str(pathlgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path+ str(pathlgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t2_name
    mask_image = brats_subset_path + str(pathlgg_list[subsetindex]) + mask_name

    # flair_image = train_paths + flair_name
    # t1_image = train_paths + t1_name
    # t1ce_image = train_paths + t1ce_name
    # t2_image = train_paths + t2_name
    # mask_image = train_paths + mask_name

    #获取每个病例的四个模态及Mask数据\n",
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
    #GetArrayFromImage()可用于将SimpleITK对象转换为ndarray,
    flair_array = sitk.GetArrayFromImage(flair_src)  
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    mask_array = sitk.GetArrayFromImage(mask)
    #对四个模态分别进行标准化,由于它们对比度不同,
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    #裁剪(偶数才行),原图大小是240*240*155
    flair_crop = crop_ceter(flair_array_nor,160,160)
    t1_crop = crop_ceter(t1_array_nor,160,160)
    t1ce_crop = crop_ceter(t1ce_array_nor,160,160)
    t2_crop = crop_ceter(t2_array_nor,160,160)
    mask_crop = crop_ceter(mask_array,160,160)
    #------
    #print(str(pathlgg_list[subsetindex]))
    #------
    #切片处理,并去掉没有病灶的切片
    # flair_crop.shape[0] 表示有切片张数
    for n_slice in range(flair_crop.shape[0]):
        if np.max(mask_crop[n_slice,:,:]) != 0:
            maskImg = mask_crop[n_slice,:,:]
            # 生成一个空的[160,160,4]的矩阵，然后把数值填进去
            FourModelImageArray = np.zeros((flair_crop.shape[1],flair_crop.shape[2],4),np.float)
            #fair图像提取，转float，填到矩阵
            flairImg = flair_crop[n_slice,:,:]
            flairImg = flairImg.astype(np.float)
            FourModelImageArray[:,:,0] = flairImg
            #t1图像提取，转float，填到矩阵
            t1Img = t1_crop[n_slice,:,:]
            t1Img = t1Img.astype(np.float)
            FourModelImageArray[:,:,1] = t1Img
            #t1ce图像提取，转float，填到矩阵
            t1ceImg = t1ce_crop[n_slice,:,:]
            t1ceImg = t1ceImg.astype(np.float)
            FourModelImageArray[:,:,2] = t1ceImg
            #t2图像提取，转float，填到矩阵
            t2Img = t2_crop[n_slice,:,:]
            t2Img = t2Img.astype(np.float)
            FourModelImageArray[:,:,3] = t2Img

            imagepath = outputImg_path + '/' + str(pathlgg_list[subsetindex]) +'_' + str(n_slice) + '.npy'
            maskpath = outputMask_path + '/' + str(pathlgg_list[subsetindex]) + '_' + str(n_slice) + '.npy'
            np.save(imagepath,FourModelImageArray)#(160,160,4) np.float dtype('float64')\n",
            np.save(maskpath,maskImg)# (160, 160) dtype('uint8') 值为0 1 2 4\n",
"print(\"Done!\")"
