import sys
import traceback
import os
from saicinpainting.evaluation.utils import move_to_device
from torchvision import transforms
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from PIL import Image
from PIL import ImageColor
import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import torch.fft
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
import json
import io
import pickle
from torchvision.transforms import ToPILImage
import boto3
import re
from PIL import Image
import itertools
from itertools import chain
import random
import shutil
import tempfile
from contextlib import redirect_stdout
import io
import os
import glob
import traceback
import copy
from itertools import product
# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s3_client = boto3.client('s3')

##########################Inference###################################

def model_fn(model_dir):
    with open(r'code/default.yaml') as file: 
        predict_config = yaml.safe_load(file)
   
    register_debug_signal_handlers()  
    predict_config['model']['path'] = 'big-lama' 
    train_config_path = os.path.join(predict_config['model']['path'], 'config.yaml') 
    with open(train_config_path, 'r') as f: 
        train_config = OmegaConf.create(yaml.safe_load(f)) 
    train_config.training_model.predict_only = True 
    train_config.visualizer.kind = 'noop' 

    out_ext = '.png' 
    checkpoint_path = os.path.join(predict_config['model']['path'],'models',predict_config['model']['checkpoint'])
    print('------------load checkpoint!!!!-----------------------')
    with redirect_stdout(io.StringIO()) as f:
    
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')        
        model.freeze() 
        model.to(device) 
    print('------------finish loading checkpoint!!!!-----------------------')
    
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    
    tempdir = tempfile.mkdtemp(dir='/tmp')
    print("TEMPDIR IS: ",tempdir)
    directory_im=f'{tempdir}/Image' 
    # Create the directory
    try:
        os.makedirs(directory_im, exist_ok = True)
        print("Directory '%s' created successfully" % directory_im)
    except OSError as error:
        print("Directory '%s' can not be created" % directory_im)

    # 2 Create directory for masks
    dir_UIMask=f'{tempdir}/UIMask' 
    try:
        os.makedirs(dir_UIMask, exist_ok = True)
        print("Directory '%s' created successfully" % dir_UIMask)
    except OSError as error:
        print("Directory '%s' can not be created" % dir_UIMask)

    # 3 Create directory for transformed masks
    dir_input_mask=f'{tempdir}/InputMask' 
    try:
        os.makedirs(dir_input_mask, exist_ok = True)
        print("Directory '%s' created successfully" % dir_input_mask)
    except OSError as error:
        print("Directory '%s' can not be created" % dir_input_mask)   

    # 4 Create directory for training dataset
    dir_train=f'{tempdir}/Train'  
    try:
        os.makedirs(dir_train, exist_ok = True)
        print("Directory '%s' created successfully" % dir_train)
    except OSError as error:
        print("Directory '%s' can not be created" % dir_train)

    # 5 Create directory for training dataset
    dir_test=f'{tempdir}/Test' 
    try:
        os.makedirs(dir_test, exist_ok = True)
        print("Directory '%s' created successfully" % dir_test)
    except OSError as error:
        print("Directory '%s' can not be created" % dir_test)
    
    directories=[tempdir,directory_im,dir_UIMask,dir_input_mask,dir_train,dir_test]

    input_json = json.loads(request_body)
    s3_image_list=input_json['input-image-location']
    mask_meta_list=input_json['mask-info']
    output_bucket=input_json['output-bucket']
    output_project=input_json['output-project']
    
    #Download images
    print("Downloading images")
    for i in range(len(input_json['input-image-location'])):
        image_location=input_json['input-image-location'][i]
        s1=image_location.split('s3://')[1]
        image_bucket=s1.split('/')[0]
        im_s3=s1.split(image_bucket+'/')[1]
        s3 = boto3.client('s3')
        with open(directory_im+'/'+str(i)+'_im.jpg', 'wb') as f:
            s3.download_fileobj(image_bucket, im_s3, f)
    #Image list
    imglist= [file for file in os.listdir(directory_im) if file.endswith('.jpg')]
    
    #Download masks
    print("Downloading masks")
    colors = []
    single_mask_list = []
    defect_name_list=[]
    hex_color_list=[]
    
    mask_location=input_json['mask-location']
    print('mask location:',mask_location)
    s1=mask_location.split('s3://')[1]
    mask_bucket=s1.split('/')[0]
    mask_s3=s1.split(mask_bucket+'/')[1]
    mask_raw_name=mask_s3.split('mask/')[1]
    print('mask s3:',mask_s3)
    print('mask name:',mask_raw_name)
    
    for i in range(len(input_json['mask-info'])-1):
        
        color_tmp = ImageColor.getcolor(input_json['mask-info'][f'{i+1}']['hex-color'], "RGB")
        
        #change that to BGR
        color_tmp2 = (color_tmp[2],color_tmp[1],color_tmp[0])
        colors.append(color_tmp2)
        
        defect_name=input_json['mask-info'][f'{i+1}']['class-name']
        defect_name_list.append(defect_name)
        hex_color_list.append(input_json['mask-info'][f'{i+1}']['hex-color'])
        
        
        s3_client.download_file(mask_bucket,mask_s3,f'{dir_UIMask}/mask_{defect_name}.png')
        single_mask_list.append(f'mask_{defect_name}.png')
   
    #Transform single masks for ML inference
    for i in range(len(single_mask_list)):
        mask= cv2.imread(dir_UIMask+'/'+single_mask_list[i])
        mask[np.logical_not(np.all(mask == colors[i], axis=-1))] = (255, 255, 255)
        #added this so i can create the mask as is expected by the service
        cv2.imwrite(dir_UIMask+'/'+single_mask_list[i], mask)
        mask[np.logical_not(np.all(mask == colors[i], axis=-1))] = (0, 0, 0)
        mask[np.all(mask == colors[i], axis=-1)] = (255, 255,  255)        
        cv2.imwrite(dir_input_mask+'/'+single_mask_list[i], mask)
    
    #number of mask
    mask_num=len(input_json['mask-info'])-1
    #Generate all combinations of masks
    all_comb=[]
    for i in range(2,mask_num+1):
        comb_list=list(itertools.combinations(range(mask_num), i))
        all_comb.append(comb_list)
    all_comb=list(chain(*all_comb))
    
    #Combine masks and save them in UIMask
    for i in range(len(all_comb)):
        class1=defect_name_list[all_comb[i][0]]
        class2=defect_name_list[all_comb[i][1]]
        mask_a = cv2.imread(dir_UIMask+'/'+'mask_'+class1+'.png')
        mask_b = cv2.imread(dir_UIMask+'/'+'mask_'+class2+'.png')
        mask_bwa = cv2.bitwise_and(mask_a,mask_b)
        
        class_name=str()
        for j in range(len(all_comb[i])-2):
            class_new=defect_name_list[all_comb[i][j+2]]
            mask_c = cv2.imread(dir_UIMask+'/'+'mask_'+class_new+'.png')
            mask_bwa = cv2.bitwise_and(mask_bwa,mask_c)
            class_name=class_name+'_'+class_new
        cv2.imwrite(dir_UIMask+'/'+"mask_"+class1+'_'+class2+class_name+'.png', mask_bwa)
        
    
    #combined mask list
    masklist= [file for file in os.listdir(dir_UIMask) if file.endswith('.png')]
    #upload mask to s3
    for i in range(len(masklist)):
        s3_client.upload_file(dir_UIMask+'/'+masklist[i],output_bucket,output_project+'/mask/MixMask/'+masklist[i])
        
    #Transform combined masks and save them in InputMask directory
    for i in range(len(all_comb)):
        class1=defect_name_list[all_comb[i][0]]
        class2=defect_name_list[all_comb[i][1]]

        mask_a = cv2.imread(dir_UIMask+'/'+'mask_'+class1+'.png')
        mask_a[np.all(mask_a == (255,255,255), axis=-1)] = (0, 0, 0)
        mask_a[np.all(mask_a != (0,0,0), axis=-1)] = (255, 255,  255)

        mask_b = cv2.imread(dir_UIMask+'/'+'mask_'+class2+'.png')
        mask_b[np.all(mask_b == (255,255,255), axis=-1)] = (0, 0, 0)
        mask_b[np.all(mask_b != (0,0,0), axis=-1)] = (255, 255,  255)

        mask_bwa = cv2.bitwise_or(mask_a,mask_b)


        class_name=str()
        for j in range(len(all_comb[i])-2):
            class_new=defect_name_list[all_comb[i][j+2]]
            mask_c = cv2.imread(dir_UIMask+'/'+'mask_'+class_new+'.png')
            mask_c[np.all(mask_c == (255,255,255), axis=-1)] = (0, 0, 0)
            mask_c[np.all(mask_c != (0,0,0), axis=-1)] = (255, 255,  255)
            mask_bwa = cv2.bitwise_or(mask_bwa,mask_c)
            class_name=class_name+'_'+class_new
        cv2.imwrite(dir_input_mask+'/'+"mask_"+class1+'_'+class2+class_name+'.png', mask_bwa)
    return(imglist,single_mask_list,masklist,mask_meta_list,output_bucket,output_project,mask_num,s3_image_list,directories,defect_name_list,hex_color_list)
    

def load_image(fname, mode='RGB', return_orig=False, auto_resize = 2000):
    img_ori = Image.open(fname).convert(mode)
    h=img_ori.size[0]
    w=img_ori.size[1]
    img_ori_2 = copy.deepcopy(img_ori)
    if h > auto_resize or w > auto_resize:
        ratio1=h/auto_resize
        ratio2=w/auto_resize
        ratio=max(ratio1,ratio2)
        h_resize=int(h/ratio)
        w_resize=int(w/ratio)

        print(h_resize,w_resize)
        img_ori_2 = img_ori.resize((h_resize, w_resize))
    else:
        h_resize=h
        w_resize=w
    img = np.array(img_ori_2)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, h, w, h_resize, w_resize, img_ori #img
    else:
        return out_img, h, w, h_resize, w_resize

def syn_gen(model,dataset_type,dir_dataset,directory_im,dir_input_mask,imname,maskname,output_bucket,output_project):
    im_short_name = (imname.split('.jpg'))[0]
    imlocation = directory_im+'/'+imname
    mask_short_name = (maskname.split('.png'))[0]
    masklocation = dir_input_mask + '/' +  maskname
    #load image and mask 
    image, h, w,h_resize,w_resize, im_ori = load_image(imlocation, mode='RGB', return_orig=True)
    im_ori = np.array(im_ori)
    mask,_,_,_,_,mask_ori =  load_image(masklocation, mode='L', return_orig=True)
    mask_ori = np.array(mask_ori)
    result = dict(image=image, mask=mask[None, ...])
    #perform ML inference 
    with torch.no_grad():
        img = torch.from_numpy(result['image']).unsqueeze(0)
        mask = torch.from_numpy(result['mask']).unsqueeze(0)
        mask = (mask > 0) * 1
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        out1 = model.generator(masked_img.to(device))
        out1 = out1.squeeze_(0)
        outimg = ToPILImage()(out1)
        outimg = outimg.crop((0,0,h_resize,w_resize))
        outimg = outimg.resize((h,w))

        new_im_name = im_short_name+'_'+mask_short_name+'.jpg'
        print('Generate synthetic defect:')
        print(new_im_name)
        print('is done!')
        pixels = np.where(mask_ori == 255)
        im_ori[pixels[0],pixels[1],:] = np.array(outimg)[pixels[0],pixels[1],:]
        im_ori_pil = Image.fromarray(im_ori)
        im_ori_pil.save(dir_dataset+'/'+new_im_name)
        # upload generated synthetic defects to s3
        s3_client.upload_file(dir_dataset+'/'+new_im_name,output_bucket,f'{output_project}/anomaly/{dataset_type}/{new_im_name}')
        #remove defects 
        os.remove(dir_dataset+'/'+new_im_name)
    return new_im_name

def predict_fn(input_object, model):
    
    imglist = input_object[0]
    single_mask_list = input_object[1]
    masklist = input_object[2] 
    mask_meta_list = input_object[3]
    output_bucket = input_object[4]
    output_project = input_object[5]
    mask_num = input_object[6]
    s3_image_list=input_object[7]
    directories = input_object[8]
    defect_name_list=input_object[9]
    hex_color_list=input_object[10]
        
    directory_im = directories[1]
    dir_UIMask = directories[2]
    dir_input_mask = directories[3]
    dir_train = directories[4]
    dir_test = directories[5]
    
  
    # Split the train and test image dataset
    if len(imglist)<20:
        print('There should be at least 20 normal images!')
    else:
        if int(len(imglist)*0.2)>10:
            len_train=len(imglist)*0.8
            len_test=len(imglist)*0.2
        else:
            len_test=10
            len_train=len(imglist)-10
        if len_train>40:
            len_train=40
            len_test=len(imglist)-40
        if len_test>40:
            len_test=40

    # Split the image dataset
    train_normal_list=random.sample(imglist, len_train)
    test_normal_list_all=list(set(imglist) - set(train_normal_list))
    test_normal_list=random.sample(test_normal_list_all, len_test)
    
    # Upload the normal images to s3
    for i in range(len(train_normal_list)):
        s3_client.upload_file(directory_im+'/'+train_normal_list[i],output_bucket,output_project+'/normal/train/'+train_normal_list[i])
    
    for j in range(len(test_normal_list)):
        s3_client.upload_file(directory_im+'/'+test_normal_list[j],output_bucket,output_project+'/normal/test/'+test_normal_list[j])
    
    print("Generating training Images")
    train_anomaly_list=[]
    for i in range(len(single_mask_list)):
        for j in range(len(train_normal_list)):
            try: 
                # get image name 
                train_imname = train_normal_list[j]
                # get mask name 
                train_maskname = single_mask_list[i]
               
                # ML inference
                train_defect=syn_gen(model,'train',dir_train,directory_im,dir_input_mask,train_imname,train_maskname,output_bucket,output_project)
                train_anomaly_list.append(train_defect)    
                print('Generating training synthetic defects is done!')
            except Exception:
                print(traceback.format_exc())
                # or
                print(sys.exc_info()[2])
                print('Error for train dataset!')          

    print('Train anomaly list',train_anomaly_list)
    
    # Generate test dataset
    print("Generating testing Images")
    test_anomaly_list=[]
    
    produc_list_test = list(product(test_normal_list,single_mask_list))    
    test_perm_list = np.random.permutation(produc_list_test)
    
    # Get the combinatino of masks
    comb_mask_list=list(set(masklist) - set(single_mask_list))
    
    produc_list = list(product(imglist,comb_mask_list))        
    rand_perm_list = np.random.permutation(produc_list)
    num_test_images = min(20,len(rand_perm_list))
    test_comb_list = rand_perm_list[0:num_test_images]        
    
    test_im_mask_list=list(test_perm_list) +list(test_comb_list)
 
    for i in range(len(test_im_mask_list)):
        try:
            # get image name 
            test_imname =test_im_mask_list[i][0]
            # get mask name      
            test_maskname=test_im_mask_list[i][1]
            # ML inference
            test_defect=syn_gen(model,'test',dir_test,directory_im,dir_input_mask,test_imname,test_maskname,output_bucket,output_project)
            test_anomaly_list.append(test_defect)
            print('Generating test synthetic defects is done!')
        except:
            print(traceback.format_exc())
            # or
            print(sys.exc_info()[2])
            print('Error for test dataset!') 

    print('Test anomaly list',test_anomaly_list)
    return (train_anomaly_list,test_anomaly_list,mask_meta_list,output_bucket,output_project,s3_image_list,directories,train_normal_list,test_normal_list,defect_name_list,hex_color_list)

def write_manifest(tempdir,dataset_type,dataset_anomaly_list,dataset_normal_list,output_bucket,output_project,defect_name_list,hex_color_list):
    manifest_name=f'{tempdir}/l4v_{dataset_type}.manifest' 
    with open(manifest_name, 'w') as the_new_file:
        for i in range(len(dataset_anomaly_list)):
            new_json = {}
            new_json['source-ref'] =f's3://{output_bucket}/{output_project}/anomaly/{dataset_type}/{dataset_anomaly_list[i]}'
            new_json['auto-label']=11
            new_json['auto-label-metadata']={"class-name": "anomaly","type": "groundtruth/image-classification"}
            
            #get defect class from mask name
            s=dataset_anomaly_list[i]
            t1=s.split('.jpg')[0]
            t2=t1.split('im_')[1]
            mask_name=t2+'.png' 
            new_json['anomaly-mask-ref']=f's3://{output_bucket}/{output_project}/mask/MixMask/'+mask_name

            s1=s.split('.jpg')[0]
            s2=s1.split('mask')[1]
            s3=s2.split('_')
            s4=s3[1:]
            metadata={}
            metadata[0]={'class-name': 'BACKGROUND', 'hex-color': '#ffffff', 'confidence': 0}

            for j in range(len(s4)):
                class_index=defect_name_list.index(s4[j])
                newdefect={}
                newdefect['class-name']=s4[j]
                newdefect['hex-color']=hex_color_list[class_index]
                newdefect['confidence']=0
                metadata[class_index]=newdefect

            new_json['anomaly-mask-ref-metadata']={}
            new_json['anomaly-mask-ref-metadata']['internal-color-map']=metadata
            new_json['anomaly-mask-ref-metadata']['type']="groundtruth/semantic-segmentation"
            
            the_new_file.write(json.dumps(new_json))
            the_new_file.write('\n')
        
        #add normal data for train dataset
        normal_json = {}
        
        for i in range(len(dataset_normal_list)):
            normal_json['source-ref']=f's3://{output_bucket}/{output_project}/normal/{dataset_type}/{dataset_normal_list[i]}'
            normal_json["auto-label"]=12
            normal_json["auto-label-metadata"]={"class-name": "normal","type": "groundtruth/image-classification"}
            the_new_file.write(json.dumps(normal_json))
            the_new_file.write('\n')
                                     
    s3_client.upload_file(manifest_name,output_bucket,output_project+f'/manifest/{dataset_type}/l4v_{dataset_type}.manifest')
                       
def output_fn(output, content_type):
    print("On the output function")
   
    train_anomaly_list=output[0]
    test_anomaly_list=output[1]
    mask_meta_list=output[2]
    output_bucket=output[3]
    output_project=output[4]
    s3_image_list=output[5]
    directories = output[6]
    train_normal_list=output[7]
    test_normal_list=output[8]
    defect_name_list=output[9]
    hex_color_list=output[10]
        
    tempdir = directories[0]
    directory_im = directories[1]
    dir_UIMask = directories[2]
    dir_input_mask = directories[3]
    dir_train = directories[4]
    dir_test = directories[5]
 
    #clean up all the directories
    shutil.rmtree(directory_im)
    shutil.rmtree(dir_UIMask)
    shutil.rmtree(dir_input_mask)
    
    # Create manifest for train dataset
    write_manifest(tempdir,'train',train_anomaly_list,train_normal_list,output_bucket,output_project,defect_name_list,hex_color_list)
    print('Training manifest is generated!')
    
    # Create manifest for test dataset
    write_manifest(tempdir,'test',test_anomaly_list,test_normal_list,output_bucket,output_project,defect_name_list,hex_color_list)
    print('Test manifest is generated!')
    
    # Remove tempdir
    shutil.rmtree(tempdir)
      
    return "done"
                

    
    


