import argparse
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence
from diffusers import AutoencoderKL
from face_alignment import NetworkSize
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from tqdm import tqdm

try:
    from utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    from musetalk.utils.face_parsing import FaceParsing

def get_file_type(file_path):
    """
    判断文件类型，支持视频、GIF和图片
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    if file_ext in video_exts:
        return 'video'
    elif file_ext == '.gif':
        return 'gif'
    else:
        return 'image'

def video2imgs(file_path, save_path, ext='.png', cut_frame=10000000):
    file_type = get_file_type(file_path)
    count = 0
    
    if file_type == 'video':
        cap = cv2.VideoCapture(file_path)
        while count < cut_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 创建PIL图像
            pil_image = Image.fromarray(rgb_frame)
            # 保存为PNG
            pil_image.save(f"{save_path}/{count:08d}.png", 'PNG')
            count += 1
        cap.release()
    
    elif file_type == 'gif':
        # 处理GIF文件
        gif = Image.open(file_path)
        for frame in ImageSequence.Iterator(gif):
            if count >= cut_frame:
                break
            # 转换为RGBA模式
            rgba_frame = frame.convert('RGBA')
            # 保存为PNG以保持透明通道
            rgba_frame.save(f"{save_path}/{count:08d}.png", 'PNG')
            count += 1

def read_imgs(img_list):
    frames = []
    alphas = []  # 存储透明通道
    print('reading images...')
    for img_path in tqdm(img_list):
        # 使用PIL读取图片以支持透明通道
        img = Image.open(img_path)
        has_alpha = img.mode == 'RGBA'
        
        if has_alpha:
            # 如果有Alpha通道，分离并保存
            alpha = img.split()[3]
            # 创建白色背景
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        else:
            alpha = None
            
        # 转换为RGB并转为numpy数组
        img = img.convert('RGB')
        frame = np.array(img)
        # 转换为BGR格式(OpenCV格式)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        alphas.append(alpha)
    return frames, alphas

def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames, alphas = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]  # 手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (
                np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]
    return coords_list, (frames, alphas)

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
            print('cuda start')

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)

        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))
        return results

def get_mask_tensor():
    """
    Creates a mask tensor for image processing.
    :return: A mask tensor.
    """
    mask_tensor = torch.zeros((256, 256))
    mask_tensor[:256 // 2, :] = 1
    mask_tensor[mask_tensor < 0.5] = 0
    mask_tensor[mask_tensor >= 0.5] = 1
    return mask_tensor

def preprocess_img(img_name, half_mask=False):
    window = []
    if isinstance(img_name, str):
        window_fnames = [img_name]
        for fname in window_fnames:
            # 使用PIL读取图片以支持透明通道
            img = Image.open(fname)
            if img.mode == 'RGBA':
                # 如果是RGBA模式，将透明背景填充为白色
                background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert('RGB')
            img = np.array(img)
            img = cv2.resize(img, (256, 256),
                             interpolation=cv2.INTER_LANCZOS4)
            window.append(img)
    else:
        if isinstance(img_name, np.ndarray):
            img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(img_name)
        window.append(img)
    x = np.asarray(window) / 255.
    x = np.transpose(x, (3, 0, 1, 2))
    x = torch.squeeze(torch.FloatTensor(x))
    if half_mask:
        x = x * (get_mask_tensor() > 0.5)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x = normalize(x)
    x = x.unsqueeze(0)  # [1, 3, 256, 256] torch tensor
    x = x.to(device)
    return x

def encode_latents(image):
    with torch.no_grad():
        init_latent_dist = vae.encode(image.to(vae.dtype)).latent_dist
    init_latents = vae.config.scaling_factor * init_latent_dist.sample()
    return init_latents

def get_latents_for_unet(img):
    ref_image = preprocess_img(img, half_mask=True)  # [1, 3, 256, 256] RGB, torch tensor
    masked_latents = encode_latents(ref_image)  # [1, 4, 32, 32], torch tensor
    ref_image = preprocess_img(img, half_mask=False)  # [1, 3, 256, 256] RGB, torch tensor
    ref_latents = encode_latents(ref_image)  # [1, 4, 32, 32], torch tensor
    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
    return latent_model_input

def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s

def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image

def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    if isinstance(image, np.ndarray):
        body = Image.fromarray(image[:, :, ::-1])
    else:
        body = image

    x, y, x1, y1 = face_box
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

current_dir = os.path.dirname(os.path.abspath(__file__))

def create_musetalk_human(file, avatar_id):
    # 保存文件设置 可以不动
    save_path = os.path.join(current_dir, f'../data/avatars/avator_{avatar_id}')
    save_full_path = os.path.join(current_dir, f'../data/avatars/avator_{avatar_id}/full_imgs')
    create_dir(save_path)
    create_dir(save_full_path)
    mask_out_path = os.path.join(current_dir, f'../data/avatars/avator_{avatar_id}/mask')
    create_dir(mask_out_path)

    # 模型
    mask_coords_path = os.path.join(current_dir, f'{save_path}/mask_coords.pkl')
    coords_path = os.path.join(current_dir, f'{save_path}/coords.pkl')
    latents_out_path = os.path.join(current_dir, f'{save_path}/latents.pt')

    with open(os.path.join(current_dir, f'{save_path}/avator_info.json'), "w") as f:
        json.dump({
            "avatar_id": avatar_id,
            "video_path": file,
            "bbox_shift": 5
        }, f)

    if os.path.isfile(file):
        file_type = get_file_type(file)
        if file_type in ['video', 'gif']:
            video2imgs(file, save_full_path, ext='png')
        else:
            # 如果是单张图片，直接复制
            shutil.copyfile(file, f"{save_full_path}/{os.path.basename(file)}")
    else:
        # 如果是目录，复制所有支持的图片
        files = os.listdir(file)
        files.sort()
        files = [f for f in files if f.split(".")[-1].lower() in ['png', 'jpg', 'jpeg', 'gif']]
        for filename in files:
            shutil.copyfile(f"{file}/{filename}", f"{save_full_path}/{filename}")

    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    print("extracting landmarks...")
    coord_list, (frame_list, alpha_list) = get_landmark_and_bbox(input_img_list, 5)
    input_latent_list = []
    idx = -1
    # maker if the bbox is not sufficient
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    for bbox, frame in zip(coord_list, frame_list):
        idx = idx + 1
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)

    frame_list_cycle = frame_list #+ frame_list[::-1]
    coord_list_cycle = coord_list #+ coord_list[::-1]
    input_latent_list_cycle = input_latent_list #+ input_latent_list[::-1]
    mask_coords_list_cycle = []
    mask_list_cycle = []
    for i, (frame, alpha) in enumerate(tqdm(zip(frame_list_cycle, alpha_list))):
        # 保存带透明通道的图片
        if alpha is not None:
            # 转换BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 创建RGBA图像
            rgba_frame = Image.fromarray(rgb_frame).convert('RGBA')
            # 设置透明通道
            rgba_frame.putalpha(alpha)
            rgba_frame.save(f"{save_full_path}/{str(i).zfill(8)}.png", 'PNG')
        else:
            cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)
            
        face_box = coord_list_cycle[i]
        mask, crop_box = get_image_prepare_material(frame, face_box)
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
        mask_coords_list_cycle += [crop_box]
        mask_list_cycle.append(mask)

    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)
    torch.save(input_latent_list_cycle, os.path.join(latents_out_path))
    if os.path.exists(f"{save_full_path}/{os.path.basename(file)}"):
        os.remove(f"{save_full_path}/{os.path.basename(file)}")

# initialize the mmpose model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(1, flip_input=False, device=device)
config_file = os.path.join(current_dir, 'utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
checkpoint_file = os.path.abspath(os.path.join(current_dir, '../models/dwpose/dw-ll_ucoco_384.pth'))
model = init_model(config_file, checkpoint_file, device=device)
vae = AutoencoderKL.from_pretrained(os.path.abspath(os.path.join(current_dir, '../models/sd-vae-ft-mse')))
vae.to(device)
fp = FaceParsing(os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/resnet18-5c106cde.pth')),
                 os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/79999_iter.pth')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",
                        type=str,
                        default=r'D:\ok\00000000.png',
                        )
    parser.add_argument("--avatar_id",
                        type=str,
                        default='3',
                        )
    args = parser.parse_args()
    create_musetalk_human(args.file, args.avatar_id)
