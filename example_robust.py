import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw, ImageFont
from CFA import CFA
# import animeface
import argparse
import time
import os
import json


def load_file_from_dir(dir_path):
    ret = []
    for file in os.listdir(dir_path):
        path_comb = os.path.join(dir_path, file)
        if os.path.isdir(path_comb):
            ret += load_file_from_dir(path_comb)
        else:
            ret.append(path_comb)
    return ret


def fmt_time(dtime):
    if dtime <= 0:
        return '0:00.000'
    elif dtime < 60:
        return '0:%02d.%03d' % (int(dtime), int(dtime * 1000) % 1000)
    elif dtime < 3600:
        return '%d:%02d.%03d' % (int(dtime / 60), int(dtime) % 60, int(dtime * 1000) % 1000)
    else:
        return '%d:%02d:%02d.%03d' % (int(dtime / 3600), int((dtime % 3600) / 60), int(dtime) % 60,
                                      int(dtime * 1000) % 1000)

if __name__ == "__main__":
    # param
    num_landmark = 24
    img_width = 128
    checkpoint_name = 'checkpoint_landmark_191116.pth.tar'
    parser = argparse.ArgumentParser(description='Anime face & landmark detector')
    parser.add_argument('-i', help='The input path of an image or directory', required=True, dest='input', type=str)
    parser.add_argument('-o', help='The output json path of the detection result', required=True, dest='output')
    parser.add_argument('-c', help='Whether crop right-half of the input image, y/n', dest='need_halfcrop')
    parser.add_argument('-d', help='Whether detect anime faces, y/n', dest='need_detect')
    parser.add_argument('-s', help='Whether show landmarks\' order, y/n', dest='show_order')

    args = parser.parse_args()
    assert os.path.exists(args.input), 'The input path does not exists'
    
    result = {}
    time_start = time.time()

    if os.path.isdir(args.input):
        files = load_file_from_dir(args.input)
    else:
        files = [args.input]
    file_len = len(files)

    src = args.input
    dst = args.output
    if args.need_halfcrop is 'y':
        need_halfcrop = True
    else:
        need_halfcrop = False
    if args.need_detect is 'y':
        need_detect = True
    else:
        need_detect = False
    if args.show_order is 'y':
        show_order = True
    else:
        show_order = False
    src_path_len = len(src)

    # detector
    face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name).cuda()

    # transform
    normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    train_transform = [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(train_transform)

    if show_order:
        font = ImageFont.truetype("/usr/share/fonts/truetype/Ubuntu-M.ttf", 15)

    for idx, file_name in enumerate(files):
        elapsed = time.time() - time_start
        eta = (file_len - idx) * elapsed / idx if idx > 0 else 0
        print('[%d/%d] Elapsed: %s, ETA: %s >> %s' % (idx+1, file_len, fmt_time(elapsed), fmt_time(eta), file_name))
        
        if '.jpg' not in file_name and '.png' not in file_name and '.jpeg' not in file_name:
             continue 
        result[file_name] = []
        # input image & detect face
        img = cv2.imread(file_name)
        half_size = int(img.shape[1] / 2)
        if need_halfcrop:
            img = img[:,half_size:,:]
        if show_order:
            img = cv2.resize(img,(1024,1024), interpolation=cv2.INTER_CUBIC)
        if need_detect:
            faces = face_detector.detectMultiScale(img)
        else:
            faces = [[ 0, 0, img.shape[0], img.shape[1]]]
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        for x_, y_, w_, h_ in faces:

            if need_detect:
                # adjust face size
                x = max(x_ - w_ / 8, 0)
                rx = min(x_ + w_ * 9 / 8, img.width)
                y = max(y_ - h_ / 4, 0)
                by = y_ + h_
                w = rx - x
                h = by - y
            else:
                x, y, w, h = x_, y_, w_, h_

            print([x, y, x+w, y+h])

            if need_detect:
                # draw result of face detection
                draw.rectangle((x, y, x + w, y + h), outline=(0, 0, 255), width=3)

            # transform image
            img_tmp = img.crop((x, y, x+w, y+h))
            img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)
            img_tmp = train_transform(img_tmp)
            img_tmp = img_tmp.unsqueeze(0).cuda()

            # estimate heatmap
            heatmaps = landmark_detector(img_tmp)
            heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

            # calculate landmark position
            landmarks = []
            for i in range(num_landmark):
                heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
                landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
                landmark_y = landmark[0] * h / img_width
                landmark_x = landmark[1] * w / img_width
                landmarks.append([float(landmark_x), float(landmark_y)])
                # print([landmark_x, landmark_y])

                # draw landmarks
                draw.ellipse((x + landmark_x - 2, y + landmark_y - 2, x + landmark_x + 2, y + landmark_y + 2), fill=(255, 0, 0))
                if show_order:
                    draw.text((x + landmark_x, y + landmark_y), str(i+1), fill=(0,0,255), font=font)
                        
            new_result = {'bbox': [float(x), float(y), float(x+w), float(y+h)],
                          'landmark': landmarks}
            result[file_name].append(new_result)
        
        if args.output:
            if ((idx+1) % 10000) == 0:
                # saving the temporary result
                with open(args.output+'output.json', 'w') as f:
                    # print(result)
                    print(str(idx+1)+" pictures have been processed!")
                    json.dump(result, f)
        if not os.path.exists(dst + '/' + file_name[src_path_len:file_name.rfind('/')]):
            os.makedirs(dst + '/' + file_name[src_path_len:file_name.rfind('/')])

        img.save(dst + '/' + file_name[src_path_len:])
        
    if args.output:
        with open(args.output+'output.json', 'w') as f:
            json.dump(result, f)

        # # output image
        # img.save('output.bmp')
