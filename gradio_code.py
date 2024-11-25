from __future__ import print_function, division

import gradio as gr
import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random


class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.phase = phase

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        # self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()
        print("===========Render Ready============")

    def infer_seqence(self, num,h,s,v, ratio = 10):
        print(ratio)
        print("==========Infer Seqence Run==========")
        self.dataset = StereoHumanDataset(self.cfg.dataset,h,s,v, phase=self.phase)
        img_path = os.path.join(self.cfg.dataset.test_data_root,'img/%s/%d.jpg')
        folder = str(num+1).zfill(4)
        input_img1 = img_path %(folder,0)
        input_img2 = img_path %(folder,1)
        img_list = []
        view_select=[0,1]
        ratio_arr = [10,20,50,100]
        duration_arr = [2,1,0.4,0.2]
        duration =  duration_arr[int(ratio)]
        ratio = ratio_arr[int(ratio)]

        total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        for idx in tqdm(range(total_frames)):
            if idx != int(num):
                continue
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)

            for i in range(ratio+1):
                i = round(i/ratio,2)
                # print("----ratio flag, i value = ",i)
                data = get_novel_calib(data, self.cfg.dataset, ratio=i, intr_key='intr_ori', extr_key='extr_ori')
                with torch.no_grad():
                    data, _, _ = self.model(data, is_train=False)
                    data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = self.tensor2np(data['novel_view']['img_pred'])
                img_list.append(render_novel)
                cv2.imwrite(self.cfg.test_out_path + '/%s_novel_%.1f.jpg' % (data['name'],i), render_novel)
        print("gen img done")
        output_gif = self.cfg.test_out_path + '/output_gif.gif'

        #BGR 2 RGB 변환
        # print("first img len",len(images))
        images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if len(img.shape) == 3 else Image.fromarray(img) for img in img_list]

        for img in img_list:
            # images.append(Image.fromarray((img * 255).astype(np.uint8)))
            img = (Image.fromarray((img * 255).astype(np.uint8)))
        print("second img len",len(images))
        # GIF 생성
        if images:
            images[0].save(
                output_gif,
                save_all=True,
                append_images=images[1:],  # 첫 이미지는 제외한 나머지를 추가
                duration=int(duration * 100),  # Pillow는 ms 단위 사용
                loop=0  # 무한 반복
            )
            print(f"GIF 생성 완료: {output_gif}")
        else:
            print("이미지를 찾을 수 없습니다.")
        return input_img1, input_img2, output_gif


    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):     #img 색상 변경
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda().unsqueeze(0)
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == "__main__":
    random.seed(42)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--test_data_root', type=str, required=True)
    # parser.add_argument('--ckpt_path', type=str, required=True)
    # parser.add_argument('--src_view', type=int, nargs='+', required=True)
    # parser.add_argument('--ratio', type=float, default=0.5)
    # arg = parser.parse_args()
    

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = "../gps_dataset/processed_data"
    cfg.dataset.use_processed_data = False
    # cfg.restore_ckpt = "../gps_dataset/GPS-GS_stage2_final.pth"
    cfg.restore_ckpt = "/home/sophie/Desktop/minseok/GPS-Gaussian/experiments/Third_Train_0928/ckpt/Third_Train_final.pth"
    cfg.test_out_path = './test_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRender(cfg, phase='test')
    # render.infer_seqence(num,h,s,v)
    # def infer_seqence(self, num,h,s,v view_select=[0,1], ratio=10):
    #변수 추가 - 샘플 번호, ratio 개수, HSV 값)

    # Gradio 인터페이스 생성
    with gr.Blocks() as demo:
        gr.Markdown("### 슬라이드와 버튼 입력으로 이미지를 처리하고 결과를 출력합니다.")
        

        # 입력 슬라이드
        slide1 = gr.Slider(minimum=0, maximum=180, step=1, label="색상: Blue = 0 / Green = 60 / Red = 120", value = 0)
        slide2 = gr.Slider(minimum=-255, maximum=255, step=1, label="채도, 선택한 값만큼 기존 값에 추가됨", value = 0)
        slide3 = gr.Slider(minimum=-255, maximum=255, step=1, label="명도, 선택한 값만큼 기존 값에 추가됨", value = 0)

        with gr.Row():
            with gr.Column():
                num = gr.Radio(["Image_1","Image_2","Image_3","Image_4","Image_5","Image_6","Image_7","Image_8"], label="Select One Image", type="index", value = "Image_1")
                ratio = gr.Radio(["10 Images","20 Images","50 Images","100 Images"], label="Select Ratio of Image", type="index", value = "10 Images")
            # 출력 요소`
            image1_output = gr.Image(label="Input Image 1")
            image2_output = gr.Image(label="Input Image 2")

        # 버튼
        submit_btn = gr.Button("Submit")
        video_output = gr.Image(label="Output Video")

        # 버튼 이벤트 연결
        submit_btn.click(
            fn=render.infer_seqence,  # 호출할 함수
            inputs=[num,slide1, slide2, slide3, ratio],  # 함수의 입력값
            outputs=[image1_output, image2_output, video_output]  # 출력 요소
        )

        # Gradio 데모 실행
    demo.launch(allowed_paths=["/home/sophie/Desktop/minseok/gps_dataset/processed_data/img","/home/sophie/Desktop/minseok/GPS-Gaussian/test_out"])