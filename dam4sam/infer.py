import os
import site
site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/dam4sam')

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from dam4sam_tracker import DAM4SAMTracker
from utils.visualization_utils import overlay_mask, overlay_rectangle
from utils.box_selector import BoxSelector

import modelargs
config = modelargs.parse('./model.json')

'''
filename .. path to input video file
init_box .. x,y,width,height of initialisation box in relative coords (0..1)
'''
def process_video(filename, init_box):
    filename = Path(filename)
    cap = cv2.VideoCapture(filename)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x, y, box_width, box_height = init_box
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outpath = filename.with_suffix('.output.mp4')
    out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

    tracker = DAM4SAMTracker(config['tracker_name'])
    did_init = False

    i = 0
    while cap.isOpened():
        print(i, end='\r')
        i += 1

        ret, frame = cap.read()
        if not ret:
            break
        frame = Image.fromarray(frame)
        if did_init:
            outputs = tracker.track(frame)
        else:
            outputs = tracker.initialize(frame, None, bbox=[x * width, y * height, (x + box_width) * width, (y + box_height) * height])
            did_init = True
        frame = (outputs['pred_mask'] * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))

    cap.release()
    out.release()
    return outpath

if __name__ == "__main__":
    import sys
    process_video(sys.argv[1], list(map(float, sys.argv[2].split(','))))
else:
    from flask import Flask, request, send_file
    from io import BytesIO
    from uuid import uuid4
    import json
    import base64
    import subprocess
    import ffmpeg

    tmproot = Path("/tmp/dam4sam")
    tmproot.mkdir(exist_ok=True)

    # todo: delete inactive instances
    instances = {}
    app = Flask(__name__)

    def get_video_info(filename):
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        fps = int(video_info['nb_frames']) / float(video_info['duration'])
        return width, height, fps

    def encode(img):
        pil_img = Image.fromarray(img) if type(img) == np.ndarray else img
        buff = BytesIO()
        pil_img.save(buff, format='WebP')
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    @app.route('/infer', methods=['POST'])
    def index():
        data = request.json if request.is_json else request.form.to_dict()
        if not (uuid := data.get('id')):

            if 'image' not in request.files:
                return { 'error': 'Missing image' }, 400

            if not (box := data.get('box')):
                return 'Initialisation request must contain a box', 400
            x, y, box_width, box_height = json.loads(box)[0]

            uuid = str(uuid4())
            file = request.files['image']
            file_type = file.headers.get('Content-Type')

            if file_type == 'image/webp':
                tracker = DAM4SAMTracker(config['tracker_name'])

                image = Image.open(request.files['image'])
                image_width, image_height = image.size

                instances[uuid] = (tracker, None)

                outputs = tracker.initialize(image, None, bbox=[x * image_width, y * image_height, (x + box_width) * image_width, (y + box_height) * image_height])
                return { 'mask': encode((outputs['pred_mask'] * 255).astype(np.uint8)), 'id': uuid }
            elif file_type == 'video/mp4':
                filename = str(tmproot / uuid) + '.mp4'
                file.save(filename)
                image_width, image_height, fps = get_video_info(filename)
                reader = (ffmpeg
                    .input(filename, ss=float(data.get("image-timestamp", 0)))
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .global_args('-nostats') # prevent stderr from clogging up, if there are still issues with hang-ups we might need to pipe, peek and read stderr manually
                    .run_async(pipe_stdout=True, quiet=True))

                encoder = (ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(image_width, image_height), r=str(fps))
                    .output(filename.replace('.mp4', '.output.mp4'), pix_fmt='yuv420p')
                    .overwrite_output()
                    .global_args('-nostats') # same as above
                    .run_async(pipe_stdin=True, quiet=True))

                tracker = DAM4SAMTracker(config['tracker_name'])
                instances[uuid] = tracker, (reader, encoder, (image_width, image_height))
                image = Image.frombytes('RGB', (image_width, image_height), reader.stdout.read(image_width * image_height * 3))
                outputs = tracker.initialize(image, None, bbox=[x * image_width, y * image_height, (x + box_width) * image_width, (y + box_height) * image_height])
                frame = (outputs['pred_mask'] * 255).astype(np.uint8)
                encoder.stdin.write(frame.tobytes())
                return { 'mask': encode(frame), 'id': uuid, 'reference': encode(image)}
            else:
                return { 'error': f'Invalid file type "{file_type}"' }
        elif instance := instances.get(uuid):
            tracker, pipeline = instance
            if pipeline:
                reader, encoder, size = pipeline
                in_bytes = reader.stdout.read(size[0] * size[1] * 3)
                if 'finish' in data or not in_bytes:
                    encoder.stdin.close()
                    reader.wait()
                    encoder.wait()
                    return send_file(f'/tmp/dam4sam/{uuid}.output.mp4')
                else:
                    image = Image.frombytes('RGB', size, in_bytes)
                    outputs = tracker.track(image)
                    frame = (outputs['pred_mask'] * 255).astype(np.uint8)
                    encoder.stdin.write(frame.tobytes())
                    return { 'mask': encode(frame), 'id': uuid, 'reference': encode(image)}
            elif 'image' not in request.files:
                return { 'error': 'Missing image' }, 400
            else:
                outputs = tracker.track(Image.open(request.files['image']))
                return { 'mask': encode((outputs['pred_mask'] * 255).astype(np.uint8)), 'id': uuid }
        else:
            return { 'error': 'Invalid id' }, 400

        raise "Unhandled branch"
