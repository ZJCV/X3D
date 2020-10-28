from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch

from tsn.model.build import build_model
from tsn.data.transforms.build import build_transform
from tsn.util.parser import parse_test_args, load_test_config
from tsn.util.distributed import get_device, get_local_rank

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    while True:
        msg = 'Waiting for action ...'
        ret, frame = camera.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = test_transform(rgb)
        frame_queue.append(rgb)

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info):
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


def inference():
    score_cache = deque()
    scores_sum = 0
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(frame_queue)[frame_interval // 2::frame_interval]

        images = torch.stack(cur_windows).transpose(0, 1).unsqueeze(0)
        images = images.to(device=device, non_blocking=True)

        with torch.no_grad():
            output_dict = model(images)
            probs = torch.softmax(output_dict['probs'], dim=1).cpu().numpy()[0]

        score_cache.append(probs)
        scores_sum += probs

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()
    camera.release()
    cv2.destroyAllWindows()


def main():
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_transform, model, device, average_size, label, result_queue, \
        frame_interval

    args = parse_test_args()
    cfg = load_test_config(args)
    average_size = 1
    threshold = 0.5

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = get_device(local_rank=get_local_rank())
    model = build_model(cfg, device)
    model.eval()
    camera = cv2.VideoCapture(cfg.VISUALIZATION.INPUT_VIDEO)

    with open(cfg.VISUALIZATION.LABEL_FILE_PATH, 'r') as f:
        label = [line.strip().split(' ')[1] for line in f]

    # prepare test pipeline from non-camera pipeline
    test_transform = build_transform(cfg, is_train=False)
    sample_length = cfg.DATASETS.CLIP_LEN * cfg.DATASETS.NUM_CLIPS * cfg.DATASETS.FRAME_INTERVAL
    frame_interval = cfg.DATASETS.FRAME_INTERVAL

    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        pw.start()
        pr.start()
        while True:
            if not pw.is_alive():
                exit(0)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
