#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import cv2
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO detection with real-time display")
    parser.add_argument("--model-pt", type=str, default="../best_model/best_model.pt",
                        help="YOLO model weights (.pt)")
    parser.add_argument("--model-yaml", type=str, default="../best_model/best_model.yaml",
                        help="YOLO model config (.yaml)")
    parser.add_argument("--data-yaml", type=str, default="../dataset.yaml",
                        help="dataset.yaml path")
    parser.add_argument("--delay", type=int, default=2000,
                        help="Delay between frames in ms (default: 2000=2s)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model_yaml).load(args.model_pt)

    with open(args.data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    test_source = data_cfg.get("test")
    if not test_source:
        raise ValueError("dataset.yaml에 'test:' 항목이 없습니다.")

    class_names = data_cfg.get("names", ["fire", "other", "smoke"])
    name_map = {i: n for i, n in enumerate(class_names)}
    try:
        model.model.names = name_map
    except Exception:
        pass

    results = model.predict(
        source=test_source,
        imgsz=640,
        conf=0.25,
        stream=True,
        verbose=False,
    )

    cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    paused = False

    for r in results:
        r.names = name_map
        img = r.plot()

        fn = os.path.basename(getattr(r, "path", ""))
        if fn:
            cv2.putText(img, fn, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("detect", img)

        if paused:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows(); raise SystemExit
                elif key == ord(' '):
                    paused = False
                    break
                elif key == ord('n'):
                    break
        else:
            key = cv2.waitKey(args.delay) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows(); raise SystemExit
            elif key == ord(' '):
                paused = True

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

