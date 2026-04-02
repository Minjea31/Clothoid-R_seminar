from ultralytics import YOLO
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='baseline')
parser.add_argument('--bs', type=int, default=8)
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--model_pt', type=str, default='yolov12n.pt')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--data', type=str, default="../dataset.yaml")
parser.add_argument("--device", type=str, default='0')

args = parser.parse_args()

model = YOLO(args.model_pt)
if not args.resume:
    model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name,
                batch=args.bs, workers=8, save_period=5, project='checkpoints')
else:
    model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name,
                batch=args.bs, workers=8, save_period=5, project='checkpoints', resume=args.resume)
