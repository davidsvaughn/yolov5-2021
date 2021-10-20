import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, box_io2, box_ios, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
import ast

def save_list(lst, fn):
    with open(fn, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)

@torch.no_grad()
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=-1, # not for NMS
         iou_thres=0.25,  # not for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         max_by_class=True,
         opt=None):
    
    print_size, print_batches = 0, 3
    log_errors = -1
    if opt is not None:
        print_size = opt.print_size
        print_batches = opt.print_batches
        max_by_class = opt.max_by_class
        log_errors = opt.log_errors
        ct = ast.literal_eval(opt.ct)
        if len(ct)==0: ct=None
    else:
        ct = None
    if print_batches<0:
        print_batches = 1000

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        # device = select_device(opt.device, batch_size=batch_size)
        device = select_device(opt.device, batch_size=2)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = torch.arange(iou_thres, 1, 0.05).to(device) # iou_thres : 0.95 : 0.05
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = data['names']#names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    names_dict = {k: v for k, v in enumerate(names)}
    coco91class = coco80_to_coco91_class()
    # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    fmt = '%{}s'.format(2+max([len(s) for s in names]))
    s = (fmt + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'F1', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, mf1 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    error_log = []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        inf_out, train_out = model(img, augment=augment)  # inference and training outputs
        t0 += time_synchronized() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        output = non_max_suppression(inf_out, labels=lb, multi_label=False, agnostic=True)#, conf_thres=conf_thres, iou_thres=iou_thres)
        t1 += time_synchronized() - t

        # Statistics per image
        idx = []
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            # Dims of all target boxes (in pixels)
            target_dims = targets[targets[:, 0] == si, -2:]

            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                idx.append(None)
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Filter out-of-frame predictions (in padding)
            gain, img_box = shapes[si][1][0][0], None
            if gain==1:
                img_box = torch.zeros([1,4])
                img_box[0,:2] = torch.FloatTensor(shapes[si][1][1])
                img_box[0,2:] = torch.FloatTensor(shapes[si][0]) + torch.FloatTensor(shapes[si][1][1])
                img_box = img_box.to(device)
                io2s = box_io2(img_box, pred[:, :4])
                k = (io2s > 0.95).nonzero(as_tuple=True)[1]
                pred = pred[k,:]
                idx.append(k)
            else:
                idx.append(None)

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Dims of all pred boxes (in pixels)
            pred_target_dims = torch.zeros(pred.shape[0], 4, dtype=torch.float32, device=device)
            pred_target_dims[:,:2] = pred[:,2:4]-pred[:,:2]

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                "class_id": int(cls),
                                "box_caption": "%s %.3f" % (names[int(cls)], conf),
                                "scores": {"class_score": conf},
                                "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names_dict}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target... index into target array....
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                k = pi[j] ## index into pred array.....
                                pred_target_dims[k,2:] = target_dims[d]
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, pred_target_dims.cpu(), target_dims.cpu()))

            # FP/FN counts (per image)
            if log_errors>-1:
                corr = correct.cpu().numpy()
                idx_conf = pred[:, 4].cpu()>conf_thres
                if idx_conf.cpu().numpy().mean()<1:
                    corr = corr[idx_conf]
                tp = corr[:,0].sum()
                fp, fn = len(corr)-tp, len(tcls)-tp
                if fp+fn > log_errors:
                    error_log.append(path.stem)


        # Plot images
        if plots and batch_i < print_batches:
            prefix = Path(paths[0]).stem if batch_size==1 else f'test_batch{batch_i}'
            f1 = save_dir / f'{prefix}_labels.jpg'  # labels
            thread1 = Thread(target=plot_images, args=(img, targets, paths, f1, names, print_size), daemon=True)
            f2 = save_dir / f'{prefix}_pred.jpg'  # predictions
            thread2 = Thread(target=plot_images, args=(img, output_to_target(output, idx), paths, f2, names, print_size), daemon=True)
            thread1.start()
            thread1.join()
            thread2.start()
            thread2.join()
            ##################################
            # display fn and fp boxes only
            if log_errors>-1 and batch_size==1:
                ## fn boxes...
                if fn>0:
                    idx_fn = np.ones(len(targets))
                    for d in detected:
                        idx_fn[d.item()]=0
                    idx_fn = np.nonzero(idx_fn)[0]
                    f = save_dir / f'{prefix}_fn.jpg'  # labels
                    thread = Thread(target=plot_images, args=(img, targets[idx_fn], paths, f, names, print_size, 16, True), daemon=True)
                    thread.start()
                    thread.join()
                ## fp boxes...
                if fp>0:
                    idx_fp = ~corr[:,0]
                    f = save_dir / f'{prefix}_fp.jpg'  # labels
                    thread = Thread(target=plot_images, args=(img, output_to_target(output, idx, idx_fp, idx_conf), paths, f, names, print_size, 16, True), daemon=True)
                    thread.start()
                    thread.join()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    conf_best = -1
    if len(stats) and stats[0].any():
        mp, mr, map50, map, mf1, ap_class, conf_best, nt, (p, r, ap50, ap, f1, cc) = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names, 
                                                                                                ct=ct, max_by_class=max_by_class, conf_thres=conf_thres)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = fmt + '%12.3g' * 7  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, mf1, map50, map))

    # Print results per class
    if (verbose or nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], f1[i], ap50[i], ap[i]))
    if conf_best>-1:
        print('\nOptimal Confidence Threshold: {0:0.3f}'.format(conf_best))
        if max_by_class:
            print('Optimal Confidence Thresholds (Per-Class): {}'.format(list(cc.round(3))))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=names)
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        # if len(error_log)>0:
        #     fn = f'{save_dir}/error_log.txt'
        #     save_list(error_log, fn)

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, mf1, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=-1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--print-batches', type=int, default=3, help='how many batches to print')
    parser.add_argument('--print-size', type=int, default=640, help='print size (pixels)')
    parser.add_argument('--max-by-class', action='store_true', help='find class specific confidence thresholds')
    parser.add_argument('--ct', type=str, default='[]', help='class confidence thresholds')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--log-errors', type=int, default=-1, help='save image names with FP+FN>log_errors')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             opt=opt
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, opt.conf_thres, opt.iou_thres, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
