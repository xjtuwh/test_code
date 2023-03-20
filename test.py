            if frame_id == 0:
                H, W, _ = frame.shape                
                # identify object
                if len(init_rect) == 4 and init_rect[2] > 0 and init_rect[2] < W and init_rect[3] > 0 and init_rect[3] < H:
                    # initialization
                    tracker = get_keep_track(image, init_rect)
                    # classify by MDNet
                    target_bbox = init_rect
                    regions = np.zeros((1, 107, 107, 3), dtype='uint8')
                    regions[0] = crop_image2(image, target_bbox, 107, 16)
                    regions = regions.transpose(0, 3, 1, 2)
                    regions = regions.astype('float32') - 128.
                    regions = torch.from_numpy(regions)
                    regions = regions.cuda()

                    if target_bbox[2] <= 24:
                        k = 0
                    if target_bbox[2] > 24 and target_bbox[2] <= 48:
                        k = 1
                    if target_bbox[2] > 48:
                        k = 2

                    with torch.no_grad():
                        score = model(regions, k)

                    cls_s0 = score[:, 1]

                    start_det = 1
                    out_res.append(init_rect)
                    gt = label_res['gt_rect'][frame_id]
                    if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                        iou_s = 0
                    else:
                        iou_s = iou(init_rect, gt)
                    out_res_1.append([init_rect, 22, iou_s])

                    # RF_module = get_ar(image, init_rect, ar_path)
                    t_valid = 1
                else:
                    t = [0]
                    cls_s0 = 0
                    k = 0
                    start_det = 1
                    out_res.append(t)

                    out_res_1.append([t, 21, 1])

                    # RF_module = None
                    tracker = None
                    t_valid = 0

            else:
                # tracking
                if tracker != None:
                    out = tracker.track(image)
                    t = out['target_bbox']
                else:
                    t = [0]

                if start_det == 0:
                    if (len(t) == 1 and t[0] == 0):
                        start_det = 1
                        out_res.append([0])
                        gt = label_res['gt_rect'][frame_id]
                        if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                            iou_s = 1
                        else:
                            iou_s = 0                        
                        out_res_1.append([0, 20, iou_s])
                    else:
                        pred_bbox = RF_module.refine(image, np.array(t))
                        x1, y1, w, h = pred_bbox
                        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                        w = x2 - x1
                        h = y2 - y1
                        pred_bbox = [x1, y1, w, h]
                        out_res.append(pred_bbox)
                        gt = label_res['gt_rect'][frame_id]
                        if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                            iou_s = 0
                        else:
                            iou_s = iou(pred_bbox, gt)
                        out_res_1.append([pred_bbox, 19, iou_s])

                        # s_t = E(t)
                        # classify by MDNet
                        target_bbox = pred_bbox
                        regions = np.zeros((1, 107, 107, 3), dtype='uint8')
                        regions[0] = crop_image2(image, target_bbox, 107, 16)
                        regions = regions.transpose(0, 3, 1, 2)
                        regions = regions.astype('float32') - 128.
                        regions = torch.from_numpy(regions)
                        regions = regions.cuda()
                        if target_bbox[2] <= 24:
                            k_t = 0
                        if target_bbox[2] > 24 and target_bbox[2] <= 48:
                            k_t = 1
                        if target_bbox[2] > 48:
                            k_t = 2

                        with torch.no_grad():
                            score = model(regions, k_t)

                        s_t = score[:, 1]
                        if s_t <= 2 and s_t < cls_s0:
                            start_det = 1

                else:
                    # detect
                    det = predictor.predict_on_image(image)["instances"]  #conf > th(0.3)
                    det_bb = det.pred_boxes  # Boxes
                    det_s = det.scores       # Boxes
                    det_bb = det_bb.tensor.numpy()
                    if (len(t) == 0) or (len(t) == 1 and t[0] == 0) or (t_valid == 0):
                        if len(det_bb) == 0:
                            out_res.append([0])
                            gt = label_res['gt_rect'][frame_id]
                            if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                iou_s = 1
                            else:
                                iou_s = 0
                            out_res_1.append([0, 18, iou_s])
                        else:
                            # d = d_top1
                            d = det_bb[det_s.argmax()].tolist()
                            d[2] = d[2] - d[0] + 1
                            d[3] = d[3] - d[1] + 1
                            d = list(map(int, d))
                            # s_d = E(d)
                            # classify by MDNet
                            target_bbox = d
                            regions = np.zeros((1, 107, 107, 3), dtype='uint8')
                            regions[0] = crop_image2(image, target_bbox, 107, 16)
                            regions = regions.transpose(0, 3, 1, 2)
                            regions = regions.astype('float32') - 128.
                            regions = torch.from_numpy(regions)
                            regions = regions.cuda()
                            if target_bbox[2] <= 24:
                                k_d = 0
                            if target_bbox[2] > 24 and target_bbox[2] <= 48:
                                k_d = 1
                            if target_bbox[2] > 48:
                                k_d = 2

                            with torch.no_grad():
                                score = model(regions, k_d)

                            s_d = score[:, 1]

                            if s_d <= 2:
                                pred_bbox = d
                                out_res.append(pred_bbox)
                                gt = label_res['gt_rect'][frame_id]
                                if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                    iou_s = 0
                                else:
                                    iou_s = iou(pred_bbox, gt)
                                out_res_1.append(
                                    [pred_bbox, 17, iou_s])
                            else:
                                pred_bbox = d
                                out_res.append(pred_bbox)
                                gt = label_res['gt_rect'][frame_id]
                                if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                    iou_s = 0
                                else:
                                    iou_s = iou(pred_bbox, gt)
                                out_res_1.append(
                                    [pred_bbox, 14, iou_s])
                                if s_d >= cls_s0:
                                    cls_s0 = s_d
                                    if k_d == k:
                                        if k == 0:
                                            # re-init T
                                            tracker = get_keep_track(image, pred_bbox)
                                            # # re-init R
                                            # RF_module = get_ar(image, pred_bbox, ar_path)
                                            t_valid = 1
                                        else:
                                            # re-loc T
                                            state = pred_bbox
                                            tracker.pos = torch.Tensor(
                                                [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
                                            tracker.target_sz = torch.Tensor([state[3], state[2]])
                                            t_valid = 1

                                    else:
                                        k = k_d
                                        if k == 0:
                                            # re-loc T
                                            state = pred_bbox
                                            tracker.pos = torch.Tensor(
                                                [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
                                            tracker.target_sz = torch.Tensor([state[3], state[2]])
                                            t_valid = 1
                                        else:
                                            if tracker != None:
                                                # re-loc T
                                                state = pred_bbox
                                                tracker.pos = torch.Tensor(
                                                    [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
                                                tracker.target_sz = torch.Tensor([state[3], state[2]])
                                                t_valid = 1

                                            else:
                                                # re-init T
                                                tracker = get_keep_track(image, pred_bbox)
                                                # # re-init R
                                                # RF_module = get_ar(image, pred_bbox, ar_path)
                                                t_valid = 1
                    # t != []
                    else:
                        if len(det_bb) == 0:
                            # s_t = E(t)
                            # classify by MDNet
                            target_bbox = t
                            regions = np.zeros((1, 107, 107, 3), dtype='uint8')
                            regions[0] = crop_image2(image, target_bbox, 107, 16)
                            regions = regions.transpose(0, 3, 1, 2)
                            regions = regions.astype('float32') - 128.
                            regions = torch.from_numpy(regions)
                            regions = regions.cuda()
                            if target_bbox[2] <= 24:
                                k_t = 0
                            if target_bbox[2] > 24 and target_bbox[2] <= 48:
                                k_t = 1
                            if target_bbox[2] > 48:
                                k_t = 2

                            with torch.no_grad():
                                score = model(regions, k_t)

                            s_t = score[:, 1]

                            if s_t <= 2 and s_t < cls_s0:
                                out_res.append([0])
                                gt = label_res['gt_rect'][frame_id]
                                if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                    iou_s = 1
                                else:
                                    iou_s = 0
                                out_res_1.append(
                                    [0, 13, iou_s])
                                t_valid = 0
                            else:
                                if s_t > 2 and s_t >= cls_s0:
                                    # pred_bbox = RF_module.refine(image, np.array(t))
                                    pred_bbox = t
                                    x1, y1, w, h = pred_bbox
                                    x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                                    w = x2 - x1
                                    h = y2 - y1
                                    pred_bbox = [x1, y1, w, h]
                                    out_res.append(pred_bbox)
                                    gt = label_res['gt_rect'][frame_id]
                                    if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                        iou_s = 0
                                    else:
                                        iou_s = iou(pred_bbox, gt)
                                    out_res_1.append(
                                        [pred_bbox, 12, iou_s])
                                else:
                                    out_res.append([0])
                                    gt = label_res['gt_rect'][frame_id]
                                    if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                        iou_s = 1
                                    else:
                                        iou_s = 0
                                    out_res_1.append(
                                        [0, 11, iou_s])
                        else:
                            # IoU(t, d)
                            iou_scores = []
                            for bb in det_bb:
                                det = bb.tolist()
                                det[2] = det[2] - det[0] + 1
                                det[3] = det[3] - det[1] + 1
                                det = list(map(int, det))
                                iou_scores.append(iou(t, det))
                            max_iou = max(iou_scores)
                            if max_iou > 0.5:
                                id = iou_scores.index(max_iou)
                                d = det_bb[id].tolist()
                                d[2] = d[2] - d[0] + 1
                                d[3] = d[3] - d[1] + 1
                                d = list(map(int, d))
                                x1 = (t[0] + d[0]) / 2
                                y1 = (t[1] + d[1]) / 2
                                w = (t[2] + d[2]) / 2
                                h = (t[3] + d[3]) / 2
                                # pred_bbox = RF_module.refine(image, np.array([x1, y1, w, h]))
                                x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                                w = x2 - x1
                                h = y2 - y1
                                pred_bbox = [x1, y1, w, h]
                                out_res.append(pred_bbox)
                                gt = label_res['gt_rect'][frame_id]
                                if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (
                                        len(gt) == 4 and gt == [1, 1, 640, 512]) or (
                                        len(gt) == 4 and gt == [0, 0, 0, 0]):
                                    iou_s = 0
                                else:
                                    iou_s = iou(pred_bbox, gt)
                                out_res_1.append(
                                    [pred_bbox, 10, iou_s])
                            if max_iou > 0.3 and max_iou <= 0.5:
                                id = iou_scores.index(max_iou)
                                d = det_bb[id].tolist()
                                d[2] = d[2] - d[0] + 1
                                d[3] = d[3] - d[1] + 1
                                d = list(map(int, d))
                                x1 = (t[0] + d[0]) / 2
                                y1 = (t[1] + d[1]) / 2
                                w = (t[2] + d[2]) / 2
                                h = (t[3] + d[3]) / 2
                                # pred_bbox = RF_module.refine(image, np.array([x1, y1, w, h]))
                                x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                                w = x2 - x1
                                h = y2 - y1
                                pred_bbox = [x1, y1, w, h]
                                out_res.append(pred_bbox)
                                gt = label_res['gt_rect'][frame_id]
                                if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (
                                        len(gt) == 4 and gt == [1, 1, 640, 512]) or (
                                        len(gt) == 4 and gt == [0, 0, 0, 0]):
                                    iou_s = 0
                                else:
                                    iou_s = iou(pred_bbox, gt)
                                out_res_1.append(
                                    [pred_bbox, 9, iou_s])
                            if max_iou <= 0.3:
                                # d = d_top1
                                d = det_bb[det_s.argmax()].tolist()
                                d[2] = d[2] - d[0] + 1
                                d[3] = d[3] - d[1] + 1
                                d = list(map(int, d))
                                # s_d = E(d)
                                # classify by MDNet
                                target_bbox = d
                                regions = np.zeros((1, 107, 107, 3), dtype='uint8')
                                regions[0] = crop_image2(image, target_bbox, 107, 16)
                                regions = regions.transpose(0, 3, 1, 2)
                                regions = regions.astype('float32') - 128.
                                regions = torch.from_numpy(regions)
                                regions = regions.cuda()
                                if target_bbox[2] <= 24:
                                    k_d = 0
                                if target_bbox[2] > 24 and target_bbox[2] <= 48:
                                    k_d = 1
                                if target_bbox[2] > 48:
                                    k_d = 2

                                with torch.no_grad():
                                    score = model(regions, k_d)

                                s_d = score[:, 1]

                                # s_t = E(t)
                                # classify by MDNet
                                target_bbox = t
                                regions = np.zeros((1, 107, 107, 3), dtype='uint8')
                                regions[0] = crop_image2(image, target_bbox, 107, 16)
                                regions = regions.transpose(0, 3, 1, 2)
                                regions = regions.astype('float32') - 128.
                                regions = torch.from_numpy(regions)
                                regions = regions.cuda()
                                if target_bbox[2] <= 24:
                                    k_t = 0
                                if target_bbox[2] > 24 and target_bbox[2] <= 48:
                                    k_t = 1
                                if target_bbox[2] > 48:
                                    k_t = 2

                                with torch.no_grad():
                                    score = model(regions, k_t)

                                s_t = score[:, 1]

                                if s_d <= s_t:
                                    if s_t <= 2 and s_t < cls_s0:
                                        out_res.append(d)
                                        gt = label_res['gt_rect'][frame_id]
                                        if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                            iou_s = 0
                                        else:
                                            iou_s = iou(d, gt)
                                        out_res_1.append(
                                            [0, 8, iou_s])
                                        t_valid = 0
                                    else:
                                        # pred_bbox = RF_module.refine(image, np.array(t))
                                        pred_bbox = t
                                        x1, y1, w, h = pred_bbox
                                        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                                        w = x2 - x1
                                        h = y2 - y1
                                        pred_bbox = [x1, y1, w, h]
                                        out_res.append(pred_bbox)
                                        gt = label_res['gt_rect'][frame_id]
                                        if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                            iou_s = 0
                                        else:
                                            iou_s = iou(pred_bbox, gt)
                                        out_res_1.append(
                                            [pred_bbox, 7, iou_s])
                                else:
                                    if s_d >= cls_s0:
                                        if s_d > 2:
                                            pred_bbox = d
                                            out_res.append(pred_bbox)
                                            gt = label_res['gt_rect'][frame_id]
                                            if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                                iou_s = 0
                                            else:
                                                iou_s = iou(pred_bbox, gt)
                                            out_res_1.append(
                                                [pred_bbox, 6, iou_s])
                                            cls_s0 = s_d
                                            # re-loc T
                                            state = pred_bbox
                                            tracker.pos = torch.Tensor(
                                                [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
                                            tracker.target_sz = torch.Tensor([state[3], state[2]])
                                            if k_d != k:
                                                k = k_d
                                        else:
                                            if k_d == k:
                                                out_res.append(d)
                                                gt = label_res['gt_rect'][frame_id]
                                                if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                                    iou_s = 0
                                                else:
                                                    iou_s = iou(d, gt)
                                                out_res_1.append(
                                                    [d, 5, iou_s])
                                            else:
                                                if s_t < cls_s0:
                                                    out_res.append(d)
                                                    gt = label_res['gt_rect'][frame_id]
                                                    if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                                        iou_s = 0
                                                    else:
                                                        iou_s = iou(d, gt)
                                                    out_res_1.append(
                                                        [d, 4, iou_s])

                                                    t_valid = 0
                                                else:
                                                    out_res.append(d)
                                                    gt = label_res['gt_rect'][frame_id]
                                                    if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                                        iou_s = 0
                                                    else:
                                                        iou_s = iou(d, gt)
                                                    out_res_1.append(
                                                        [d, 3, iou_s])
                                    else:
                                        if s_d > 2:
                                            out_res.append(d)
                                            gt = label_res['gt_rect'][frame_id]
                                            if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                                iou_s = 0
                                            else:
                                                iou_s = iou(d, gt)
                                            out_res_1.append(
                                                [d, 2, iou_s])
                                        else:
                                            out_res.append(d)
                                            gt = label_res['gt_rect'][frame_id]
                                            if (len(gt) == 1 and gt[0] == 0) or len(gt) == 0 or (len(gt) == 4 and gt == [1, 1, 640, 512]) or (len(gt) == 4 and gt == [0, 0, 0, 0]):
                                                iou_s = 0
                                            else:
                                                iou_s = iou(d, gt)
                                            out_res_1.append(
                                                [0, 1, iou_s])
                                            t_valid = 0
