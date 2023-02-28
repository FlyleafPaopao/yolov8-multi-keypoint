# yolov8-multi-keypoint
This repo would give multi-task keypoint detect code based yolov8. The  landmarks or keypoints with different classes and numbers can be detected at the same time.
⭐ This code has been completely released and the documentation will be improved gradually.⭐ 
This repo is almost the raw [yolov8] (https://github.com/ultralytics/ultralytics). And the following repo or article are referred:
[yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)
[yolo-pose-artical](https://arxiv.org/abs/2204.06806)
[yolov7-pose-estimation](https://github.com/RizwanMunawar/yolov7-pose-estimation)

Based aboving work, we made a little upgrade:
⭐ The pose-keypoints or landmarks detection based yolov8 was achieved, which is anchor free and owns decoupled head.
⭐ The muticlass with different points number detection was achieved, and these parameters can be set conveniently by .yaml.
⭐ ⭐ The most important, we introduced a new val.py for point accuracy measurement, such as:
Class  Images Instances Box(P R  mAP50 mAP50-95 AVG_distance MAX_distance  interval(0,1]  interval(1,2]  interval(2,3]  interval(3,4]  interval(4,5]  interval(5,6]  interval(6,7]  interval(7,8]  interval(8,9] interval(9,10]interval(10,+∞]          kpt_p          kpt_r        kpt_p_x        kpt_r_x        kpt_p_y       kpt_r_y): 100%|██████████| 51/51 [01:54<00:00,  2.24s/it]
all        407      74821      0.929       0.84      0.915      0.591           1.18            223          0.311          0.312          0.157         0.0781         0.0442         0.0274         0.0181         0.0123        0.00837        0.00625         0.0253          0.956          0.955          0.994          0.992          0.969          0.967

The sample model config file is v8/yolov8s.yaml. And the point number config file is datasets/test.yaml.