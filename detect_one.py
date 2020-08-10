
caffe_root='/home/workspace/install/caffe_install/caffe-master/'
import sys
import os
sys.path.insert(0,caffe_root+'python')
import rtsp
import caffe
import cv2
from utils import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser('YOLOv3')

    parser.add_argument('--prototxt', type=str, default='/home/workspace/nnie/package/yolov3/darknet2caffe/caffe_model/yolov3_6_1.prototxt')
    parser.add_argument('--caffemodel', type=str, default='/home/workspace/nnie/package/yolov3/darknet2caffe/caffe_model/yolov3_6_1.caffemodel')

    # parser.add_argument('--prototxt', type=str, default='/mnt/workspcae/caffe/test_model/xuhao-caffe-80/yolov3_80.prototxt')
    # parser.add_argument('--caffemodel', type=str, default='/mnt/workspcae/caffe/test_model/xuhao-caffe-80/yolov3_80.caffemodel')

    # parser.add_argument('--classfile', type=str, default='/mnt/workspcae/darknet/train/flame_detect/cfg/train.data')
    parser.add_argument('--classfile', type=str, default='/mnt/workspcae/caffe/test_model/2020-8-3/cfg/voc.names')

    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--resolution', type=int, default=416)

    return parser.parse_args()

def USBCamera():
    args = parse_args()
    global counter;
    model = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    inp_dim = args.resolution, args.resolution
    # capture = cv2.VideoCapture("/mnt/furg-fire-dataset/car2.mp4")
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        # 以下两步设置显示屏的宽高
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        counter=0
        # 持续读取摄像头数据
        while True:
            read_code, img_ori = capture.read()
            if not read_code:
                break



            img = img_prepare(img_ori, inp_dim)

            # cv2.imshow("?", img.transpose([1,2,0]))
            # cv2.waitKey()
            model.blobs['data'].data[:] = img
            output = model.forward()

            rects = rects_prepare(output)
            mapping = get_classname_mapping(args.classfile)

            scaling_factor = min(1, args.resolution / img_ori.shape[1])
            for pt1, pt2, cls, prob in rects:
                print(
                    "**********************************************************************************************************")
                pt1[0] -= (args.resolution - scaling_factor * img_ori.shape[1]) / 2
                pt2[0] -= (args.resolution - scaling_factor * img_ori.shape[1]) / 2
                pt1[1] -= (args.resolution - scaling_factor * img_ori.shape[0]) / 2
                pt2[1] -= (args.resolution - scaling_factor * img_ori.shape[0]) / 2

                pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
                pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
                pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
                pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])

                label = "{}:{:.2f}".format(mapping[cls], prob)
                color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

                cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, 1)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
                cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, -1)
                cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                            cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)

            if len(rects)!=0 and int(cls)<4:
                counter+=1

                # cv2.imwrite("/mnt/workspcae/fire_%d.jpg"%counter,img_ori)

            cv2.imshow('video', img_ori)

            if cv2.waitKey(30) == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()



def USBCamera_writetovideo():
    args = parse_args()
    global counter;
    model = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    inp_dim = args.resolution, args.resolution
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        # 以下两步设置显示屏的宽高
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        counter=0
        # 持续读取摄像头数据
        while True:
            read_code, img_ori = capture.read()
            if not read_code:
                break



            img = img_prepare(img_ori, inp_dim)

            # cv2.imshow("?", img.transpose([1,2,0]))
            # cv2.waitKey()
            model.blobs['data'].data[:] = img
            output = model.forward()

            rects = rects_prepare(output)
            mapping = get_classname_mapping(args.classfile)

            scaling_factor = min(1, args.resolution / img_ori.shape[1])
            for pt1, pt2, cls, prob in rects:
                print(
                    "**********************************************************************************************************")
                pt1[0] -= (args.resolution - scaling_factor * img_ori.shape[1]) / 2
                pt2[0] -= (args.resolution - scaling_factor * img_ori.shape[1]) / 2
                pt1[1] -= (args.resolution - scaling_factor * img_ori.shape[0]) / 2
                pt2[1] -= (args.resolution - scaling_factor * img_ori.shape[0]) / 2

                pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
                pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
                pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
                pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])

                label = "{}:{:.2f}".format(mapping[cls], prob)
                color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

                cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, 1)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
                cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, -1)
                cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                            cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)

            if len(rects)!=0:
                counter+=1
                cv2.imwrite("/mnt/workspcae/libin/tar_%d.jpg"%counter,img_ori)

            cv2.imshow('video', img_ori)

            if cv2.waitKey(30) == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

def UseRtsp(rtscap,model,args):
    """
    Use rtsp get video stream
    :param rtscap:
    :return:
    """



    inp_dim = args.resolution, args.resolution

    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame()  # read_latest_frame() 替代 read()

        if frame is None:
            continue
        img_ori = frame.copy()



        img = img_prepare(img_ori, inp_dim)
        #
        model.blobs['data'].data[:] = img
        output = model.forward()

        rects = rects_prepare(output)
        mapping = get_classname_mapping(args.classfile)

        scaling_factor = min(1, args.resolution / img_ori.shape[1])
        for pt1, pt2, cls, prob in rects:
            print(
                "**********************************************************************************************************")
            pt1[0] -= (args.resolution - scaling_factor * img_ori.shape[1]) / 2
            pt2[0] -= (args.resolution - scaling_factor * img_ori.shape[1]) / 2
            pt1[1] -= (args.resolution - scaling_factor * img_ori.shape[0]) / 2
            pt2[1] -= (args.resolution - scaling_factor * img_ori.shape[0]) / 2

            pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
            pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
            pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
            pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])

            label = "{}:{:.2f}".format(mapping[cls], prob)
            color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

            cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
            cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, -1)
            cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                        cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)



        cv2.imshow('video', img_ori)

        if cv2.waitKey(30) == ord('q'):
            break

    cv2.destroyAllWindows()
def main():
    ssb_cam=0
    rtsp_flag=1
    if ssb_cam:
        # USBCamera_writetovideo()
        USBCamera()

    elif rtsp_flag:
        args = parse_args()

        model = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

        rtscap = rtsp.RTSCapture.create('rtsp://admin:Aa123456@192.168.1.11:554/Streaming/Channels/301')
        rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向
        UseRtsp(rtscap,model,args)
        rtscap.stop_read()
        rtscap.release()
        cv2.destroyAllWindows()


    else:
        pass
    args = parse_args()

    model = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    img_ori = cv2.imread(args.image)
    inp_dim = args.resolution, args.resolution



    img = img_prepare(img_ori, inp_dim)

    #cv2.imshow("?", img.transpose([1,2,0]))
    #cv2.waitKey()
    model.blobs['data'].data[:] = img
    output = model.forward()

    rects = rects_prepare(output)
    mapping = get_classname_mapping(args.classfile)

    scaling_factor = min(1, args.resolution / img_ori.shape[1])
    for pt1, pt2, cls, prob in rects:
        print("**********************************************************************************************************")
        pt1[0] -= (args.resolution - scaling_factor*img_ori.shape[1])/2
        pt2[0] -= (args.resolution - scaling_factor*img_ori.shape[1])/2
        pt1[1] -= (args.resolution - scaling_factor*img_ori.shape[0])/2
        pt2[1] -= (args.resolution - scaling_factor*img_ori.shape[0])/2

        pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])

        label = "{}:{:.2f}".format(mapping[cls], prob)
        color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

        cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
        cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, -1)
        cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                    cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)
    cv2.imshow(args.image, img_ori)
    cv2.imwrite('./Test.jpg',img_ori)
    cv2.waitKey()



if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    main()
