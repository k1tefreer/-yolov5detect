from multiprocessing import Process, Array
from PIL import Image
import numpy as np
import io
import cv2
import torch

from models.experimental import attempt_load
from utils.general import non_max_suppression

from shapely.geometry import Polygon

import time


def load_detector(model_path, device='cpu'):
    model = attempt_load(model_path, device)
    model.eval()
    return model


def calculate_intersection_ratio(polygon_a_vertices, polygon_b_vertices):
    # 创建多边形A和B的shapely对象
    polygon_a = Polygon(polygon_a_vertices)
    polygon_b = Polygon(polygon_b_vertices)

    # 计算交集多边形C
    # intersection_polygon = intersection(polygon_a, polygon_b)
    intersection_polygon = polygon_a.intersection(polygon_b)
    # 检查是否有交集
    if intersection_polygon.is_empty:
        return 0.00  # 没有交集，返回0

    # 计算多边形B的面积和交集C的面积
    area_b = polygon_b.area
    area_intersection = intersection_polygon.area

    # 计算交集占多边形B的比例
    ratio = area_intersection / area_b

    # 保留两位小数
    return round(ratio, 2)


# 推理然后判断是否和墙有交集
def fanyue_pic(imageslst, imgs, model, wallPoints, thresh_person, inter_ratio, device='cpu'):
    # 前向传播
    # while 1:

    pred = model(imgs)[0]  # 前向传播，只获取预测结果（因为我们只预测一张图片）

    # 非极大值抑制（NMS）
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5)  # 你可以调整 conf_thres 和 iou_thres

    # condition = pred[:, -1] == 0
    # # 使用布尔索引来选择满足条件的行
    # pred = pred[condition]
    # 解析预测结果
    for ba, pic in zip(pred, imageslst):
        for det in ba:  # det 是一个列表，每个元素是一个字典，表示一个检测到的物体
            per = det.cpu().numpy()
            x1, y1, x2, y2, score, cls = per
            print(x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if score > thresh_person and cls == 0:
                # polygon_a_vertices = wallPoints
                polygon_person = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                ratio = calculate_intersection_ratio(wallPoints, polygon_person)
                # print(per)
                # print("ratio:  ",ratio)
                print(x1, y1, x2, y2)
                pic = cv2.resize(pic, (640, 640))
                cv2.rectangle(pic, (x1, y1), (x2, y2), (255, 0, 0), 3)
                for i in range(len(wallPoints) - 1):
                    cv2.line(pic, wallPoints[i], wallPoints[i + 1], (255, 255, 0), 5)
                cv2.line(pic, wallPoints[0], wallPoints[- 1], (255, 255, 0), 5)
        cv2.imshow('roi', pic)
        cv2.waitKey(5)
        # cv2.rectangle(img_org, (x1,y1), (x2,y2), (255, 0, 0), 3)

        # cv2.imshow('fuck',img_org)
        # cv2.waitKey(0)
        # 这里可以根据你的需要添加更多逻辑，比如绘制边界框等
        # 368 180     460 536


def batch_test1(shared_memory1, shape, model):
    # model = load_detector('yolov5s.pt', device='cuda:0')
    wallPoints = [(0, 0), (600, 0), (600, 600), (0, 600)]
    thresh_person = 0.1
    inter_ratio = 0.1
    img0 = np.ndarray(shape)
    while True:

        img_data = np.frombuffer(shared_memory1, dtype=np.uint8).reshape(shape)
        cv2.imshow('fuck', img_data)
        cv2.waitKey(1)

        np.copyto(img0, img_data)
        img = cv2.resize(img0, (640, 640))
        img = np.uint(img)
        cv2.imshow('dealing', img)
        cv2.waitKey(40)
        img1 = torch.from_numpy(img).to('cuda:0')
        img1 = img1.float()
        img1 /= 255
        img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)

        pred = model(img1)[0]  # 前向传播，只获取预测结果（因为我们只预测一张图片）

        # 非极大值抑制（NMS）
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5)  # 你可以调整 conf_thres 和 iou_thres
        for ba in pred:
            for det in ba:  # det 是一个列表，每个元素是一个字典，表示一个检测到的物体
                per = det.cpu().numpy()
                x1, y1, x2, y2, score, cls = per
                print(x1, y1, x2, y2)
            #
            # for img in imageslst:
            #     img = cv2.resize(img, (640, 640))
            #     # cv2.imshow('dealing', img)
            #     # cv2.waitKey(40)
            #     img = torch.from_numpy(img).to('cuda:0')
            #     img = img.float()
            #     #img /= 255
            #     tensor_images.append(img)
            # # 使用torch.stack来创建一个新的维度（batch维度），这要求所有图像具有相同的尺寸
            # # 注意：stack会增加一个新的维度（在这里是第一个维度），所以输出张量的形状将是(batch_size, C, H, W)
            # batch_tensor = torch.stack(tensor_images, dim=0)
            # batch_tensor = batch_tensor.permute(0, 3, 1, 2)
            # fanyue_pic(imageslst,batch_tensor, model, wallPoints, thresh_person, inter_ratio, device='cuda:0')


def load_image_into_shared_memory(rtsp_url, shared_memory1, shared_memory2):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error opening {rtsp_url}")
    while (not cap.isOpened()):
        cap = cv2.VideoCapture(rtsp_url)
    i = 0
    while True:
        # time.sleep(0.04)
        ret, frame = cap.read()
        i = i + 1
        cv2.imshow(rtsp_url, frame)
        cv2.waitKey(1)
        # 打开图片并转换为灰度图
        # img = Image.open('friday.png').convert('L')
        #     img = cv2.imread('friday.png')
        if i % 2 == 0:
            img_data = np.array(frame)

            # 将图像数据拷贝到共享内存

            shared_memory1[:] = img_data.flatten()
            shared_memory2[:] = img_data.flatten()


def read_image_from_shared_memory(shared_memory, shape, name):
    # 从共享内存读取图像数据并恢复原始形状
    # img_data = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8).reshape(shape)
    while 1:
        img_data = np.frombuffer(shared_memory, dtype=np.uint8).reshape(shape)

        # img = Image.fromarray(img_data)
        # img.show()
        cv2.imshow(name, img_data)
        cv2.waitKey(1)


def test_pt():
    model = load_detector('yolov5s.pt', device='cuda:0')
    rtsp_url = 'rtsp://admin:jiankong123@192.168.23.17:554/cam/realmonitor?channel=1&subtype=0'
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()

        img = cv2.resize(frame, (640, 640))
        # cv2.imshow('dealing', img)
        # cv2.waitKey(40)
        img = torch.from_numpy(img).to('cuda:0')
        img = img.float()
        img /= 255
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        pred = model(img)[0]  # 前向传播，只获取预测结果（因为我们只预测一张图片）

        # 非极大值抑制（NMS）
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5)  # 你可以调整 conf_thres 和 iou_thres
        for ba in pred:
            for det in ba:  # det 是一个列表，每个元素是一个字典，表示一个检测到的物体
                per = det.cpu().numpy()
                x1, y1, x2, y2, score, cls = per
                print(x1, y1, x2, y2)


if __name__ == '__main__':
    # test_pt()
    # 加载图像并获取其大小和数据类型
    # img = Image.open('friday.png').convert('L')
    # img = cv2.imread('friday.png')
    # img_data = np.array(img)
    # print(img_data.shape)
    # rtsp_url = 'zx.mp4'
    rtsp_url = 'rtsp://admin:jiankong123@192.168.23.19:554/cam/realmonitor?channel=1&subtype=0'
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    print(frame.shape)

    model1 = load_detector('yolov5s.pt', device='cuda:0')
    model2 = load_detector('yolov5s.pt', device='cuda:0')
    # 创建共享内存数组2764800
    shared_array1 = Array('B', frame.size, lock=False)
    shared_array2 = Array('B', frame.size, lock=False)

    # 创建并启动将图像数据加载到共享内存的进程
    p1 = Process(target=load_image_into_shared_memory, args=(rtsp_url, shared_array1, shared_array2))
    p1.start()
    # p2 = Process(target=read_image_from_shared_memory, args=(shared_array1, frame.shape,'p2'))
    # p2.start()

    # 创建并启动从共享内存读取图像数据的进程
    # name = 'p2'
    # p2 = Process(target=batch_test1, args=(shared_array1, frame.shape,model1))
    # p2.start()
    # name = 'p3'
    p2 = Process(target=batch_test1, args=(shared_array2, frame.shape, model2))
    p2.start()

    p1.join()
    p2.join()
