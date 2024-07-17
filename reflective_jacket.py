import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import pathlib

# 解决在Windows环境下的路径问题
pathlib.PosixPath = pathlib.WindowsPath

# 替换为您的类别标签
labels = ['Safety-Helmet', 'Reflective-Jacket']


def load_detector(model_path, device='cpu'):
    model = attempt_load(model_path, device)
    model.eval()  # 将模型设置为评估（推理）模式
    return model


def test_pt():
    # 加载模型
    model = load_detector('runs/reflective_jacket.pt', device='cuda:0')

    # 替换为您的视频流地址或图像路径
    rtsp_url = 'own_datas/test/reflect2.mp4'  # 这里假设是一张图片，如果是视频可以调整为视频流地址
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 打印当前帧的形状（调试用）
        print(frame.shape)

        # 调整输入图像大小为模型的输入尺寸
        img = cv2.resize(frame, (640, 640))

        # 将图像转换为 PyTorch Tensor，并移动到 CUDA 设备上
        img = torch.from_numpy(img).to('cuda:0').float() / 255.0  # 对图像进行归一化处理
        img = img.unsqueeze(0).permute(0, 3, 1, 2)  # 调整维度顺序以匹配模型要求 (batch_size, channels, height, width)

        # 使用模型进行推理
        with torch.no_grad():
            pred = model(img)[0]  # 前向传播，只获取预测结果（因为我们只预测一张图片）

        # 将预测结果转移到 CPU 上进行 NMS
        pred = pred.cpu()
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5)

        # 处理检测结果
        for detections in pred:
            if detections is not None and len(detections):
                for det in detections:
                    per = det.cpu().numpy()
                    x1, y1, x2, y2, score, cls = per[:6]

                    # 只处理类别为 1 (Reflective-Jacket) 的情况
                    if int(cls) == 1:
                        # 获取标签名称
                        label_name = labels[int(cls)]
                        # 在图像上绘制边界框和标签
                        cv2.rectangle(frame, (int(x1 / 640 * frame.shape[1]), int(y1 / 640 * frame.shape[0])),
                                      (int(x2 / 640 * frame.shape[1]), int(y2 / 640 * frame.shape[0])), (0, 255, 0), 2)
                        label = f"{label_name}, Score: {score:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    2)

        # 创建一个可调整大小的窗口
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

        # 显示结果图像
        cv2.imshow("Detection", frame)

        # 按下任意键退出当前帧的显示，按 'q' 键退出整个程序
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pt()
