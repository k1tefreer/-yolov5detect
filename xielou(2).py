import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

# Replace with your class labels
labels = ['water']

def load_detector(model_path, device='cpu'):
    model = attempt_load(model_path, device)
    model.eval()
    return model

def test_pt():
    # Load model
    model = load_detector('runs/best.pt', device='cuda:0')

    # Replace with your video stream URL or image path
    rtsp_url = 'own_datas/test/aaa.mp4'  # Assuming it's a video, adjust if it's an image
    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize input frame to match model's input size
        img = cv2.resize(frame, (640, 640))

        # Convert image to PyTorch Tensor and move to CUDA device
        img = torch.from_numpy(img).to('cuda:0').float() / 255.0
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        # Perform inference with the model
        with torch.no_grad():
            pred = model(img)[0]

        # Move predictions to CPU for NMS
        pred = pred.cpu()
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5)

        # Process detections
        for detections in pred:
            if detections is not None and len(detections):
                for det in detections:
                    per = det.cpu().numpy()
                    x1, y1, x2, y2, score, cls = per[:6]

                    # Get label name
                    label_name = labels[int(cls)]

                    # Draw bounding box and label on the frame
                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1 / 640 * frame.shape[1]), int(y1 / 640 * frame.shape[0])),
                                  (int(x2 / 640 * frame.shape[1]), int(y2 / 640 * frame.shape[0])), (0, 255, 0), 2)
                    label = f"{label_name}, Score: {score:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("Detection", frame)

        #  cv2.waitKey(1) 只等待 1 毫秒后就会返回 整个循环会迅速地一帧接一帧地执行 呈现出视频连续播放的效果。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pt()
