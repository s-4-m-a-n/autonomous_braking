import cv2
import torch
from helper import plot_bounding_box

# config
VIDEO_SOURCE_PATH = "../test videos/1.mp4"
OUTPUT_IMAGE_PATH = "./config image/1.jpg"
CONFIDENCE = 0.8

# loading source
source = cv2.VideoCapture(VIDEO_SOURCE_PATH)
win_name = "press s to take screenshot"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#loading model
model = torch.hub.load(r'yolov5', 'custom', path=r'yolov5s.pt', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


while cv2.waitKey(1) != 27 or cv2.waitKey(1) != 'q':
    ok, frame = source.read()
    if not ok:
        print("error")
        break

    results = model([frame])

    # draw rectangle bounding box
    b_frame = plot_bounding_box(frame.copy(), results, model.names, CONFIDENCE)

    cv2.imshow(win_name, b_frame)
    print("\n press s or S to take screenshot")

    if cv2.waitKey(1) == ord('s') or cv2.waitKey(1) == ord('S'):
        cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
        print("image saved")
        while True:
            print("\n do you want to re-take screenshot")
            print("press y/n")
            choice = input()
            if choice == 'n' or choice == 'N':
                exit(0)
            elif choice == 'y' or choice == 'Y':
                break
            else:
                print("invalid choice")

    

source.release()
cv2.destroyWindow(win_name)
