import os
import argparse
import cv2
import mediapipe as mp


def process_img(img, face_detection):
    """Process an image to detect and blur faces."""
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)

            # Apply blur to the detected face region
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (100, 100))

    return img


def main(args):
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == 'image':
            if not args.filePath or not os.path.isfile(args.filePath):
                print("Invalid image file path.")
                return

            img = cv2.imread(args.filePath)
            if img is None:
                print("Error reading image.")
                return

            processed_img = process_img(img, face_detection)
            cv2.imwrite(os.path.join(output_dir, 'output.png'), processed_img)
            print("Image saved to", os.path.join(output_dir, 'output.png'))

        elif args.mode == 'video':
            if not args.filePath or not os.path.isfile(args.filePath):
                print("Invalid video file path.")
                return

            cap = cv2.VideoCapture(args.filePath)
            if not cap.isOpened():
                print("Error opening video file.")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25

            output_video = cv2.VideoWriter(
                os.path.join(output_dir, 'output.mp4'),
                cv2.VideoWriter_fourcc(*'MP4V'),
                frame_rate,
                (frame_width, frame_height)
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_img(frame, face_detection)
                output_video.write(processed_frame)

            cap.release()
            output_video.release()
            print("Video saved to", os.path.join(output_dir, 'output.mp4'))

        elif args.mode == 'webcam':
            cap = cv2.VideoCapture(args.deviceIndex)
            if not cap.isOpened():
                print(f"Error opening webcam (index {args.deviceIndex}).")
                return

            print("Press 'q' to quit the webcam view.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_img(frame, face_detection)
                cv2.imshow('Webcam', processed_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face detection and blurring using MediaPipe.")
    parser.add_argument("--mode", type=str, choices=['image', 'video', 'webcam'], required=True,
                        help="Mode of operation: 'image', 'video', or 'webcam'.")
    parser.add_argument("--filePath", type=str, help="Path to the image or video file.")
    parser.add_argument("--deviceIndex", type=int, default=0, help="Webcam device index (default: 0).")

    args = parser.parse_args()

    if args.mode in ['image', 'video'] and not args.filePath:
        parser.error(f"--filePath is required for {args.mode} mode.")
    else:
        main(args)

# Run these in the terminal:

# For Image Mode:

# python face_detection_mediapipe.py --mode image --filePath path/to/image.jpg

# For Video Mode:

# python face_detection_mediapipe.py --mode video --filePath path/to/video.mp4

# For Webcam Mode:

# python face_detection_mediapipe.py --mode webcam --deviceIndex 0
