import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
import torch # type: ignore

model = YOLO('weights/y8best.pt')  

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

                label = box.cls.item()
                confidence = box.conf.item()

                class_name = model.names[int(label)] if hasattr(model, 'names') else str(label)

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                # Display label and confidence
                cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Pothole Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
