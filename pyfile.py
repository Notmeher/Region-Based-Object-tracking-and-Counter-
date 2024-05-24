import cv2
from collections import defaultdict
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture("testing/Bottle.mp4")
START = (1432, -2)
END = (1432, 1300)

# Function to determine if a point is above or below the line
def is_above_line(point, start, end):
    return (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0]) > 0

track_history = defaultdict(list)
crossed_objects = {}
cross_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, classes=[39], persist=True, tracker="bytetrack.yaml")

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy()

    # Visualize the results on the frame
    annotated_frame = frame.copy() 

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        center_x, center_y = int(x), int(y)
        track = track_history[track_id]
        track.append((center_x, center_y))

        if track_id not in crossed_objects:
            crossed_objects[track_id] = "above" if is_above_line((center_x, center_y), START, END) else "below"

        if crossed_objects[track_id] == "above" and not is_above_line((center_x, center_y), START, END):
            cross_count += 1
            crossed_objects[track_id] = "below"
        elif crossed_objects[track_id] == "below" and is_above_line((center_x, center_y), START, END):
            cross_count += 1
            crossed_objects[track_id] = "above"

        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(annotated_frame, top_left, bottom_right, (0, 255, 0), 2)

        text_size, _ = cv2.getTextSize('bottle', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.rectangle(annotated_frame, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), -1)
        cv2.putText(annotated_frame, 'bottle', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.line(annotated_frame, START, END, (255, 255, 255), 4) 

    count_text = f"Count: {cross_count}"
    count_text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    count_text_x = annotated_frame.shape[1] - count_text_size[0] - 10 
    count_text_y = 30
    cv2.rectangle(annotated_frame, (count_text_x - 5, count_text_y - count_text_size[1] - 5),
                  (count_text_x + count_text_size[0] + 5, count_text_y + 5), (255, 0, 0), -1)
    cv2.putText(annotated_frame, count_text, (count_text_x, count_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
