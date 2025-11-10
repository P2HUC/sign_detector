import os
import argparse

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

parser = argparse.ArgumentParser(description='Collect hand sign images per class')
parser.add_argument('--num-classes', type=int, default=3, help='Number of class labels to capture (0..N-1)')
parser.add_argument('--dataset-size', type=int, default=100, help='Images to capture per class')
parser.add_argument('--start-class', type=int, default=0, help='Start from this class index (resume)')
parser.add_argument('--add-class', type=int, default=None, help='Capture only this class index (e.g., 3)')
parser.add_argument('--label-name', type=str, default=None, help='Capture only this named label (e.g., Hello)')
args = parser.parse_args()

number_of_classes = max(1, args.num_classes)
dataset_size = max(1, args.dataset_size)
start_class = max(0, args.start_class)

def open_first_available_camera(preferred_indices=(0, 1, 2, 3)):
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    return None


cap = open_first_available_camera()
if cap is None:
    print("ERROR: No available camera. Try connecting a webcam and rerun.")
    raise SystemExit(1)
def _next_start_index(dir_path: str) -> int:
    try:
        existing = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    except FileNotFoundError:
        return 0
    max_idx = -1
    for name in existing:
        base, ext = os.path.splitext(name)
        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        try:
            idx = int(base)
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1


def collect_for_class(j: int):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Keep trying to read a valid frame without crashing
            if cv2.waitKey(10) == ord('q'):
                break
            continue
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start after the last existing index to avoid overwriting
    counter = _next_start_index(os.path.join(DATA_DIR, str(j)))
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Skip this iteration if frame is invalid
            if cv2.waitKey(10) == ord('q'):
                break
            continue
        cv2.putText(
            frame,
            'Capturing: S=Skip  N=Next class  Q=Quit',
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f'Class {j}  Saved: {counter}/{dataset_size}',
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)
        if key == ord('n'):
            # Move to next class early
            break
        if key == ord('s'):
            # Skip saving this frame
            continue

        save_path = os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter))
        if cv2.imwrite(save_path, frame):
            counter += 1


def collect_for_label_name(label_name: str):
    label_dir = os.path.join(DATA_DIR, label_name)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print('Collecting data for label {}'.format(label_name))

    # Ready gate
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if cv2.waitKey(10) == ord('q'):
                return
            continue
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start after the last existing index to avoid overwriting
    counter = _next_start_index(label_dir)
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            if cv2.waitKey(10) == ord('q'):
                return
            continue
        cv2.putText(
            frame,
            'Capturing: S=Skip  Q=Quit',
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f'Label {label_name}  Saved: {counter}/{dataset_size}',
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)
        if key == ord('s'):
            continue

        save_path = os.path.join(label_dir, '{}.jpg'.format(counter))
        if cv2.imwrite(save_path, frame):
            counter += 1


if args.label_name is not None:
    collect_for_label_name(args.label_name)
elif args.add_class is not None:
    # Capture only the specified class index
    collect_for_class(args.add_class)
else:
    for j in range(start_class, number_of_classes):
        collect_for_class(j)

cap.release()
cv2.destroyAllWindows()
