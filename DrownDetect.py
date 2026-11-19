"""摄像头多人检测演示。参数说明：python DrownDetect.py --help。"""
import argparse
import time

import cv2
import cvlib as cv
from person_tracker import PersonTracker


def parse_args():
    parser = argparse.ArgumentParser(description='多人检测和实验性风险提示')
    parser.add_argument('--camera', type=int, default=0, help='摄像头编号，默认 0')
    parser.add_argument('--waterline', type=float, default=None,
                        help='画面水面参考线位置，0 到 1，例如 0.35；仅固定机位适用')
    parser.add_argument('--enable-risk', action='store_true', help='启用实验性风险提示，必须同时设置水面线')
    parser.add_argument('--window', type=float, default=10.0, help='连续观察窗口，秒')
    parser.add_argument('--max-missing', type=float, default=2.0, help='目标丢失后保留 ID 的时间，秒')
    parser.add_argument('--max-gap', type=float, default=1.5, help='超过此观察间隔则重新累计窗口，秒')
    parser.add_argument('--movement', type=float, default=0.30, help='窗口内相对人体框对角线的运动阈值')
    parser.add_argument('--vertical-ratio', type=float, default=1.2, help='框高/框宽的竖直姿态近似阈值')
    parser.add_argument('--score-threshold', type=float, default=0.75, help='启发式评分告警阈值，0 到 1，并非概率')
    args = parser.parse_args()
    if args.waterline is not None and not 0 <= args.waterline <= 1:
        parser.error('--waterline 必须在 0 到 1 之间')
    if not 0 <= args.score_threshold <= 1:
        parser.error('--score-threshold 必须在 0 到 1 之间')
    if args.enable_risk and args.waterline is None:
        parser.error('--enable-risk 需要同时设置 --waterline')
    if args.window <= 0 or args.max_missing <= 0 or args.max_gap <= 0 or args.movement < 0 or args.vertical_ratio <= 0:
        parser.error('时间和比例参数必须为正，movement 不能为负')
    return args


def main():
    args = parse_args()
    tracker = PersonTracker(max_missing=args.max_missing, max_observation_gap=args.max_gap,
                            window=args.window, movement_threshold=args.movement,
                            vertical_ratio=args.vertical_ratio, waterline=args.waterline,
                            enable_risk=args.enable_risk, score_threshold=args.score_threshold)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print('Could not open camera:', args.camera)
        camera.release()
        return
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print('Could not read frame')
                break
            boxes, labels, confidences = cv.detect_common_objects(frame)
            detections = [(box, confidence) for box, label, confidence in zip(boxes, labels, confidences)
                          if label == 'person']
            tracks = tracker.update(detections, time.monotonic(), frame.shape[0])
            if args.waterline is not None:
                y = round(args.waterline * (frame.shape[0] - 1))
                cv2.line(frame, (0, y), (frame.shape[1] - 1, y), (0, 255, 255), 2)
                cv2.putText(frame, 'Waterline reference', (5, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for track in tracks:
                if not track.visible:
                    continue
                color = (0, 0, 255) if track.risk else (255, 0, 0)
                x1, y1, x2, y2 = track.box
                score = '{:.2f}'.format(track.features['score']) if track.features else 'warming up'
                text = 'ID {} SCORE {} {}'.format(track.id, score, 'POSSIBLE RISK' if track.risk else '')
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow('Person tracking - experimental', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
