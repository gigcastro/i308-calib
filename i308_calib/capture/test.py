# import argparse
#
# import cv2
#
# from i308_calib.capture import capture
# from i308_calib.capture.capture import CaptureConfig
#
#
# def start(args):
#
#     # gets capture configuration
#     config = get_capture_config(args)
#
#     cap = capture.new_video_capture(config)
#
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#
#         # if frame is read correctly ret is True
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         show_img = frame.copy()
#
#         cv2.imshow('frame', show_img)
#
#         k = cv2.waitKey(1)
#         if k == ord('q'):
#             # quit
#             break
#
#         elif k == ord('h'):
#
#             keys_help = """
#
#                 h: help,
#                     shows help.
#                 q: quit,
#                     terminates the calibration app.
#
#             """
#             print(keys_help)
#
#         elif k == ord('w'):
#
#             capture.to_yaml(config, args.output)
#
#
#     # When everything done, release the capture
#     # th_cap.stop()
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# def parse_args():
#     arg_parser = argparse.ArgumentParser(
#         description="tests capture device",
#         formatter_class=argparse.RawTextHelpFormatter
#     )
#
#     add_capture_args(arg_parser)
#
#     args = arg_parser.parse_args()
#
#     if not args.video:
#         arg_parser.error("-v (video) must be specified")
#
#     return args
#
#
# def run():
#     args = parse_args()
#     start(args)
#
#
# if __name__ == '__main__':
#     run()
