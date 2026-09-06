import cv2


class CameraManager:

    def __init__(self, source, camera_id="CAM-01"):
        self.source = source
        self.camera_id = camera_id
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[ERROR] {self.camera_id} connection failed")
            return False

        print(f"[ONLINE] {self.camera_id}")
        return True

    def read_frame(self):
        if self.cap is None:
            return False, None

        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            print(f"[OFFLINE] {self.camera_id}")
            