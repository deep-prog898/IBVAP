import cv2

class VideoStream:
    def __init__(self, source, camera_id="CAM-MAIN", max_dim=960):
        self.source = source
        self.camera_id = camera_id
        self.max_dim = max_dim
        self.cap = None
        self.fps = 30
        self.width = 960
        self.height = 540
        self.skip_stride = 1

    def connect(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[ERROR] {self.camera_id} connection failed to {self.source}")
            return False
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
            
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If source is 50-60 FPS (like test4.mp4), skip every 2nd frame so it plays in real-time
        if self.fps >= 50:
            self.skip_stride = 2
        else:
            self.skip_stride = 1
            
        # Downscale 4K/huge videos to max_dim (1280) for real-time CPU performance
        max_orig = max(orig_w, orig_h)
        if max_orig > self.max_dim:
            scale = self.max_dim / max_orig
            self.width = int(orig_w * scale)
            self.height = int(orig_h * scale)
        else:
            self.width = orig_w
            self.height = orig_h
        
        print(f"[ONLINE] {self.camera_id} - Source: {orig_w}x{orig_h}@{self.fps:.0f}FPS -> Stream: {self.width}x{self.height} (skip={self.skip_stride})")
        return True

    def read_frame(self):
        if self.cap is None:
            return False, None
            
        # Fast grab skip for high FPS videos to prevent slow-motion playback
        if self.skip_stride > 1:
            for _ in range(self.skip_stride - 1):
                self.cap.grab()
                
        ret, frame = self.cap.read()
        if ret and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()
            print(f"[OFFLINE] {self.camera_id}")

