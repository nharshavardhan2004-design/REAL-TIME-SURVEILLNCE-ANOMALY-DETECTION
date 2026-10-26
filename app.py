import torch 
 
def _is_tensor_video_clip(clip): 
if not torch.is_tensor(clip): 
raise TypeError(f"clip should be Tensor. Got {type(clip)}") 
 
if not clip.ndimension() == 4: 
raise ValueError(f"clip should be 4D. Got {clip.dim()}D") 
 
return True 
 
def crop(clip, i, j, h, w): 
""" 
    Args: 
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W) 
    """ 
    assert len(clip.size()) == 4, "clip should be a 4D tensor" 
    return clip[..., i : i + h, j : j + w] 
 
def resize(clip, target_size, interpolation_mode): 
    assert len(target_size) == 2, "target size should be tuple (height, width)" 
    return torch.nn.functional.interpolate( 
        clip, size=target_size, mode=interpolation_mode, align_corners=False 
    ) 
 
def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"): 
    """ 
    Do spatial cropping and resizing to the video clip 
    Args: 
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W) 
        i (int): i in (i,j) i.e coordinates of the upper left corner. 
        j (int): j in (i,j) i.e coordinates of the upper left corner. 
        h (int): Height of the cropped region. 
        w (int): Width of the cropped region. 
        size (tuple(int, int)): height and width of resized clip 
    Returns: 
        clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W) 
    """ 
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor" 
    clip = crop(clip, i, j, h, w) 
    clip = resize(clip, size, interpolation_mode) 
    return clip 
def center_crop(clip, crop_size): 
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor" 
    h, w = clip.size(-2), clip.size(-1) 
    th, tw = crop_size 
    assert h >= th and w >= tw, "height and width must be no smaller than crop_size" 
 
    i = int(round((h - th) / 2.0)) 
    j = int(round((w - tw) / 2.0)) 
    return crop(clip, i, j, th, tw) 
def to_tensor(clip): 
    """ 
    Convert tensor data type from uint8 to float, divide value by 255.0 and 
   
  permute the dimenions of clip tensor 
    Args: 
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C) 
    Return: 
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W) 
    """ 
    _is_tensor_video_clip(clip) 
    if not clip.dtype == torch.uint8: 
        raise TypeError( 
            f"clip tensor should have data type uint8. Got {str(clip.dtype)}" 
        ) 
    return clip.float().permute(3, 0, 1, 2) / 255.0 
 
def normalize(clip, mean, std, inplace=False): 
    """ 
    Args: 
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W) 
        mean (tuple): pixel RGB mean. Size is (3) 
        std (tuple): pixel standard deviation. Size is (3) 
    Returns: 
        normalized clip (torch.tensor): Size is (C, T, H, W) 
    """ 
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor" 
    if not inplace: 
        clip = clip.clone() 
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device) 
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device) 
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None]) 
    return clip 
 
def hflip(clip): 
    """ 
    Args: 
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W) 
    Returns: 
        flipped clip (torch.tensor): Size is (C, T, H, W) 
    """ 
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor" 
    return clip.flip(-1) 
  
"""This module contains a procedure for real time anomaly detection.""" 
 
import argparse 
import logging 
import sys 
from collections.abc import Callable 
 
import cv2 
import numpy as np 
import torch 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 
from matplotlib.figure import Figure 
from numpy.lib.function_base import copy 
from PyQt5 import QtGui 
from PyQt5.QtCore import Qt, QThread  # pylint: disable=no-name-in-module 
 
from PyQt5.QtGui import QIcon, QPalette, QPixmap  # pylint: disable=no-name-in-module 
from PyQt5.QtMultimedia import (  # pylint: disable=no-name-in-module 
    QCameraInfo, 
    QMediaPlayer, 
) 
from PyQt5.QtWidgets import QComboBox  # pylint: disable=no-name-in-module 
from PyQt5.QtWidgets import ( 
    QApplication, 
    QGridLayout, 
    QLabel, 
    QPushButton, 
    QStyle, 
    QWidget, 
) 
from torch import Tensor, nn 
from feature_extractor import to_segments 
from network.TorchUtils import get_torch_device 
from utils.load_model import load_models 
from utils.stack import Stack 
from utils.types import Device, FeatureExtractor 
from utils.utils import build_transforms 
 
MAX_PREDS = 50 
 
# pylint disable=missing-function-docstring 
def get_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="Video Demo For Anomaly Detection") 
 
    parser.add_argument( 
        "--feature_extractor", 
        required=True, 
        help="path to the 3d model for feature extraction", 
    ) 
    parser.add_argument( 
        "--feature_method", 
        default="c3d", 
        choices=["c3d", "mfnet", "r3d101", "r3d152", "r2plus1d", "s3d", "mvit"], 
        help="method to use for feature extraction", 
    ) 
    parser.add_argument( 
        "--ad_model", required=True, help="path to the trained AD model" 
    ) 
    parser.add_argument( 
        "--clip_length", 
        type=int, 
        default=16, 
        help="define the length of each input sample", 
    ) return parser.parse_args() 
 
def features_extraction( 
    frames, 
    model: FeatureExtractor, 
    device: Device, 
   
  frame_stride: int = 1, 
    transforms: Callable | None = None, 
) -> list[np.ndarray]: 
    """Extracts features of the video. The returned features will be returned 
    after averaging over the required number of video segments. 
 
    Args: 
        frames: a sequence of video frames to predict 
        model: model to use for feature extraction 
        device: device to use for loading data 
        frame_stride: interval between frames to load 
 
    Returns: 
        feature (1, feature_dim), usually (1, 4096) as in the original paper 
    """ 
 
    frames = torch.tensor(frames, device=device)  # pylint: disable=not-callable 
    if transforms is not None: 
        frames = transforms(frames) 
    data = frames[:, range(0, frames.shape[1], frame_stride), ...] 
    data = data.unsqueeze(0) 
 
    with torch.no_grad(): 
        outputs = model(data.to(device)).detach().cpu() 
 
    return to_segments(outputs.numpy(), 1) 
 
def ad_prediction( 
    model: nn.Module, features: Tensor, device: Device = "cuda" 
) 
-> np.ndarray: 
    """Creates prediction for the given feature vectors. 
 
    Args: 
        model: model to use for anomaly detection 
        features: features of the video clips 
        device: device to use for loading the features 
 
    Returns: 
        Anomaly predictions for the video segments 
    """ 
    logging.info("Performing anomaly detection...") 
    features = torch.tensor(features).to(device)  # pylint: disable=not-callable 
    with torch.no_grad(): 
        preds = model(features) 
 
    return preds.detach().cpu().numpy().flatten() 
class VideoThread(QThread):
   """Read video stream and store frames in a stack.""" 
 
    def __init__( 
        self, stack: Stack, preprocess_fn: Callable, camera_view: QLabel 
    ) -> None: 
        super().__init__() 
        self._run_flag = True 
        
 self._stack = stack 
        self._preprocess_fn = preprocess_fn 
        self._camera_view = camera_view 
 
    def run(self) -> None: 
        # capture from web cam 
        cap = cv2.VideoCapture(0) 
        while self._run_flag: 
            ret, cv_img = cap.read() 
            if ret: 
                qt_img = self._preprocess_fn(cv_img) 
                self._camera_view.setPixmap(qt_img) 
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) 
                self._stack.put(cv_img) 
        # shut down capture system 
        cap.release() 
 
    def stop(self) -> None: 
        """Sets run flag to False and waits for thread to finish.""" 
        self._run_flag = False 
        self.wait() 
 
class VideoConsumer(QThread): 
    """Consume frames from a stack and perform predictions.""" 
 
    def __init__(self, stack: Stack, prediction_function: Callable) -> None: 
        super().__init__() 
        self._run_flag = True 
        self._stack = stack 
        self._prediction_function = prediction_function 
 
    def run(self) -> None: 
        while self._run_flag: 
            with self._stack._lock: 
                if not self._stack.full(): 
                    continue 
 
                batch = copy(list(reversed(self._stack.get()))) 
            self._prediction_function(batch) 
 
    def stop(self) -> None: 
        """Sets run flag to False and waits for thread to finish.""" 
        self._run_flag = False 
        self.wait() 
 
# pylint: disable=missing-class-docstring 
class MplCanvas(FigureCanvasQTAgg): 
 # pylint: disable=unused-argument 
    def __init__(self, parent=None, width=5, height=4, dpi=100) -> None: 
        self.fig = Figure(figsize=(width, height), dpi=dpi) 
        self.axes = self.fig.add_subplot(111) 
        super().__init__(self.fig) 
 
class Window(QWidget): 
   
 
  """Anomaly detection live gui Based on media player code from: 
 
    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/ 
    """ 
    def __init__(self, clip_length=16, transforms=None) -> None: 
        super().__init__() 
 
        self.camera = None 
        self.current_camera_name = None 
        self.clip_length = clip_length 
 
        self.frames_stack = Stack(max_size=self.clip_length) 
 
        self.transforms = transforms 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
        self.setWindowTitle("Anomaly Media Player") 
        self.setGeometry(350, 100, 700, 500) 
        self.setWindowIcon(QIcon("player.png")) 
 
        p = self.palette() 
        p.setColor(QPalette.Window, Qt.black) 
        self.setPalette(p) 
 
        self.init_ui() 
 
        self.y_pred = [] 
 
        self.show() 
 
    def init_ui(self) -> None:    
 
 """Create media player object.""" 
        # setup camera 
        self.available_cameras = QCameraInfo.availableCameras() 
        self.camera_view = QLabel() 
        self.frames_stack = Stack(max_size=self.clip_length) 
        self.camera_id = None 
        self.select_camera() 
 
        # creating a combo box for selecting camera 
        camera_selector = QComboBox() 
        camera_selector.setStatusTip("Choose camera") 
        camera_selector.setToolTip("Select Camera") 
        camera_selector.setToolTipDuration(2500) 
        camera_selector.addItems(
           [camera.description() for camera in self.available_cameras] 
        ) 
        camera_selector.currentIndexChanged.connect(self.select_camera) 
 
        # create button for playing 
        self.playBtn = QPushButton() 
        self.playBtn.setEnabled(True) 
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)) 
        self.playBtn.clicked.connect(self.play_video) 
 
      
 
   # create grid layout 
  grid_layout = QGridLayout() 
 
        # AD signal 
        self.graphWidget = MplCanvas(self, width=5, height=1, dpi=100) 
 
        # set widgets to the hbox layout 
        grid_layout.addWidget(self.graphWidget, 0, 0, 1, 5) 
        grid_layout.addWidget(self.camera_view, 1, 0, 5, 5) 
 
        self.setLayout(grid_layout) 
        # create the video capture thread 
        self.thread = VideoThread( 
            stack=self.frames_stack, 
            preprocess_fn=self.convert_cv_qt, 
            camera_view=self.camera_view, 
        ) 
        self._video_consumer = VideoConsumer( 
            stack=self.frames_stack, prediction_function=self.perform_prediction 
        ) 
        self.thread.start() 
        self._video_consumer.start() 
 
    def perform_prediction(self, batch): 
        features = features_extraction( 
            frames=batch, 
            model=feature_extractor, 
            device=self.device, 
            transforms=self.transforms, 
        ) 
        new_pred = ad_prediction( 
            model=anomaly_detector, 
            features=features, 
            device=self.device, 
        )[0] 
        self.y_pred.append(new_pred) 
        if len(self.y_pred) > MAX_PREDS: 
            del self.y_pred[0] 
        self.plot() 
 
    def convert_cv_qt(self, cv_img: np.ndarray) -> QPixmap: 
        """Convert from an opencv image to QPixmap.""" 
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape 
        display_height, display_width = ( 
            self.camera_view.height(), 
            self.camera_view.width(), 
        ) 
        bytes_per_line = ch * w 
        convert_to_Qt_format = QtGui.QImage( 
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888 
        ) 
        p = convert_to_Qt_format.scaled( 
           
  display_width, display_height, Qt.KeepAspectRatio 
        ) 
        return QPixmap.fromImage(p) 
 
    def select_camera(self, camera=0) -> None: 
        """Select camera to display.""" 
        # getting the selected camera 
        self.camera = cv2.VideoCapture(camera) 
 
        # getting current camera name 
        self.current_camera_name = self.available_cameras[camera].description() 
 
    def play_video(self) -> None: 
        """Change the state of the media player.""" 
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState: 
            self.mediaPlayer.pause() 
        else: 
            self.mediaPlayer.play() 
 
    def mediastate_changed(self, *_args) -> None: 
        """Called when the state of the media player changes.""" 
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState: 
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause)) 
 
        else: 
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)) 
 
    def handle_errors(self) -> None: 
        self.playBtn.setEnabled(False) 
        self.label.setText("Error: " + self.mediaPlayer.errorString()) 
 
    def plot(self) -> None: 
        ax = self.graphWidget.axes 
        ax.clear() 
        ax.set_xlim(0, MAX_PREDS) 
        ax.set_ylim(-0.1, 1.1) 
        ax.plot(self.y_pred, "*-", linewidth=5) 
        self.graphWidget.draw() 
 
if __name__ == "__main__": 
    args = get_args() 
 
    anomaly_detector, feature_extractor = load_models( 
        args.feature_extractor,
   args.ad_model, 
        features_method=args.feature_method, 
        device=get_torch_device(), 
    ) 
 
    transforms = build_transforms(mode=args.feature_method) 
 
    app = QApplication(sys.argv) 
    window = Window(args.clip_length, transforms) 
 
    sys.exit(app.exec_()) 
 
"""This module contains a video feature loader.""" 
 
import logging 
import os 
 
import numpy as np 
import torch 
from torch import Tensor 
from torch.utils import data 
 
from feature_extractor import read_features 
 
class FeaturesLoader: 
    """Loads video features that are stored as text files.""" 
 
    def __init__( 
        self, 
        features_path: str, 
        annotation_path: str, 
        bucket_size: int = 30, 
        iterations: int = 20000, 
    ) -> None: 
        """ 
        Args: 
            features_path: Path to the directory that contains the features in text files 
            annotation_path: Path to the annotation file 
            bucket_size: Size of each bucket 
            iterations: How many iterations the loader should perform 
        """ 
 
        super().__init__() 
        self._features_path = features_path 
        self._bucket_size = bucket_size 
 
        # load video list 
        ( 
            self.features_list_normal, 
            self.features_list_anomaly, 
        ) = FeaturesLoader._get_features_list( 
            features_path=self._features_path, annotation_path=annotation_path 
        ) 
 
        self._iterations = iterations 
       self._features_cache = {} 
        self._i = 0 
 
    def __len__(self) -> int: 
        return self._iterations 
 
    def __getitem__(self, index: int, max_retries: int = 10) -> tuple[Tensor, Tensor]: 
        if self._i == len(self): 
            self._i = 0 
            raise StopIteration 
        succ = False 
        retries = 0 
 
        while not succ: 
            try: 
                feature, label = self.get_features() 
                succ = True 
            except Exception as e: 
                retries += 1 
                if retries >= max_retries: 
                    raise RuntimeError( 
                        f"FeaturesLoader:: failed to load a sample after {max_retries} retries" 
                    ) from e 
                index = np.random.choice(range(0, self.__len__())) 
                logging.warning( 
                    f"FeaturesLoader:: ERROR!! (Force using another index:\n{index})\n{e}" 
                ) 
 
        self._i += 1 
        return feature, label 
 
    def get_features(self) -> tuple[Tensor, Tensor]: 
        """Fetches a bucket sample from the dataset.""" 
        normal_paths = np.random.choice( 
            self.features_list_normal, size=self._bucket_size 
        ) 
        abnormal_paths = np.random.choice( 
            self.features_list_anomaly, size=self._bucket_size 
        ) 
        all_paths = np.concatenate([normal_paths, abnormal_paths]) 
        features = torch.stack( 
            [ 
                read_features(f"{feature_subpath}.txt", self._features_cache) 
                for feature_subpath in all_paths 
            ] 
        ) 
        return ( 
            features, 
            torch.cat([torch.zeros(self._bucket_size), torch.ones(self._bucket_size)]), 
        ) 
 
    @staticmethod 
    def _get_features_list( 
        features_path: str, annotation_path: str 
    ) -> tuple[list[str], list[str]]:
         """Retrieves the paths of all feature files contained within the 
        annotation file. 
 
        Args: 
            features_path: Path to the directory that contains feature text files 
            annotation_path: Path to the annotation file 
 
        Returns: 
            Tuple[List[str], List[str]]: Two list that contain the corresponding paths of normal and abnormal 
                feature files. 
        """ 
        if not os.path.exists(features_path): 
            raise FileNotFoundError( 
                f"FeaturesLoader:: failed to locate features path: `{features_path}'" 
            ) 
        features_list_normal = [] 
        features_list_anomaly = [] 
        with open(annotation_path) as f: 
            lines = f.read().splitlines(keepends=False) 
            for line in lines: 
                items = line.split() 
                file = items[0].split(".")[0] 
                file = file.replace("/", os.sep) 
                feature_path = os.path.join(features_path, file) 
                if "Normal" in feature_path: 
                    features_list_normal.append(feature_path) 
                else: 
                    features_list_anomaly.append(feature_path) 
 
        return features_list_normal, features_list_anomaly 
 
    @property 
    def get_feature_dim(self) -> int: 
        return self[0][0].shape[-1] 
 
class FeaturesLoaderVal(data.Dataset): 
    """Loader for video features for validation phase.""" 
 
    def __init__(self, features_path: str, annotation_path: str) -> None: 
        super().__init__() 
        self.features_path = features_path 
        # load video list 
        self.state = "Normal" 
        self.features_list = FeaturesLoaderVal._get_features_list( 
            features_path=features_path, annotation_path=annotation_path 
        ) 
 
    def __len__(self) -> int: 
        return len(self.features_list) 
 
    def __getitem__(self, index: int, max_retries: int = 10): 
        succ = False 
        retries = 0 
        while not succ: 
            try:
               data = self.get_feature(index) 
                succ = True 
            except Exception as e: 
                retries += 1 
                if retries >= max_retries: 
                    raise RuntimeError( 
                        f"FeaturesLoaderVal:: failed to load a sample after {max_retries} retries" 
                    ) from e 
                logging.warning( 
                    f"FeaturesLoaderVal:: ERROR!! (Force using another index:\n{index})\n{e}" 
                ) 
 
        return data 
 
    def get_feature(self, index: int) -> tuple[Tensor, Tensor, int]: 
        """Fetch feature that matches given index in the dataset. 
 
        Args: 
            index (int): Index of the feature to fetch. 
 
        Returns: 
            Tuple[Tensor, Tensor, int]: Feature tensor, start/end anomaly frame couples, and 
                the total length of the video. 
        """ 
        feature_subpath, start_end_couples, length = self.features_list[index] 
        features = read_features(f"{feature_subpath}.txt") 
        return features, start_end_couples, length 
 
    @staticmethod 
    def _get_features_list( 
        features_path: str, annotation_path: str 
    ) -> list[tuple[str, Tensor, int]]: 
        """Retrieves the paths of all feature files contained within the 
        annotation file. 
 
        Args: 
            features_path: Path to the directory that contains feature text files 
            annotation_path: Path to the annotation file 
 
        Returns: 
            List[Tuple[str, Tensor, int]]: A list of tuples that describe each video and the temporal annotations 
                of anomalies in the videos 
        """ 
        if not os.path.exists(features_path): 
            raise FileNotFoundError( 
                f"FeaturesLoaderVal:: failed to locate features path: `{features_path}'" 
            ) 
        features_list = [] 
        with open(annotation_path) as f: 
            lines = f.read().splitlines(keepends=False) 
            for line in lines: 
                items = line.split() 
                anomalies_frames = [int(x) for x in items[3:]] 
                start_end_couples = torch.tensor( 
                    [anomalies_frames[:2], anomalies_frames[2:]]
                    ) 
                file = items[0].split(".")[0] 
                file = file.replace("/", os.sep) 
                feature_path = os.path.join(features_path, file) 
                length = int(items[1]) 
 
                features_list.append((feature_path, start_end_couples, length)) 
 
        return features_list 
"""This module contains a video demo for anomaly detection.""" 
 
import argparse 
import logging 
import sys 
from os import path 
import numpy as np 
import torch 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 
from matplotlib.figure import Figure 
from PyQt5.QtCore import Qt, QUrl  # pylint: disable=no-name-in-module 
from PyQt5.QtGui import QIcon, QPalette  # pylint: disable=no-name-in-module 
from PyQt5.QtMultimedia import (  # pylint: disable=no-name-in-module 
    QMediaContent, 
    QMediaPlayer, 
) 
from PyQt5.QtMultimediaWidgets import QVideoWidget  # pylint: disable=no-name-in-module 
from PyQt5.QtWidgets import QApplication  # pylint: disable=no-name-in-module 
from PyQt5.QtWidgets import ( 
    QFileDialog, 
    QGridLayout, 
    QLabel, 
    QMessageBox, 
    QProgressBar, 
    QPushButton, 
    QSizePolicy, 
    QSlider, 
    QStyle, 
    QWidget, 
) 
from torch import Tensor 
from tqdm import tqdm 
 
from data_loader import SingleVideoIter 
from feature_extractor import read_features, to_segments 
from network.anomaly_detector_model import AnomalyDetector 
from network.TorchUtils import get_torch_device 
from utils.load_model import load_models 
from utils.types import Device, FeatureExtractor 
from utils.utils import build_transforms 
 
APP_NAME = "Anomaly Media Player" 
 
def get_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="Video Demo For Anomaly Detection") 
parser.add_argument( 
        "--feature_extractor", 
        required=True, 
        help="path to the 3d model for feature extraction", 
    ) 
    parser.add_argument( 
        "--feature_method", 
        default="c3d", 
        choices=["c3d", "mfnet", "r3d101", "r3d101"], 
        help="method to use for feature extraction", 
    ) 
    parser.add_argument( 
        "--ad_model", required=True, help="path to the trained AD model" 
    ) 
    parser.add_argument( 
        "--n_segments", 
        type=int, 
        default=32, 
        help="number of segments to use for features averaging", 
    ) 
 
    return parser.parse_args() 
 
def features_extraction( 
    video_path: str, 
    model: FeatureExtractor, 
    device: Device, 
    batch_size: int = 1, 
    frame_stride: int = 1, 
    clip_length: int = 16, 
    n_segments: int = 32, 
    progress_bar=None, 
) -> list[np.ndarray]: 
    """Extracts features of the video.The returned features will be returned 
    after averaging over the required number of video segments. 
 
    :param video_path: path of the video to predict 
    :param model: model to use for feature extraction 
    :param device: device to use for loading data 
    :param batch_size: batch size to use for loading data 
    :param frame_stride: interval between frames to load 
    :param clip_length: number of frames to use for loading each video 
        sample 
    :param n_segments: how many chunks the video should be divided into 
    :param progress_bar: TODO 
    :return: features list (n_segments, feature_dim), usually (32, 4096) 
        as in the original paper 
    """ 
    data_loader = SingleVideoIter( 
        clip_length=clip_length, 
        frame_stride=frame_stride, 
        video_path=video_path, 
        video_transform=build_transforms(mode=args.feature_method), 
        return_label=False, 
    ) 
data_iter = torch.utils.data.DataLoader( 
        data_loader, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # 4, # change this part accordingly 
        pin_memory=True, 
    ) 
 
    logging.info("Extracting features...") 
    features = torch.tensor([])  # pylint: disable=not-callable 
 
    if progress_bar is not None: 
        progress_bar.setRange(0, len(data_iter)) 
 
    with torch.no_grad(): 
        for i, data in tqdm(enumerate(data_iter)): 
            outputs = model(data.to(device)).detach().cpu() 
            features = torch.cat([features, outputs]) 
 
            if progress_bar is not None: 
                progress_bar.setValue(i + 1) 
 
    features = features.numpy() 
    return to_segments(features, n_segments) 
 
def ad_prediction(model: AnomalyDetector, features: Tensor, device: Device) -> Tensor: 
    """Creates prediction for the given feature vectors. 
 
    :param model: model to use for anomaly detection 
    :param features: features of the video clips 
    :param device: device to use for loading the features 
    :return: anomaly predictions for the video segments 
    """ 
    logging.info("Performing anomaly detection...") 
    features = torch.tensor(features).to(device)  # pylint: disable=not-callable 
    with torch.no_grad(): 
        preds = model(features) 
 
    return preds.detach().cpu().numpy().flatten() 
 
class MplCanvas(FigureCanvasQTAgg): 
    # pylint: disable=unused-argument 
    def __init__(self, parent=None, width=5, height=4, dpi=100) -> None: 
        self.fig = Figure(figsize=(width, height), dpi=dpi) 
        self.axes = self.fig.add_subplot(111) 
        super().__init__(self.fig) 
 
class Window(QWidget): 
    """Anomaly detection gui Based on media player code from: 
 
    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/ 
    """ 
    def __init__(self) -> None: 
        super().__init__()
       self.device = get_torch_device() 
 
        self.setWindowTitle(APP_NAME) 
        self.setGeometry(350, 100, 700, 500) 
        self.setWindowIcon(QIcon("player.png")) 
 
        p = self.palette() 
        p.setColor(QPalette.Window, Qt.black) 
        self.setPalette(p) 
 
        self.init_ui() 
 
        self._y_pred = torch.tensor([]) 
        self.duration = None 
 
        self.show() 
 
    def init_ui(self): 
       
  # create media player object 
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface) 
 
        # create videowidget object 
        video_widget = QVideoWidget() 
 
        # create open button 
        open_btn = QPushButton("Open Video") 
        open_btn.clicked.connect(self.open_file) 
 
        # create button for playing 
        self.playBtn = QPushButton() 
        self.playBtn.setEnabled(False) 
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)) 
        self.playBtn.clicked.connect(self.play_video) 
 
        # create slider 
        self.slider = QSlider(Qt.Horizontal) 
        self.slider.setRange(0, 0) 
        self.slider.sliderMoved.connect(self.set_position) 
 
        # create label 
        self.label = QLabel() 
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum) 
 
        # create grid layout 
        grid_layout = QGridLayout() 
 
        # AD signal 
 
        self.graphWidget = MplCanvas(self, width=5, height=1, dpi=100) 
 
        # Feature extraction progress bar 
        self.pbar = QProgressBar() 
        self.pbar.setTextVisible(True) 
       self.device = get_torch_device() 
 
        self.setWindowTitle(APP_NAME) 
        self.setGeometry(350, 100, 700, 500) 
        self.setWindowIcon(QIcon("player.png")) 
 
        p = self.palette() 
        p.setColor(QPalette.Window, Qt.black) 
        self.setPalette(p) 
 
        self.init_ui() 
 
        self._y_pred = torch.tensor([]) 
        self.duration = None 
 
        self.show() 
 
    def init_ui(self): 
       
  # create media player object 
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface) 
 
        # create videowidget object 
        video_widget = QVideoWidget() 
 
        # create open button 
        open_btn = QPushButton("Open Video") 
        open_btn.clicked.connect(self.open_file) 
 
        # create button for playing 
        self.playBtn = QPushButton() 
        self.playBtn.setEnabled(False) 
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)) 
        self.playBtn.clicked.connect(self.play_video) 
 
        # create slider 
        self.slider = QSlider(Qt.Horizontal) 
        self.slider.setRange(0, 0) 
        self.slider.sliderMoved.connect(self.set_position) 
 
        # create label 
        self.label = QLabel() 
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum) 
 
        # create grid layout 
        grid_layout = QGridLayout() 
 
        # AD signal 
 
        self.graphWidget = MplCanvas(self, width=5, height=1, dpi=100) 
 
        # Feature extraction progress bar 
        self.pbar = QProgressBar() 
        self.pbar.setTextVisible(True) 
       progress_bar=self.pbar, 
                ) 
            ) 
            features = torch.tensor(features) 
 
        elif self.feature_source == "Load features from file": 
            f_filename, _ = QFileDialog.getOpenFileName(self, "Open Features File") 
            features = read_features(file_path=f_filename) 
 
        self._y_pred = ad_prediction( 
            model=anomaly_detector, 
            features=features, 
            device=self.device, 
        ) 
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename))) 
        self.playBtn.setEnabled(True) 
        self.label.setText("Done! Click the Play button") 
   
  def _features_msgbtn(self, i) -> None: 
        self.feature_source = i.text()  # pylint: disable=attribute-defined-outside-init 
 
    def play_video(self): 
         
if self.mediaPlayer.state() == QMediaPlayer.PlayingState: 
            self.mediaPlayer.pause() 
        else: 
            self.mediaPlayer.play() 
 
    # pylint: disable=unused-argument 
    def mediastate_changed(self, state): 
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState: 
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause)) 
 
        else: 
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)) 
 
    def position_changed(self, position): 
        self.slider.setValue(position) 
 
    def duration_changed(self, duration): 
        self.slider.setRange(0, duration) 
        self.duration = duration 
 
        if self._y_pred is not None: 
            self._y_pred = np.repeat(self._y_pred, duration // len(self._y_pred)) 
 
    def set_position(self, position): 
        self.mediaPlayer.setPosition(position) 
 
    def handle_errors(self): 
        self.playBtn.setEnabled(False) 
        self.label.setText("Error: " + self.mediaPlayer.errorString()) 
 
    def plot(self, position): 
        if self._y_pred is not None:
              ax = self.graphWidget.axes 
            ax.clear() 
            ax.set_xlim(0, self.mediaPlayer.duration()) 
            ax.set_ylim(-0.1, 1.1) 
            ax.plot(self._y_pred[:position], "*-", linewidth=7) 
            self.graphWidget.draw() 
 
if __name__ == "__main__": 
    args = get_args() 
 
    anomaly_detector, feature_extractor = load_models( 
        args.feature_extractor, 
        args.ad_model, 
        features_method=args.feature_method, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    ) 
    app = QApplication(sys.argv) 
    window = Window() 
    sys.exit(app.exec_()) 

