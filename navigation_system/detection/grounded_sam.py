import attr
import os
import time
from typing import Any, Union, List, Tuple
from abc import ABCMeta, abstractmethod

import cv2
import torch
import numpy as np

try:
    from habitat import Config
except ImportError:  # Habitat 0.2.x no longer exports Config
    from typing import Any as Config

import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

from navigation_system.detection.vendor.repvit_sam.setup_repvit_sam import build_sam_repvit


VisualObservation = Union[torch.Tensor, np.ndarray]


@attr.s(auto_attribs=True)
class Segment(metaclass=ABCMeta):
    config: Config
    device: torch.device
    
    def __attrs_post_init__(self):
        self._create_model(self.config, self.device)
    
    @abstractmethod
    def _create_model(self, config: Config, device: torch.device) -> None:
        pass
    
    @abstractmethod
    def segment(self, image: VisualObservation, **kwargs) -> Any:
        pass
    

@attr.s(auto_attribs=True)
class GroundedSAM(Segment):
    height: float = 480.
    width: float = 640.
    
    def _create_model(self, config: Config, device: torch.device) -> Any:
        detection_model_cfg = config.DETECTION.MODEL
        detection_threshold_cfg = config.DETECTION.THRESHOLDS

        grounding_dino_config_path = detection_model_cfg.GROUNDING_DINO_CONFIG_PATH
        grounding_dino_checkpoint_path = detection_model_cfg.GROUNDING_DINO_CHECKPOINT_PATH
        sam_checkpoint_path = detection_model_cfg.SAM_CHECKPOINT_PATH
        sam_encoder_version = detection_model_cfg.SAM_ENCODER_VERSION
        repvit_sam_checkpoint_path = detection_model_cfg.REPVIT_SAM_CHECKPOINT_PATH

        self.box_threshold = float(detection_threshold_cfg.BOX)
        self.text_threshold = float(detection_threshold_cfg.TEXT)
        self._dino_disabled_reason = ""
        self._dino_disabled_warned = False
        self._dino_runtime_mode = "unknown"

        dino_device = device
        use_cpu_fallback = str(
            os.getenv("SPACEVLN_GROUNDINGDINO_CPU_FALLBACK", "")
        ).strip().lower() in {"1", "true", "yes", "on"}

        try:
            from groundingdino.models.GroundingDINO import ms_deform_attn

            custom_ops_available = bool(
                getattr(ms_deform_attn, "_CUSTOM_OPS_AVAILABLE", False)
            )
        except Exception:
            custom_ops_available = False

        try:
            self.grounding_dino_model = Model(
                model_config_path=grounding_dino_config_path,
                model_checkpoint_path=grounding_dino_checkpoint_path,
                device=dino_device,
            )
            if custom_ops_available:
                self._dino_runtime_mode = (
                    "cuda_custom_ops" if str(dino_device) != "cpu" else "cpu_custom_ops"
                )
            else:
                self._dino_runtime_mode = (
                    "cuda_pytorch_fallback"
                    if str(dino_device) != "cpu"
                    else "cpu_pytorch_fallback"
                )
                print(
                    "[WARN] GroundingDINO custom CUDA op is unavailable; "
                    "using the PyTorch fallback path instead. "
                    "Detection still runs, and if the model is on CUDA it still uses GPU, "
                    "but it will be slower than the custom op."
                )
        except Exception as exc:
            if use_cpu_fallback and str(dino_device) != "cpu":
                dino_device = torch.device("cpu")
                print(
                    "[WARN] GroundingDINO init on CUDA failed; "
                    "retrying on CPU because SPACEVLN_GROUNDINGDINO_CPU_FALLBACK=1. "
                    f"Reason: {type(exc).__name__}: {exc}"
                )
                try:
                    self.grounding_dino_model = Model(
                        model_config_path=grounding_dino_config_path,
                        model_checkpoint_path=grounding_dino_checkpoint_path,
                        device=dino_device,
                    )
                    self._dino_runtime_mode = "cpu_retry_fallback"
                except Exception as cpu_exc:
                    self.grounding_dino_model = None
                    self.sam_predictor = None
                    self._dino_disabled_reason = (
                        "GroundingDINO initialization failed on both CUDA and CPU. "
                        f"CUDA error: {type(exc).__name__}: {exc}. "
                        f"CPU error: {type(cpu_exc).__name__}: {cpu_exc}"
                    )
                    return
            else:
                self.grounding_dino_model = None
                self.sam_predictor = None
                self._dino_disabled_reason = (
                    "GroundingDINO initialization failed and detection was disabled. "
                    "Set SPACEVLN_GROUNDINGDINO_CPU_FALLBACK=1 to retry on CPU. "
                    f"Reason: {type(exc).__name__}: {exc}"
                )
                return

        if detection_model_cfg.USE_REPVIT_SAM:
            sam = build_sam_repvit(checkpoint=repvit_sam_checkpoint_path)
            sam.to(device=device)
        else:
            sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path).to(device=device)
        self.sam_predictor = SamPredictor(sam)
        self.grounding_dino_model.model.eval()
        
    def _segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def _process_detections(self, detections: sv.Detections) -> sv.Detections:
        # 兼容旧版本 supervision：手动计算 box_area
        if hasattr(detections, 'box_area'):
            box_areas = detections.box_area
        else:
            # 手动计算：(x2 - x1) * (y2 - y1)
            box_areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                       (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        
        # supervision<=0.4.0 has no built-in `mask` field on Detections.
        masks = getattr(detections, "mask", None)

        i = len(detections) - 1
        while i >= 0:
            if box_areas[i] / (self.width * self.height) < 0.95:
                i -= 1
                continue
            else:
                detections.xyxy = np.delete(detections.xyxy, i, axis=0)
                if masks is not None:
                    masks = np.delete(masks, i, axis=0)
                if detections.confidence is not None:
                    detections.confidence = np.delete(detections.confidence, i)
                if detections.class_id is not None:
                    detections.class_id = np.delete(detections.class_id, i)
                if detections.tracker_id is not None:
                    detections.tracker_id = np.delete(detections.tracker_id, i)
            i -= 1

        if masks is not None:
            detections.mask = masks
            
        return detections

    def _empty_segment_result(
        self,
        image: VisualObservation,
        reason: str = "",
    ) -> Tuple[np.ndarray, List[str], np.ndarray, sv.Detections]:
        height, width = image.shape[:2]
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )
        detections.mask = np.empty((0, height, width), dtype=np.float32)
        if reason:
            print(f"[WARN] GroundedSAM detection skipped: {reason}")
        return detections.mask, [], image.copy(), detections
    
    @torch.no_grad()
    def segment(self, image: VisualObservation, **kwargs) -> Tuple[np.ndarray, List[str], np.ndarray]:
        if getattr(self, "grounding_dino_model", None) is None:
            reason = ""
            if not bool(getattr(self, "_dino_disabled_warned", False)):
                reason = str(getattr(self, "_dino_disabled_reason", "") or "")
                self._dino_disabled_warned = True
            return self._empty_segment_result(image, reason)

        classes = kwargs.get("classes", [])
        box_threshold = float(kwargs.get("box_threshold", self.box_threshold))
        text_threshold = float(kwargs.get("text_threshold", self.text_threshold))
        box_annotator = sv.BoxAnnotator()
        # 兼容旧版本 supervision（没有 MaskAnnotator）
        try:
            mask_annotator = sv.MaskAnnotator()
        except AttributeError:
            mask_annotator = None
        labels = []
        # t1 = time.time()
        try:
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
        except Exception as exc:
            return self._empty_segment_result(
                image,
                f"{type(exc).__name__}: {exc}",
            )
        # t2 = time.time()
        detections = self._process_detections(detections)

        if len(detections.xyxy) == 0:
            return self._empty_segment_result(image)
        
        # 兼容不同版本的 supervision：使用属性而不是迭代
        for i in range(len(detections.xyxy)):
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            class_id = detections.class_id[i] if detections.class_id is not None else None
            class_name = None
            try:
                if class_id is not None:
                    class_index = int(class_id)
                    if 0 <= class_index < len(classes):
                        class_name = str(classes[class_index])
            except (TypeError, ValueError):
                class_name = None

            if class_name is None and len(classes) == 1:
                class_name = str(classes[0])
            labels.append(f"{class_name or 'unknown'} {confidence:0.2f}")
        # t3 = time.time()
        detections.mask = self._segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        # t4 = time.time()
        # print("grounding dino: ", t2 - t1)
        # print("process detections: ", t3 - t2)
        # print("sam: ", t4 - t3)
        # annotated_image.shape=(h,w,3)
        if mask_annotator is not None:
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        else:
            # 旧版本 supervision：手动绘制 mask
            annotated_image = image.copy()
            if detections.mask is not None:
                for mask in detections.mask:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask_bool = mask.astype(bool)
                    annotated_image[mask_bool] = annotated_image[mask_bool] * 0.5 + color_mask * 0.5
        
        # 兼容不同版本的 BoxAnnotator.annotate() API
        try:
            # 新版本：labels 参数
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        except TypeError:
            # 旧版本：没有 labels 参数，手动绘制文本
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            # 手动添加标签（cv2 已在文件开头导入）
            for i, (xyxy, label) in enumerate(zip(detections.xyxy, labels)):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # detectins.mask.shape=[num_detected_classes, h, w]
        # attention: sometimes the model can't detect all classes, so num_detected_classes <= len(classes)
        return (detections.mask.astype(np.float32), labels, annotated_image, detections)
    

class BatchWrapper:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.
    """
    def __init__(self, model) -> None:
        self.model = model
    
    def __call__(self, images: List[VisualObservation]) -> List:
        return [self.model(image) for image in images]
