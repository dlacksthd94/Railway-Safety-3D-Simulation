import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline, AutoImageProcessor, AutoModel
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Base Class
# ============================================================================

class ObjectDetector(ABC):
    """Abstract base class for all object detectors."""

    def __init__(self, cfg: Any, model_name: str, confidence_threshold: float = 0.5):
        """
        Initialize detector.

        Args:
            cfg: Configuration object
            model_name: Name of the model
            confidence_threshold: Minimum confidence score to keep detections
        """
        self.cfg = cfg
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = self._get_device()
        self.model = None

    @staticmethod
    def _get_device() -> str:
        """Get appropriate device (cuda or cpu)."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform detection on image.

        Args:
            image: Input image as PIL Image

        Returns:
            Dict with keys: 'boxes', 'confidences', 'classes'
            - boxes: List of [x1, y1, x2, y2] in pixel coordinates
            - confidences: List of confidence scores
            - classes: List of class indices or names
        """
        pass

    def standardize_detections(
        self,
        detection_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert detector output to standardized format.

        Args:
            detection_result: Output from detector

        Returns:
            List of detection dicts with keys: class, confidence, x1, y1, x2, y2
        """
        detections = []
        boxes = detection_result.get("boxes", [])
        confidences = detection_result.get("confidences", [])
        classes = detection_result.get("classes", [])

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = box

            if isinstance(cls, (int, np.integer)):
                if hasattr(self, 'text_input') and self.text_input:
                    class_name = self.text_input[cls]
                elif hasattr(self, 'visual_input') and self.visual_input:
                    class_name = self.visual_input.split('.')[0]  # Use filename without extension
                else:
                    class_name = f"class_{cls}"
            else:
                class_name = str(cls)

            detection = {
                "class": class_name,
                "confidence": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
            detections.append(detection)

        return detections


# ============================================================================
# YOLOE Detector
# ============================================================================

class YOLOEDetector(ObjectDetector):
    """Detector for YOLOE segmentation models."""

    SUPPORTED_MODELS = [
        "yoloe-26n-seg",
        "yoloe-26s-seg",
        "yoloe-26m-seg",
        "yoloe-26l-seg",
        "yoloe-26x-seg",
        "yoloe-v8s-seg",
        "yoloe-v8l-seg",
        "yoloe-11s-seg",
        "yoloe-11l-seg",
    ]

    def __init__(
        self,
        
        cfg: Any,
        model_name: str,
        confidence_threshold: float,
        text_input: Optional[List[str]] = None,
        visual_input: Optional[str] = None,
    ):
        """
        Initialize YOLOE detector.

        Args:
            model_name: Model name (e.g., 'yoloe-26n-seg')
            confidence_threshold: Confidence threshold
            text_input: List of class text labels to detect (optional, used for text prompts)
            visual_input: Reference image filename for visual prompts (optional, used for visual prompts)
        """
        super().__init__(cfg, model_name, confidence_threshold)
        self.model_path = model_name + ".pt"
        self.text_input = text_input
        self.visual_input = visual_input
        self.fp_visual_input = None
        self.visual_prompts = None
        self.load_model()

    def load_model(self) -> None:
        """Load YOLOE model."""
        self.model = YOLO(self.model_path)  # type: ignore
        self.model.to(self.device)  # type: ignore
        logger.info(f"Loaded YOLOE model: {self.model_name} on {self.device}")

        # Set user-defined classes from TextClass
        if self.text_input is not None:
            assert isinstance(self.text_input, list) and all(isinstance(cls, str) for cls in self.text_input), "text_input must be a list of strings"
            self.model.set_classes(self.text_input)  # type: ignore
        if self.visual_input is not None:
            assert isinstance(self.visual_input, str), "visual_input must be a string representing the reference image filename for visual prompts"
            fp_visual_input = os.path.join(self.cfg.path.dir_reference_image, self.visual_input) # type: ignore
            self.fp_visual_input = fp_visual_input
            ref_image = cv2.imread(fp_visual_input)
            h, w, c = ref_image.shape
            visual_prompts = dict(
                bboxes=np.array([[0, 0, w, h]]),
                cls=np.array([0]),
            )
            self.visual_prompts = visual_prompts
        logger.info(f"Set custom classes: {self.model.names}")  # type: ignore

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect objects in image using YOLOE.

        Args:
            image: Input image as PIL Image

        Returns:
            Standardized detection dict
        """
        if self.visual_input is not None:
            results = self.model.predict(
                image, 
                conf=self.confidence_threshold, 
                refer_image=self.fp_visual_input, 
                visual_prompts=self.visual_prompts, predictor=YOLOEVPSegPredictor, verbose=False
            )
        else:
            results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        
        result = results[0]

        boxes = []
        confidences = []
        classes = []

        # Extract detections
        if result.boxes is not None:  # type: ignore
            for box in result.boxes:  # type: ignore
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # type: ignore
                conf = box.conf[0].cpu().item()  # type: ignore
                cls_idx = int(box.cls[0].cpu().item())  # type: ignore

                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                confidences.append(float(conf))
                classes.append(cls_idx)

        return {
            "boxes": boxes,
            "confidences": confidences,
            "classes": classes,
        }


# ============================================================================
# Grounding DINO Detector
# ============================================================================

class GroundingDINODetector(ObjectDetector):
    """Detector for Grounding DINO models using HuggingFace transformers."""

    SUPPORTED_MODELS = [
        "IDEA-Research/grounding-dino-tiny",
        "IDEA-Research/grounding-dino-base",
    ]

    def __init__(
        self,
        cfg,
        model_name: str,
        confidence_threshold: float,
        text_input: Optional[List[str]],
    ):
        """
        Initialize Grounding DINO detector using HuggingFace.

        Args:
            model_name: Model ID from HuggingFace (e.g., 'IDEA-Research/grounding-dino-tiny')
            text_prompt: List of class labels for detection
            confidence_threshold: Confidence threshold
            text_input: List of class text labels to detect (used for text prompts)
        """
        super().__init__(cfg, model_name, confidence_threshold)
        self.text_input = text_input
        self.processor = None
        self.load_model()

    def load_model(self) -> None:
        """Load Grounding DINO model from HuggingFace."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
            
        logger.info(f"Loaded Grounding DINO model: {self.model_name} on {self.device}")
        
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect objects in image using Grounding DINO.

        Args:
            image: Input image as PIL Image
        Returns:
            Standardized detection dict
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=self.text_input,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence_threshold,
            text_threshold=0.5,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]
        
        # Extract boxes, confidences, and classes
        result_boxes = []
        confidences = []
        classes = []

        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            # Box is already in [x1, y1, x2, y2] pixel coordinates
            x1, y1, x2, y2 = box.tolist()
            conf = score.item()
            
            result_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            confidences.append(float(conf))
            classes.append(label)

        return {
            "boxes": result_boxes,
            "confidences": confidences,
            "classes": classes,
        }


# ============================================================================
# Feature Matching Detector
# ============================================================================

class FeatureMatchingDetector(ObjectDetector):
    """Detector using SAM segmentation + CLIP embeddings for feature matching."""

    SUPPORTED_MODELS = [
        "facebook/sam-vit-base",
        "facebook/sam-vit-large",
        "facebook/sam-vit-huge",
        "facebook/sam3",
    ]

    def __init__(
        self,
        cfg,
        model_name: str,
        confidence_threshold: float,
        text_input: Optional[List[str]],
        visual_input: Optional[str],
    ):
        """
        Initialize Feature Matching Detector.

        Args:
            cfg: Configuration object
            model_name: Model name (ignored, uses SAM + CLIP internally)
            confidence_threshold: Similarity threshold for feature matching
            text_input: List of class text labels to detect (not used for feature matching)
            visual_input: Reference image
        """
        super().__init__(cfg, model_name, confidence_threshold)
        
        self.text_input = text_input
        self.visual_input = visual_input
        self.visual_input_embedding = None
        self.fp_visual_input = None
        
        self.sam_model_name = model_name
        self.sam_processor = None
        self.sam_model = None
        self.embedding_model_name = 'facebook/dinov2-giant' # facebook/dinov2-large, facebook/dinov2-base, facebook/dinov2-small
        self.embedding_processor = None
        self.embedding_model = None
        
        self.load_model()

    def load_model(self) -> None:
        """Load SAM and embedding models."""
        if 'sam3' in self.sam_model_name:
            raise NotImplementedError("SAM3 model loading not implemented yet. Please implement loading logic for SAM3 or use a supported SAM model.")
        elif 'sam' in self.sam_model_name:
            from transformers import pipeline
            self.sam_model = pipeline(
                "mask-generation",
                model=self.sam_model_name,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported SAM model: {self.sam_model_name}. Supported models: {self.SUPPORTED_MODELS}")
            
        logger.info(f"Loaded SAM model: {self.sam_model_name} on {self.device}")

        # Load DINOv2 for embeddings
        self.embedding_processor = AutoImageProcessor.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name, device_map='auto')
        _ = self.embedding_model.eval()
        logger.info(f"Loaded embedding model: {self.embedding_model_name} on {self.device}")

        # Load and precompute reference image embedding
        if self.visual_input:
            ref_image_path = os.path.join(self.cfg.path.dir_reference_image, self.visual_input)
            self.fp_visual_input = ref_image_path
            ref_image = Image.open(ref_image_path)
            self.visual_input_embedding = self._get_image_embedding(ref_image)

    def _segment_objects_with_sam(self, image: Image.Image, points_per_batch: int, dp_sam_checkpoint: str) -> List[torch.Tensor]:
        """
        Segment all objects in image using SAM.

        Args:
            image: PIL Image

        Returns:
            List of masks
        """
        fp_masks = os.path.join(dp_sam_checkpoint, *image.filename.split('/')[3:]).replace('.jpg', '_masks.pt') # type: ignore
        if os.path.exists(fp_masks):
            masks = torch.load(fp_masks)
        else:
            if 'sam3' in self.sam_model_name:
                raise NotImplementedError("SAM3 inference not implemented yet. Please implement inference logic for SAM3 or use a supported SAM model.")
            elif 'sam' in self.sam_model_name:
                outputs = self.sam_model(image, points_per_batch=points_per_batch) # type: ignore
                masks = outputs["masks"]
            else:
                raise ValueError(f"Unsupported SAM model: {self.sam_model_name}. Supported models: {self.SUPPORTED_MODELS}")
            os.makedirs(os.path.dirname(fp_masks), exist_ok=True)
            torch.save(masks, fp_masks)
            
            visualizer = DetectionVisualizer(font_size=12, line_thickness=2, visual_input=self.fp_visual_input) # Visualize masks to check segmentation quality
            image_mask = visualizer.draw_masks(image, masks, alpha=1)
            fp_image_mask = os.path.join(dp_sam_checkpoint, *image.filename.split('/')[3:]) # type: ignore
            os.makedirs(os.path.dirname(fp_image_mask), exist_ok=True)
            cv2.imwrite(fp_image_mask, image_mask)

        return masks
    
    def _convert_masks_to_bboxes(self, masks: List[torch.Tensor], padding: float) -> List[List[int]]:
        """
        Convert SAM masks to bounding boxes.

        Args:
            masks: List of binary masks (H, W)
            padding: Padding ratio to add around the bounding box (e.g., 0.1 for 10% padding)

        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        bboxes = []
        for mask in masks:
            mask_array = mask.cpu().numpy()
            y_indices, x_indices = np.where(mask_array > 0)
            
            if len(x_indices) > 0 and len(y_indices) > 0:
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                width, height = x2 - x1, y2 - y1
                # Apply padding
                x1 = int(max(0, x1 - int(padding * width)))
                y1 = int(max(0, y1 - int(padding * height)))
                x2 = int(min(mask_array.shape[1], x2 + int(padding * width)))
                y2 = int(min(mask_array.shape[0], y2 + int(padding * height)))
                bboxes.append([x1, y1, x2, y2])
        assert len(bboxes) == len(masks), "Number of bounding boxes should match number of masks"

        return bboxes
    
    def _filter_masks_and_bboxes(self, masks: List[torch.Tensor], bboxes: List[List[int]], image: Image.Image) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Filter bounding boxes based on size and aspect ratio.

        Args:
            masks: List of binary masks (H, W)
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            image: PIL Image for reference
        
        Returns:
            Tuple of filtered masks and bounding boxes
        """
        filtered_masks = []
        filtered_bboxes = []
        img_width, img_height = image.size
        for mask, bbox in zip(masks, bboxes):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # Filter bboxes that has either width or height more than 90% of the image size
            if width >= 0.9 * img_width or height >= 0.9 * img_height:
                continue
            filtered_masks.append(mask)
            filtered_bboxes.append(bbox)

        return filtered_masks, filtered_bboxes
        
    def _crop_bboxes(self, image: Image.Image, bboxes: List[List[int]]) -> List[Image.Image]:
        """
        Crop image using bounding boxes.

        Args:
            image: PIL Image
            bboxes: List of bounding boxes [x1, y1, x2, y2] in pixel coordinates

        Returns:
            List of cropped PIL Images
        """
        cropped_images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            cropped_images.append(image.crop((x1, y1, x2, y2)))

        return cropped_images

    def _get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Get image embedding using CLIP vision encoder.

        Args:
            image: PIL Image

        Returns:
            Image embedding (feature vector)
        """
        # Prepare image using processor
        inputs = self.embedding_processor(images=image, return_tensors="pt").to(self.device) # type: ignore

        with torch.no_grad():
            outputs = self.embedding_model(**inputs) # type: ignore
            image_embedding = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
            
            # Normalize embedding
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)

        return image_embedding.cpu()
    
    def _get_image_embeddings(self, images: List[Image.Image], image: Image.Image, dp_sam_checkpoint: str) -> torch.Tensor:
        """
        Get image embeddings for a list of images.

        Args:
            images: List of PIL Images
        Returns:
            Image embeddings (feature vectors) as torch tensor of shape (N, D)
        """
        fp_crop_embeddings = os.path.join(dp_sam_checkpoint, *image.filename.split('/')[3:]).replace('.jpg', '_crop_embeddings.pt') # type: ignore
        if os.path.exists(fp_crop_embeddings):
            crop_embeddings = torch.load(fp_crop_embeddings)
        else:
            embeddings = []
            for image in images:
                embedding = self._get_image_embedding(image)
                embeddings.append(embedding)
            if len(embeddings) > 0:
                crop_embeddings = torch.vstack(embeddings)
            else:
                crop_embeddings = torch.empty((0, self.visual_input_embedding.shape[1])) # type: ignore

            os.makedirs(os.path.dirname(fp_crop_embeddings), exist_ok=True)
            torch.save(crop_embeddings, fp_crop_embeddings)
            
        return crop_embeddings
    
    def _get_cosine_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: Embedding vector (1, D)
            embedding2: Embedding vector (1, D)

        Returns:
            Cosine similarity score (0-1)
        """
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        return float(similarity)

    def detect(self, image: np.ndarray|Image.Image) -> Dict[str, Any]:
        """
        Detect matching objects using SAM segmentation and feature matching.

        Args:
            image: PIL Image or numpy array

        Returns:
            Standardized detection dict
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
        
        if self.visual_input_embedding is None:
            raise RuntimeError("Reference image embedding not loaded. Check visual_input.")

        dp_sam_checkpoint = os.path.join(self.cfg.path.dir_image_preprocessed, '.checkpoint', self.sam_model_name)
        
        masks = self._segment_objects_with_sam(image, points_per_batch=64, dp_sam_checkpoint=dp_sam_checkpoint)
        
        bboxes = self._convert_masks_to_bboxes(masks, padding=0.1)
        masks, bboxes = self._filter_masks_and_bboxes(masks, bboxes, image)
        assert len(masks) == len(bboxes), "Number of masks and bounding boxes should match after filtering"
    
        cropped_images = self._crop_bboxes(image, bboxes)
        crop_embeddings = self._get_image_embeddings(cropped_images, image, dp_sam_checkpoint=dp_sam_checkpoint)
        assert len(crop_embeddings) == len(bboxes), "Number of crop embeddings should match number of bounding boxes"

        result_masks = []
        boxes = []
        confidences = []
        classes = []
        for mask, bbox, crop_embedding in zip(masks, bboxes, crop_embeddings):
            similarity = self._get_cosine_similarity(self.visual_input_embedding, crop_embedding)

            if similarity >= self.confidence_threshold:
                result_masks.append(mask)
                boxes.append(bbox)
                confidences.append(similarity)
                classes.append(0)  # All matches are same class (target object)

        return {
            "masks": result_masks,
            "boxes": boxes,
            "confidences": confidences,
            "classes": classes,
        }


# ============================================================================
# Detection Visualizer
# ============================================================================

class DetectionVisualizer:
    """Draws bounding boxes on images with class-specific colors."""

    def __init__(self, font_size: int, line_thickness: int, text_input: Optional[List[str]] = None, visual_input: Optional[str] = None):
        """
        Initialize visualizer.

        Args:
            font_size: Font size for class labels
            line_thickness: Thickness of bounding box lines
        """
        self.font_size = font_size
        self.line_thickness = line_thickness
        self.text_input = text_input
        self.visual_input = visual_input

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a given class name."""
        if self.text_input:
            idx = self.text_input.index(class_name)
            cmap = cm.get_cmap("tab20", len(self.text_input))
            rgba = cmap(idx)[:3]
            return tuple(int(c * 255) for c in rgba)  # type: ignore
        elif self.visual_input:
            # For visual prompts, use a fixed color
            return (0, 255, 0)  # Green for visual prompt detections
        else:
            # Default color
            return (0, 255, 0)
    
    def draw_detections(
        self,
        image: np.ndarray|Image.Image,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (H, W, C) in BGR format or PIL Image
            detections: List of detection dicts

        Returns:
            Image with drawn bounding boxes
        """
        if isinstance(image, Image.Image):
            output = np.array(image.convert("RGB"))[:, :, ::-1].copy()  # Convert PIL to BGR
        else:
            output = image.copy()

        for detection in detections:
            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])
            class_name = detection["class"]
            confidence = detection["confidence"]

            # Get color for this class
            color = self._get_color(class_name)

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.line_thickness)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            label_y = max(y1 - 5, label_size[1] + 5)
            cv2.rectangle(
                output,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + baseline),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                output,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output

    def draw_masks(self, image: np.ndarray|Image.Image, masks: List[torch.Tensor], alpha: float) -> np.ndarray:
        """
        Draw segmentation masks on image.

        Args:
            image: Input image (H, W, C) in BGR format or PIL Image
            masks: List of binary masks (H, W)

        Returns:
            Image with drawn masks
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))[:, :, ::-1].copy()  # Convert PIL to BGR

        # output = np.zeros_like(image)
        output = image.copy()
        for mask in masks:
            color = np.random.randint(0, 255, (3,)).tolist()
            mask_array = mask.cpu().numpy()
            output[mask_array > 0] = ((1 - alpha) * image[mask_array > 0] + alpha * np.array(color)).astype(np.uint8)

        return output

# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def create_detector(
    cfg: Any,
    model_name: str,
    confidence_threshold: float,
    text_input: Optional[List[str]] = None,
    visual_input: Optional[str] = None,
) -> ObjectDetector:
    """
    Factory function to create appropriate detector.

    Args:
        model_name: Name of the model
        confidence_threshold: Confidence threshold
        text_input: List of class text labels to detect (optional)
        visual_input: Reference image for visual prompts (optional)

    Returns:
        Initialized ObjectDetector instance
    """
    if "dino" in model_name.lower():
        assert visual_input is None, "Grounding DINO does not support additional visual prompts. Please set visual_input to None."
        return GroundingDINODetector(
            cfg=cfg,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            text_input=text_input,
        )
    elif "yoloe" in model_name.lower():
        return YOLOEDetector(
            cfg=cfg,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            text_input=text_input,
            visual_input=visual_input,
        )
    elif "sam" in model_name.lower():
        return FeatureMatchingDetector(
            cfg=cfg,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            text_input=text_input,
            visual_input=visual_input, # type: ignore
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def preprocess_image(
    cfg,
    model_name: str,
    confidence_threshold: float,
    text_input: Optional[List[str]] = None,
    visual_input: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main preprocessing pipeline for object detection.

    Args:
        cfg: Configuration object with path information
        model_name: Name of detection model
        confidence_threshold: Minimum confidence to keep detections
        text_input: List of class text labels to detect (optional, used for text prompts)
        visual_input: Reference image filename for visual prompts (optional, used for visual prompts)

    Returns:
        DataFrame with detection results
    """
    # Initialize detector
    detector = create_detector(
        cfg=cfg,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        text_input=text_input,
        visual_input=visual_input,
    )
    visualizer = DetectionVisualizer(font_size=12, line_thickness=2, text_input=text_input, visual_input=visual_input)

    # Load image sequence dataframe
    df_image_seq = pd.read_csv(cfg.path.df_image_seq)

    all_detections = []
    image_stats = {
        "total_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {},
    }

    # Process each image
    df_image_seq = df_image_seq.sort_values(by=['crossing_id', 'seq_id', 'img_pos'])
    df_image_seq = df_image_seq.iloc[:100]
    df_image_seq = df_image_seq[~df_image_seq[['crossing_id', 'seq_id']].duplicated()]
    for _, row in tqdm(df_image_seq.iterrows(), total=len(df_image_seq)):
        crossing_id = row['crossing_id']
        seq_id = row['seq_id']
        img_pos = row['img_pos']
        img_id = row['img_id']
        image_rel_path = os.path.join(crossing_id, seq_id, f"{str(img_pos).zfill(4)}_{img_id}.jpg")
        image_path = os.path.join(cfg.path.dir_image_seq, image_rel_path)

        image_stats["total_images"] += 1

        image = Image.open(image_path)

        # Run detection with visual class if specified
        detection_result = detector.detect(image)
        
        # Normalize detections
        detections = detector.standardize_detections(detection_result)

        if detections:
            image_stats["images_with_detections"] += 1
            image_stats["total_detections"] += len(detections)

            # Update class statistics
            for detection in detections:
                class_name = detection["class"]
                image_stats["detections_by_class"][class_name] = image_stats["detections_by_class"].get(class_name, 0) + 1

            # Save annotated image (bboxes)
            image_bbox = visualizer.draw_detections(image, detections)
            fp_image_bbox = os.path.join(cfg.path.dir_image_preprocessed, model_name, image_rel_path)
            os.makedirs(os.path.dirname(fp_image_bbox), exist_ok=True)
            cv2.imwrite(fp_image_bbox, image_bbox)

            # Save masks on image if available (for SAM-based detections)
            if "masks" in detection_result:
                image_mask = visualizer.draw_masks(image, detection_result["masks"], alpha=1)
                fp_image_mask = os.path.join(cfg.path.dir_image_preprocessed, model_name, image_rel_path.replace('.jpg', '_mask.jpg'))
                os.makedirs(os.path.dirname(fp_image_mask), exist_ok=True)
                cv2.imwrite(fp_image_mask, image_mask)

            # Add to results
            for detection in detections:
                result_row = {
                    "crossing_id": crossing_id,
                    "seq_id": seq_id,
                    "img_pos": img_pos,
                    "img_id": img_id,
                    **detection,
                }
                all_detections.append(result_row)

    # Create output dataframe
    df_preprocessed = pd.DataFrame(all_detections)

    # Save dataframe
    output_csv_path = os.path.join(cfg.path.dir_image_preprocessed, model_name, cfg.path.df_image_preprocessed.split('/')[-1])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_preprocessed.to_csv(output_csv_path, index=False)

    # Print summary statistics
    _print_summary_stats(image_stats, len(df_image_seq))

    return df_preprocessed


def _print_summary_stats(image_stats: Dict[str, Any], total_rows: int) -> None:
    """Print summary statistics of preprocessing."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total rows in df_image_seq: {total_rows}")
    print(f"Successfully processed: {image_stats['total_images']}")
    print(f"Images with detections: {image_stats['images_with_detections']}")
    print(f"Total detections: {image_stats['total_detections']}")

    if image_stats["total_images"] > 0:
        print(
            f"Average detections per image: "
            f"{image_stats['total_detections'] / image_stats['total_images']:.2f}"
        )

    if image_stats["detections_by_class"]:
        print("\nDetections by class:")
        for class_name, count in sorted(
            image_stats["detections_by_class"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {class_name}: {count}")

    print("=" * 60 + "\n")


# ============================================================================
# Utility Functions
# ============================================================================

def get_supported_models() -> Dict[str, List[str]]:
    """Get list of supported models by type."""
    return {
        "yoloe": YOLOEDetector.SUPPORTED_MODELS,
        "grounding_dino": GroundingDINODetector.SUPPORTED_MODELS,
        "feature_matching": FeatureMatchingDetector.SUPPORTED_MODELS,
    }


if __name__ == "__main__":
    # Example usage
    preprocess_image(cfg, model_name='facebook/sam3', confidence_threshold=0.5, visual_input='crossbuck_4.jpg') # cfg is config file the contains input/output dir path
