from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from sface import SFace
from yunet import YuNet

LAB_DIR = Path(__file__).resolve().parent
FACE_ENROLL_DIR = LAB_DIR / "face_enroll" / "FAR"
MODELS_DIR = LAB_DIR / "modelos"
MODEL_SFACE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
MODEL_YUNET = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scores_for_directory(
    directory: Path,
    reference_embedding: np.ndarray,
    recognizer: SFace,
    detector: YuNet,
) -> tuple[list[float], int]:
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio de imágenes: {directory}")

    image_paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No hay imágenes compatibles en: {directory}")

    scores = []
    skipped = 0
    for image_path in image_paths:
        image = cv.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        faces = detector.infer(image)
        if len(faces) != 1:
            skipped += 1
            continue

        embedding = recognizer.infer(image, faces[0][:-1])
        score, _ = recognizer.match(reference_embedding, embedding)
        scores.append(float(score))

    if not scores:
        raise ValueError(f"No se pudieron evaluar imágenes con exactamente una cara en: {directory}")
    return scores, skipped


def _select_threshold(
    genuine_scores: list[float],
    impostor_scores: list[float],
    target_far: float | None,
) -> tuple[float, float, float]:
    all_scores = np.asarray(genuine_scores + impostor_scores, dtype=np.float64)
    thresholds = np.append(np.unique(all_scores), np.nextafter(all_scores.max(), np.inf))

    metrics = []
    for threshold in thresholds:
        far = float(np.mean(np.asarray(impostor_scores) >= threshold))
        frr = float(np.mean(np.asarray(genuine_scores) < threshold))
        metrics.append((float(threshold), far, frr))

    if target_far is None:
        threshold, far, frr = min(metrics, key=lambda item: (abs(item[1] - item[2]), (item[1] + item[2]) / 2))
    else:
        eligible = [item for item in metrics if item[1] <= target_far]
        if not eligible:
            raise ValueError("No se encontró un umbral que cumpla el FAR solicitado.")
        threshold, far, frr = min(eligible, key=lambda item: (item[2], item[0]))

    return threshold, far, frr


def evaluate_far_frr(
    embedding_path: str | Path,
    real_dir: str | Path = FACE_ENROLL_DIR / "REAL",
    fake_dir: str | Path = FACE_ENROLL_DIR / "FAKE",
    *,
    target_far: float | None = None,
    sface_model: str | Path = MODEL_SFACE,
    yunet_model: str | Path = MODEL_YUNET,
) -> dict[str, Any]:
    """Estima un umbral SFace usando imágenes genuinas y de impostores.

    Si target_far es None, selecciona el punto de operación más cercano al EER.
    Si se indica un valor entre 0 y 1, elige el menor FRR que respete ese FAR.
    """
    if target_far is not None and not 0 <= target_far <= 1:
        raise ValueError("target_far debe estar entre 0 y 1.")

    reference_embedding = np.asarray(np.load(embedding_path, allow_pickle=False), dtype=np.float32)
    if reference_embedding.ndim == 1:
        reference_embedding = reference_embedding.reshape(1, -1)
    if reference_embedding.ndim != 2 or reference_embedding.shape[0] != 1:
        raise ValueError("El embedding debe tener forma (dimensión,) o (1, dimensión).")
    if not np.isfinite(reference_embedding).all():
        raise ValueError("El embedding contiene valores no finitos.")

    recognizer = SFace(modelPath=str(sface_model), disType=0)
    detector = YuNet(modelPath=str(yunet_model))

    genuine_scores, skipped_real = _scores_for_directory(
        Path(real_dir), reference_embedding, recognizer, detector
    )
    impostor_scores, skipped_fake = _scores_for_directory(
        Path(fake_dir), reference_embedding, recognizer, detector
    )
    threshold, far, frr = _select_threshold(genuine_scores, impostor_scores, target_far)

    result = {
        "threshold": threshold,
        "threshold_percent": threshold * 100,
        "far": far,
        "far_percent": far * 100,
        "frr": frr,
        "frr_percent": frr * 100,
        "genuine_samples": len(genuine_scores),
        "impostor_samples": len(impostor_scores),
        "skipped_real": skipped_real,
        "skipped_fake": skipped_fake,
        "target_far": target_far,
    }
    if target_far is None:
        result["eer_percent"] = (far + frr) / 2 * 100
    return result