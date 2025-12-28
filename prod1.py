import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------
# Device & Model Setup
# ---------------------------

def get_device() -> str:
    """Return the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory // 1_000_000_000
            logger.info(f"GPU: {name} | VRAM: {total} GB")
        except Exception:
            pass
    else:
        try:
            import psutil  # optional
            ram_gb = psutil.virtual_memory().total // 1_000_000_000
            logger.info(f"System RAM: {ram_gb} GB")
        except Exception:
            pass
    return device


def load_clip(device: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model & processor on the given device."""
    logger.info("Loading CLIP model: openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


# ---------------------------
# Image & Text Processing
# ---------------------------

def _resize_to_target(image_tensor: torch.Tensor, target_size: Tuple[int, int] = (240, 240)) -> torch.Tensor:
    """Resize to target (H, W) and ensure 3 channels."""
    resized = transforms.Resize(target_size)(image_tensor)
    if resized.shape[0] == 1:
        resized = resized.expand(3, -1, -1)
    return resized


def load_transform_images(image_paths: List[Path]) -> List[torch.Tensor]:
    """Load image files and apply simple resize transform, returning tensors."""
    out: List[torch.Tensor] = []
    for p in image_paths:
        try:
            img = read_image(str(p))
            img = _resize_to_target(img)
            out.append(img)
        except Exception as e:
            logger.warning(f"Failed to read/transform image {p}: {e}")
    return out


def get_text_embedding(text: str, model: CLIPModel, processor: CLIPProcessor, device: str) -> torch.Tensor:
    """Compute a single text embedding (returned on CPU)."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb[0].cpu()


def get_image_embeddings(image_paths: List[Path], model: CLIPModel, processor: CLIPProcessor, device: str) -> torch.Tensor:
    """Compute image embeddings (returned on CPU)."""
    images = load_transform_images(image_paths)
    if not images:
        return torch.empty(0)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return embs.cpu()


# ---------------------------
# Filesystem Helpers
# ---------------------------

def gather_images(input_dirs: List[Path]) -> List[Path]:
    """Collect all .jpg images from given directories."""
    images: List[Path] = []
    for d in input_dirs:
        if not d.exists():
            logger.warning(f"Input dir not found: {d}")
            continue
        images.extend(d.glob("*.jpg"))
    logger.info(f"Found {len(images)} images")
    return images


def chunks(seq: List[Path], size: int) -> List[List[Path]]:
    """Yield successive chunks of size from seq."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ---------------------------
# Embedding Workflows
# ---------------------------

def save_batch_embeddings(images: List[Path], batch_size: int, save_dir: Path, model: CLIPModel, processor: CLIPProcessor, device: str) -> List[Path]:
    """Compute and save image embeddings in batches; return list of .pt files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    total_batches = int(np.ceil(len(images) / max(batch_size, 1))) if images else 0
    logger.info(f"Batching {len(images)} images into {total_batches} batches of size {batch_size}")
    files: List[Path] = []
    for i, batch in enumerate(chunks(images, batch_size), start=1):
        logger.info(f"Processing batch {i}/{total_batches}")
        embs = get_image_embeddings(batch, model, processor, device)
        if embs.numel() == 0:
            logger.warning(f"Batch {i} produced empty embeddings; skipping save.")
            continue
        fp = save_dir / f"batch_{i}.pt"
        torch.save(embs, fp)
        files.append(fp)
    logger.info(f"Saved {len(files)} batch files to {save_dir}")
    return files


def load_all_embeddings(embedding_files: List[Path]) -> torch.Tensor:
    """Load and concatenate all saved embedding tensors."""
    tensors: List[torch.Tensor] = []
    for fp in embedding_files:
        try:
            tensors.append(torch.load(fp))
        except Exception as e:
            logger.warning(f"Failed to load {fp}: {e}")
    if not tensors:
        return torch.empty(0)
    return torch.cat(tensors)


def rank_images_by_text(text_embedding: torch.Tensor, image_embeddings: torch.Tensor) -> List[float]:
    """Compute cosine similarity scores between text and each image embedding."""
    scores: List[float] = []
    cos = torch.nn.CosineSimilarity()
    for img_emb in tqdm(image_embeddings, desc="Scoring", unit="image"):
        score = cos(text_embedding.unsqueeze(0), img_emb.unsqueeze(0))
        scores.append(float(score))
    return scores


# ---------------------------
# CLI
# ---------------------------

def run(
    query_text: Optional[str],
    input_dirs: Optional[List[str]] = None,
    save_dir: str = "data/fashion-images/embeddings",
    batch_size: int = 500,
    top_k: int = 5,
    embed_only: bool = False,
) -> None:
    """Main workflow: embed images, optionally search by text and print top-k results."""
    device = get_device()
    model, processor = load_clip(device)

    # Default input dirs replicate the notebook behavior
    default_dirs = [
        Path("data/Apparel/Boys/Images/images_with_product_ids"),
        Path("data/Apparel/Girls/Images/images_with_product_ids"),
    ]
    dirs = [Path(d) for d in (input_dirs or [])] or default_dirs
    images = gather_images(dirs)

    save_path = Path(save_dir)
    batch_files = save_batch_embeddings(images, batch_size, save_path, model, processor, device)

    if embed_only:
        logger.info("Embedding completed (embed_only=True). Skipping search.")
        return

    if not query_text:
        logger.warning("No query text provided; use --query to search. Exiting.")
        return

    image_embeddings = load_all_embeddings(batch_files)
    if image_embeddings.numel() == 0:
        logger.error("No embeddings available to search.")
        return

    text_emb = get_text_embedding(query_text, model, processor, device)
    scores = rank_images_by_text(text_emb, image_embeddings)

    # Order images by score desc
    idx = np.argsort(scores)[::-1]
    top_idxs = idx[:top_k]
    top_scores = np.array(scores)[top_idxs]

    # Reconstruct the images order matching embeddings
    # We saved embeddings in batch order, matching input 'images'.
    # Therefore, 'images' corresponds to the same order used to compute embeddings.
    top_images = np.array(images)[top_idxs]

    logger.info("Top results:")
    for rank, (img, sc) in enumerate(zip(top_images, top_scores), start=1):
        logger.info(f"#{rank}: {img} | score={sc:.4f}")


def _parse_args(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute CLIP embeddings for images and search by text.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text query to rank images. If omitted, run embed-only unless --embed-only is false.",
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="*",
        default=None,
        help="List of directories containing .jpg images. Defaults to Apparel Boys/Girls paths.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/fashion-images/embeddings",
        help="Directory to save batch embedding .pt files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for image embedding computation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to print when searching.",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Only compute and save embeddings; skip ranking/search.",
    )

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    run(
        query_text=args.query,
        input_dirs=args.input_dirs,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
        embed_only=args.embed_only,
    )
