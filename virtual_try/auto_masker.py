import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from lang_sam import LangSAM


class AutoInpaintMaskGenerator:
    def __init__(
        self,
        langsam_model: LangSAM | None = None,
    ):
        """
        langsam_model: an instance of LangSAM already loaded
        threshold: mask score threshold for filtering masks
        mask_selection:
            - "best": use the highest-scoring mask only
            - "union": combine all masks passing threshold
        """
        if langsam_model is None:
            sam_path = hf_hub_download(
                repo_id="facebook/sam2.1-hiera-large",
                filename="sam2.1_hiera_large.pt",
            )
            langsam_model = LangSAM(
                "sam2.1_hiera_large",
                sam_path,
            )

        self.model = langsam_model

    def generate_mask(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Generate a binary mask for inpainting.

        Returns:
            A 2D P (dtype=uint8), with 255 for masked regions and 0 elsewhere.
        """
        result = self.model.predict(
            texts_prompt=[prompt],
            images_pil=[image],
        )[0]

        masks = result["masks"]  # (N, H, W)
        scores = np.atleast_1d(result["mask_scores"]) # Ensure it's always at least 1D

        # If only one mask returned, expand dims
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]  # Make it (1, H, W)

        if len(masks) == 0:
            raise ValueError("No masks found.")

        # Filter masks by score threshold
        valid_indices = scores >= threshold
        if len(valid_indices) == 0:
            raise ValueError("No masks scored the required threshold.")
        
        combined_mask = np.any(masks[valid_indices], axis=0)

        # Convert to uint8 binary mask for inpainting
        binary_mask = (combined_mask.astype(np.uint8)) * 255  # 0 or 255

        # Apply dilation, to give more flexibility to the inpainting model
        kernel = np.ones((10, 10), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        return dilated_mask
