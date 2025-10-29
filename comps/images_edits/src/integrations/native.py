# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from diffusers.utils import load_image
from diffusers import AutoPipelineForImage2Image
import torch
import base64
from io import BytesIO
from PIL import Image
import os
import tempfile
import threading
import time
import re

from typing import List, Union, Optional
from fastapi import Form, File, UploadFile, Depends

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType, SDOutputs
from comps.cores.proto.api_protocol import (
    ImagesEditsInput,
)


logger = CustomLogger("opea_ImagesEdits")
logflag = os.getenv("LOGFLAG", False)


pipe = None
args = None
initialization_lock = threading.Lock()
initialized = False


def initialize(
    model_name_or_path="Qwen/Qwen-Image-Edit-2509",
    device="cpu",
    token=None,
    bf16=True,
    use_hpu_graphs=False,
):
    global pipe, args, initialized
    with initialization_lock:
        if not initialized:
            # initialize model and tokenizer
            if os.getenv("MODEL", None):
                model_name_or_path = os.getenv("MODEL")
            kwargs = {}
            if bf16:
                kwargs["torch_dtype"] = torch.bfloat16
            if not token:
                token = os.getenv("HF_TOKEN")
            if device == "hpu":
                from optimum.habana.transformers.gaudi_configuration import GaudiConfig
                from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

                gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
                gaudi_config_kwargs["use_torch_autocast"] = True
                gaudi_config = GaudiConfig(**gaudi_config_kwargs)
                kwargs["use_habana"] = True
                kwargs["use_hpu_graphs"] = use_hpu_graphs
                kwargs["gaudi_config"] = gaudi_config
                kwargs["token"] = token
                adapt_transformers_to_gaudi()

                if "stable-diffusion-xl" in model_name_or_path:
                    from optimum.habana.diffusers import GaudiStableDiffusionXLImg2ImgPipeline

                    pipe = GaudiStableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_name_or_path,
                        **kwargs,
                    )
                    logger.info("GaudiStableDiffusionXLImg2ImgPipeline loaded.")
                elif "Qwen-Image-Edit-2509" in model_name_or_path:
                    from optimum.habana.diffusers import GaudiQwenImageEditPlusPipeline

                    pipe = GaudiQwenImageEditPlusPipeline.from_pretrained(
                        model_name_or_path,
                        **kwargs,
                    )
                    logger.info(f"GaudiQwenImageEditPlusPipeline loaded. use_hpu_graphs:{use_hpu_graphs}")
                else:
                    raise NotImplementedError(
                        "Only support stable-diffusion-xl now, " + f"model {model_name_or_path} not supported."
                    )
            elif device == "cpu":
                pipe = AutoPipelineForImage2Image.from_pretrained(model_name_or_path, token=token, **kwargs)
                logger.info("AutoPipelineForImage2Image loaded.")
            else:
                raise NotImplementedError(f"Only support cpu and hpu device now, device {device} not supported.")
            logger.info(f"device:{device} {model_name_or_path} model initialized.")
            initialized = True


@OpeaComponentRegistry.register("OPEA_IMAGES_EDITS")
class OpeaImagesEdits(OpeaComponent):
    """A specialized ImagesEdits component derived from OpeaComponent for Stable Diffusion model .

    Attributes:
        model_name_or_path (str): The name of the Stable Diffusion model used.
        device (str): which device to use.
        token(str): Huggingface Token.
        bf16(bool): Is use bf16.
        use_hpu_graphs(bool): Is use hpu_graphs.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: dict = None,
        seed=42,
        model_name_or_path="Qwen/Qwen-Image-Edit-2509",
        device="cpu",
        token=None,
        bf16=True,
        use_hpu_graphs=False,
    ):
        super().__init__(name, ServiceType.IMAGES_EDITS.name.lower(), description, config)
        initialize(
            model_name_or_path=model_name_or_path, device=device, token=token, bf16=bf16, use_hpu_graphs=use_hpu_graphs
        )
        self.pipe = pipe
        self.seed = seed
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaImagesEdits health check failed.")

    async def invoke(self, input: ImagesEditsInput) -> SDOutputs:
        """Invokes the ImagesEdits service to generate Images for the provided input.

        Args:
            input (ImagesEditsInput): The input in images edits  format.
        """

        start = time.time()
        logger.info(f"Loaded {input.image} images from input.")
        logger.info(f"prompt: {input.prompt}.")
        if input.image and isinstance(input.image, list):
            image = []
            for img in input.image:
                logger.info(f"Loading image from path: {img.filename}")
                contents =  await img.read()
                image_open = Image.open(BytesIO(contents))
                image.append(image_open)
            logger.info(f"Loaded {len(image)} images from input.")

        generator = torch.manual_seed(42)
        prompt = input.prompt
        #guidance_scale = input.guidance_scale
        #true_cfg_scale = input.cfg
        #num_inference_steps = input.num_inference_steps
        guidance_scale = 4
        true_cfg_scale = 4
        num_inference_steps = 20
        results_openai = []
        for i in range(input.n):
            images = pipe(image=image,
                        prompt=prompt,
                        generator=generator,
                        true_cfg_scale=true_cfg_scale,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        ).images

            image_path = os.path.join(os.getcwd(), prompt.strip().replace(" ", "_").replace("/", ""))
            os.makedirs(image_path, exist_ok=True)
            
            for i, image in enumerate(images):
                save_path = os.path.join(image_path, f"image_{i+1}.png")
                image.save(save_path)
                with open(save_path, "rb") as f:
                    bytes = f.read()
                b64_str = base64.b64encode(bytes).decode()
                results_openai.append({"b64_json": b64_str})

        return SDOutputs(background="opaque", created=int(time.time()), data=results_openai, output_format="jpeg", quality="high", size="0x0", usage={"input_tokens": 0, "output_tokens": 0, " total_tokens": 0, "input_tokens_details": {"text_tokens": 0, "image_tokens": 0}},)

    def check_health(self) -> bool:
        """Checks the health of the ImagesEdits service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            if self.pipe:
                return True
            else:
                return False
        except Exception as e:
            # Handle connection errors, timeouts, etc.
            logger.error(f"Health check failed: {e}")
        return False
