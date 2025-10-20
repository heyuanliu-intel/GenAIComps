# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from diffusers.utils import load_image
from diffusers import AutoPipelineForImage2Image
import torch
import base64
import os
import tempfile
import threading

from typing import List, Union
from fastapi import Form, UploadFile

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType, SDOutputs

logger = CustomLogger("opea_imagetoimage")
logflag = os.getenv("LOGFLAG", False)


pipe = None
args = None
initialization_lock = threading.Lock()
initialized = False


def initialize(
    model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
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
                kwargs(
                    {
                        "use_habana": True,
                        "use_hpu_graphs": use_hpu_graphs,
                        "gaudi_config": "Habana/stable-diffusion",
                        "token": token,
                    }
                )
                if "stable-diffusion-xl" in model_name_or_path:
                    from optimum.habana.diffusers import GaudiStableDiffusionXLImg2ImgPipeline

                    pipe = GaudiStableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_name_or_path,
                        **kwargs,
                    )
                else:
                    raise NotImplementedError(
                        "Only support stable-diffusion-xl now, " + f"model {model_name_or_path} not supported."
                    )
            elif device == "cpu":
                pipe = AutoPipelineForImage2Image.from_pretrained(model_name_or_path, token=token, **kwargs)
            else:
                raise NotImplementedError(f"Only support cpu and hpu device now, device {device} not supported.")
            logger.info("Stable Diffusion model initialized.")
            initialized = True


@OpeaComponentRegistry.register("OPEA_IMAGE2IMAGE")
class OpeaImageToImage(OpeaComponent):
    """A specialized ImageToImage component derived from OpeaComponent for Stable Diffusion model .

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
        model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
        device="cpu",
        token=None,
        bf16=True,
        use_hpu_graphs=False,
    ):
        super().__init__(name, ServiceType.IMAGE2IMAGE.name.lower(), description, config)
        initialize(
            model_name_or_path=model_name_or_path, device=device, token=token, bf16=bf16, use_hpu_graphs=use_hpu_graphs
        )
        self.pipe = pipe
        self.seed = seed
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaImageToImage health check failed.")

    async def invoke(self,
                     image: Union[str, UploadFile, List[UploadFile]],  # accept base64 string or UploadFile
                     mask: Union[UploadFile, List[UploadFile]],
                     prompt: str = Form(None),
                     background: str = "auto",
                     input_fidelity: str = Form(None),
                     model: str = Form(None),
                     n: int = 1,
                     output_compression: int = 100,
                     output_format: str = "png",
                     partial_images: int = 0,
                     quality: str = "auto",
                     response_format: str = "openai",
                     size: str = Form(None),
                     stream: bool = Form(None),
                     user: str = Form(None),
                     seed: str = 0,
                     guidance_scale: str = Form(None),
                     true_cfg_scale: str = Form(None),
                     num_inference_steps: str = Form(None),
                     num_images_per_prompt: str = Form(None),
                     ):
        """Invokes the ImageToImage service to generate Images for the provided input.

        Args:
            input (SDImg2ImgInputs): The input in SD images  format.
        """

        if isinstance(image, list):
            input_images = []
            for img in image:
                image_content = await img.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(image_content)
                    temp_file_path = temp_file.name
                input_images.append(load_image(temp_file_path).convert("RGB"))
                os.unlink(temp_file_path)
                generator = torch.manual_seed(seed)
                images = pipe(image=input_images,
                              prompt=prompt,
                              generator=generator,
                              true_cfg_scale=true_cfg_scale,
                              num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              num_images_per_prompt=num_images_per_prompt).images
        else:
            image_content = base64.b64decode(image) if isinstance(image, str) else await image.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_content)
                temp_file_path = temp_file.name
            image = load_image(temp_file_path).convert("RGB")
            os.unlink(temp_file_path)
            generator = torch.manual_seed(seed)
            images = pipe(image=input_images,
                          prompt=prompt,
                          generator=generator,
                          true_cfg_scale=true_cfg_scale,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          num_images_per_prompt=num_images_per_prompt).images

        image_path = os.path.join(os.getcwd(), prompt.strip().replace(" ", "_").replace("/", ""))
        os.makedirs(image_path, exist_ok=True)
        results = []
        results_openai = []
        for i, image in enumerate(images):
            save_path = os.path.join(image_path, f"image_{i+1}.png")
            image.save(save_path)
            with open(save_path, "rb") as f:
                bytes = f.read()
            b64_str = base64.b64encode(bytes).decode()
            results.append(b64_str)
            results_openai.append({"b64_json": b64_str})

        if response_format == "openai":
            pass
        else:
            return SDOutputs(images=results)

    def check_health(self) -> bool:
        """Checks the health of the ImageToImage service.

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
