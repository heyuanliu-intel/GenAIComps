# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import threading

from diffusers.utils import export_to_video
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import Text2VideoInput, Text2VideoOutput

logger = CustomLogger("opea_Text2Video")

# Global variables for the model pipeline and initialization state
pipe = None
initialization_lock = threading.Lock()
initialized = False


def initialize(
    model_name_or_path: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    device: str = "hpu",
    token: str = None,
    bf16: bool = True,
    use_hpu_graphs: bool = False,
):
    """Initialize the model pipeline in a thread-safe manner."""
    global pipe, initialized
    with initialization_lock:
        if initialized:
            return

        model_name = os.getenv("MODEL", model_name_or_path)
        # hf_token = os.getenv("HF_TOKEN", token)
        kwargs = {}
        if bf16:
            kwargs["torch_dtype"] = torch.bfloat16

        if device == "hpu":
            from optimum.habana.diffusers import GaudiWanPipeline

            pipe = GaudiWanPipeline.from_pretrained(
                model_name,
                use_habana=True,
                use_hpu_graphs=use_hpu_graphs,
                gaudi_config="Habana/stable-diffusion",
                **kwargs,
            )
            logger.info("GaudiWanPipeline loaded.")
        else:
            raise NotImplementedError(f"Device '{device}' is not supported. Only 'hpu' are supported.")

        logger.info(f"Model '{model_name}' initialized on device '{device}'.")
        initialized = True


@OpeaComponentRegistry.register("OPEA_TEXT2VIDEO")
class OpeaText2Video(OpeaComponent):
    """A specialized Text2Video component for video generation."""

    def __init__(
        self,
        name: str,
        description: str,
        config: dict = None,
        seed: int = 42,
        model_name_or_path: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        device: str = "hpu",
        token: str = None,
        bf16: bool = True,
        use_hpu_graphs: bool = False,
        video_dir: str = "/home/user/video",
    ):
        """
        Initializes the OpeaText2Video component.

        Args:
            name (str): The name of the component.
            description (str): A description of the component.
            config (dict, optional): Configuration dictionary. Defaults to None.
            seed (int, optional): Random seed for generation. Defaults to 42.
            model_name_or_path (str, optional): The model identifier. Defaults to "Wan-AI/Wan2.2-TI2V-5B-Diffusers".
            device (str, optional): The device to run the model on. Defaults to "hpu".
            token (str, optional): Hugging Face authentication token. Defaults to None.
            bf16 (bool, optional): Whether to use bfloat16 precision. Defaults to True.
            use_hpu_graphs (bool, optional): Whether to use HPU graphs for optimization. Defaults to False.
        """
        super().__init__(name, ServiceType.TEXT2VIDEO.name.lower(), description, config)
        initialize(
            model_name_or_path=model_name_or_path,
            device=device,
            token=token,
            bf16=bf16,
            use_hpu_graphs=use_hpu_graphs,
        )
        self.pipe = pipe
        self.seed = seed
        self.video_dir = video_dir
        self.generator = torch.manual_seed(self.seed)
        self.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        if not self.check_health():
            logger.error("OpeaText2Video health check failed upon initialization.")

    async def invoke(self, input: Text2VideoInput) -> Text2VideoOutput:
        """
        Generates a video based on the provided text prompt.

        Args:
            input (Text2VideoInput): The input data containing the prompt and other parameters.

        Returns:
            Text2VideoOutput: The generated video as a base64 encoded GIF string.
        """
        job_file = os.path.join(self.video_dir, "job.txt")
        created = time.time()
        id = f"video_{int(created)}"
        status = "queued"
        quality = "standard"
        job = [id, status, int(created), input.prompt, input.input_reference, input.seconds, input.size, quality]
        with open(job_file, "w") as f:
            f.write(",".join(job))
            f.write("\n")

        logger.info(f"Job {id} queued with prompt: {input.prompt}")
        return Text2VideoOutput(id=id, model=os.getenv("MODEL"), status=status, progress=0, created_at=int(created), seconds=str(input.seconds), size=input.size, quality=quality)

    def check_health(self) -> bool:
        """
        Checks if the model pipeline is initialized.

        Returns:
            bool: True if the pipeline is ready, False otherwise.
        """
        if self.pipe is None:
            logger.error("Health check failed: Model pipeline is not initialized.")
            return False
        return True
