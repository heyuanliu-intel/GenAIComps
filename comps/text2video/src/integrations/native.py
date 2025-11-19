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
        hf_token = os.getenv("HF_TOKEN", token)
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


@OpeaComponentRegistry.register("OPEA_TEXT_TO_VIDEO")
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
        self.generator = torch.manual_seed(self.seed)

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
        start_time = time.time()

        # Extract parameters from input and config
        prompt = input.prompt
        width, height = input.size.split("x")
        guidance_scale = self.config.get("guidance_scale", 5.0)
        num_inference_steps = self.config.get("num_inference_steps", 25)
        fps = 16
        num_frames = input.seconds * fps

        logger.info(f"Generating video for prompt: '{prompt}'")
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=self.generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
        ).frames[0]

        id = "video_123"
        export_to_video(output, f"{id}.mp4", fps=fps)

        latency = time.time() - start_time
        logger.info(f"Video generation completed in {latency:.2f} seconds.")

        return Text2VideoOutput(id=id, seconds=input.seconds, size=input.size)

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
