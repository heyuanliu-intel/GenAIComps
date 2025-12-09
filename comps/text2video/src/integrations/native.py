# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import threading

from diffusers import ModularPipeline
from diffusers.utils import export_to_video
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import Text2VideoInput, Text2VideoOutput

logger = CustomLogger("opea_Text2Video")

# Global variables for the model pipeline and initialization state
pipe = None
pipe_image = None
image_processor = None
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
    global pipe, pipe_image, image_processor, initialized
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
            from optimum.habana.diffusers import GaudiWanImageToVideoPipeline

            pipe = GaudiWanPipeline.from_pretrained(
                model_name,
                use_habana=True,
                use_hpu_graphs=use_hpu_graphs,
                gaudi_config="Habana/stable-diffusion",
                **kwargs,
            )
            logger.info(f"GaudiWanPipeline with {model_name} loaded.")

            pipe_image = GaudiWanImageToVideoPipeline.from_pretrained(
                model_name,
                use_habana=True,
                use_hpu_graphs=use_hpu_graphs,
                gaudi_config="Habana/stable-diffusion",
                **kwargs,
            )
            logger.info(f"GaudiWanImageToVideoPipeline with {model_name} loaded.")

            image_processor_model = os.getenv("IMAGE_PROCESSOR")
            image_processor = ModularPipeline.from_pretrained(image_processor_model, trust_remote_code=True)
            logger.info(f"ModularPipeline with {image_processor_model} loaded.")
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
        self.pipe_image = pipe_image
        self.image_processor = image_processor
        self.seed = seed
        self.video_dir = video_dir
        self.generator = torch.manual_seed(self.seed)
        self.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        if not self.check_health():
            logger.error("OpeaText2Video health check failed upon initialization.")

        # Ensure video directory exists
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir, exist_ok=True)
        # Start background worker thread to process queued jobs
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._job_worker, name="text2video-job-worker", daemon=True)
        self._worker_thread.start()

    async def invoke(self, input: Text2VideoInput) -> Text2VideoOutput:
        """
        Generates a video based on the provided text prompt.

        Args:
            input (Text2VideoInput): The input data containing the prompt and other parameters.
        """
        job_file = os.path.join(self.video_dir, "job.txt")
        created = time.time()
        job_id = f"video_{int(created)}"
        status = "queued"
        quality = "standard"
        job = [
            job_id,
            status,
            int(created),
            input.prompt,
            input.seconds,
            input.size,
            quality,
        ]

        if input.input_reference:
            image_file = os.path.join(self.video_dir, f"{job_id}_input_reference")
            contents = await input.input_reference.read()
            with open(image_file, "wb") as img_f:
                img_f.write(contents)
            job.append(image_file)
        else:
            job.append("N/A")

            # Append the new job to the job file
        with open(job_file, "a") as f:
            f.write(os.getenv("SEP").join(map(str, job)))
            f.write("\n")

        logger.info(f"Job {job_id} queued with prompt: {input.prompt}")
        return Text2VideoOutput(
            id=job_id,
            model=os.getenv("MODEL"),
            status=status,
            progress=0,
            created_at=int(created),
            seconds=str(input.seconds),
            size=input.size,
            quality=quality,
        )

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

    def export_to_video(self, id, prompt, seconds, size, input_reference):
        """Exports a sequence of frames to a video file."""
        guidance_scale = self.config.get("guidance_scale", 5.0)
        num_inference_steps = self.config.get("num_inference_steps", 25)
        fps = int(self.config.get("fps", 16))
        num_frames = int(seconds) * fps
        width, height = size.split("x")

        if input_reference != "N/A":
            logger.info(f"Processing input reference image for image {input_reference}.")
            image = self.image_processor(
                image=input_reference,
                max_area=int(height)*int(width),
                output="processed_image"
            )
            output = self.pipe_image(
                image=image,
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                width=int(width),
                height=int(height),
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).frames[0]
        else:
            output = self.pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                generator=self.generator,
                width=int(width),
                height=int(height),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
            ).frames[0]

        export_to_video(output, os.path.join(self.video_dir, f"{id}.mp4"), fps=fps)
        logger.info(f"Exported video for job {id} to {self.video_dir}/{id}.mp4")

    def _job_worker(self):
        """Background worker to poll job.txt and process queued jobs."""
        job_file = os.path.join(self.video_dir, "job.txt")
        while not getattr(self, "_stop_event", threading.Event()).is_set():
            try:
                if not os.path.exists(job_file):
                    time.sleep(1.0)
                    continue

                # Read all jobs
                with open(job_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]

                updated_lines = []
                sep = os.getenv("SEP")
                for line in lines:
                    parts = line.split(sep)
                    if len(parts) < 8:
                        # Malformed line, keep as is
                        updated_lines.append(line)
                        continue

                    id, status, created_str, prompt, seconds, size, quality, input_reference = parts[:8]
                    if status == "queued":
                        self.export_to_video(id, prompt, seconds, size, input_reference)
                        status = "completed"
                        updated_job = [id, status, created_str, prompt, seconds, size, quality, input_reference]
                        updated_lines.append(sep.join(map(str, updated_job)))
                    else:
                        updated_lines.append(line)

                # Persist updated job list only if any queued processed
                if updated_lines != lines:
                    with open(job_file, "w") as f:
                        for l in updated_lines:
                            f.write(f"{l}\n")

            except Exception as e:
                logger.error(f"Job worker encountered an error: {e}")

            # Poll interval
            time.sleep(1.0)
