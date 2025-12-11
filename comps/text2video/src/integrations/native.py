# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import Text2VideoInput, Text2VideoOutput

logger = CustomLogger("opea_Text2Video")


@OpeaComponentRegistry.register("OPEA_TEXT2VIDEO")
class OpeaText2Video(OpeaComponent):
    """A specialized Text2Video component for video generation."""

    def __init__(
        self,
        name: str,
        description: str,
        config: dict = None,
        video_dir: str = "/home/user/video"
    ):
        """
        Initializes the OpeaText2Video component.

        Args:
            name (str): The name of the component.
            description (str): A description of the component.
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(name, ServiceType.TEXT2VIDEO.name.lower(), description, config)
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        if not self.check_health():
            logger.error("OpeaText2Video health check failed upon initialization.")

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
            input.fps,
            input.shift,
            input.steps,
            input.guide_scale,
            input.audio_guide_scale,
            input.audio_type,
            input.seed
        ]

        if input.input_reference:
            image_file = os.path.join(self.video_dir, f"{job_id}_input_reference")
            contents = await input.input_reference.read()
            with open(image_file, "wb") as img_f:
                img_f.write(contents)
            job.append(image_file)
        else:
            job.append("N/A")

        if input.audio:
            audio_file = os.path.join(self.video_dir, f"{job_id}_audio")
            contents = await input.audio.read()
            with open(audio_file, "wb") as audio_f:
                audio_f.write(contents)
            job.append(audio_file)
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
        return True
