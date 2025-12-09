# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

from fastapi import status
from fastapi.responses import FileResponse, JSONResponse

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import Text2VideoInput, Text2VideoOutput
from comps.text2video.src.integrations.native import OpeaText2Video


# Initialize logger and component loader
logger = CustomLogger("text2video")
component_loader = None
LOGFLAG = os.getenv("LOGFLAG", "False").lower() in ("true", "1", "t")


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/models",
    host="0.0.0.0",
    port=9396,
    methods=["GET"],
)
async def models():
    """Get the available model information."""
    model_name_or_path = os.getenv("MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    if LOGFLAG:
        logger.info(f"Get model: {model_name_or_path}")

    model_info = {
        "object": "list",
        "data": [
            {
                "id": model_name_or_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "opea",
                "root": model_name_or_path,
                "parent": None,
                "max_model_len": None,
                "permission": [],
            }
        ],
    }
    return model_info


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos",
    host="0.0.0.0",
    port=9396,
    input_datatype=Text2VideoInput,
    output_datatype=Text2VideoOutput,
)
@register_statistics(names=["opea_service@text2video"])
async def text2video(input_data: Text2VideoInput) -> Text2VideoOutput:
    """
    Process a text-to-video generation request.

    Args:
        input_data (Text2VideoInput): The input data containing the prompt.

    Returns:
        Text2VideoOutput: The result of the video generation.
    """
    start = time.time()
    if component_loader:
        results = await component_loader.invoke(input_data)
    else:
        raise RuntimeError("Component loader is not initialized.")
    latency = time.time() - start
    statistics_dict["opea_service@text2video"].append_latency(latency, None)
    return results


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos/{video_id}",
    host="0.0.0.0",
    port=9396,
    methods=["GET"],
)
@register_statistics(names=["opea_service@text2video"])
async def get_video(video_id: str):
    job_file = os.path.join(os.getenv("VIDEO_DIR"), "job.txt")
    if os.path.exists(job_file):
        with open(job_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                job = line.strip().split(",")
                if job[0] == video_id:
                    return Text2VideoOutput(
                        id=job[0],
                        model=os.getenv("MODEL"),
                        status=job[1],
                        progress=100 if job[1] == "completed" else 0,
                        created_at=int(job[2]),
                        seconds=job[5],
                        size=job[6],
                        quality=job[7],
                    )
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": f"Video with id {video_id} not found."})


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos/{video_id}/content",
    host="0.0.0.0",
    port=9396,
    methods=["GET"],
)
@register_statistics(names=["opea_service@text2video"])
async def get_video_content(video_id: str):
    video_file = os.path.join(os.getenv("VIDEO_DIR"), f"{video_id}.mp4")
    if os.path.exists(video_file):
        return FileResponse(video_file, media_type="video/mp4", filename=f"{video_id}.mp4")
    else:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": f"Video with id {video_id} not found."})


def main():
    """
    Main function to set up and run the text-to-video microservice.
    """
    global component_loader

    parser = argparse.ArgumentParser(description="Text-to-Video Microservice")
    parser.add_argument(
        "--model_name_or_path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="Model name or path."
    )
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Enable HPU graphs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., 'cpu', 'hpu').")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token for private models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision.")
    parser.add_argument("--video_dir", type=str, default="/home/user/video", help="Video output directory.")

    args = parser.parse_args()

    if os.getenv("MODEL") is None:
        os.environ["MODEL"] = args.model_name_or_path

    if os.getenv("VIDEO_DIR") is None:
        os.environ["VIDEO_DIR"] = args.video_dir

    text2video_component_name = os.getenv("TEXT2VIDEO_COMPONENT_NAME", "OPEA_TEXT2VIDEO")

    try:
        component_loader = OpeaComponentLoader(
            component_name=text2video_component_name,
            description=f"OPEA IMAGES_GENERATIONS Component: {text2video_component_name}",
            config=args.__dict__,
            seed=args.seed,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            token=args.token,
            bf16=args.bf16,
            use_hpu_graphs=args.use_hpu_graphs,
            video_dir=args.video_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize component loader: {e}")
        exit(1)

    logger.info("Text-to-video server started.")
    opea_microservices["opea_service@text2video"].start()


if __name__ == "__main__":
    main()
