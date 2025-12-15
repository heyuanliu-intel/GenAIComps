# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

from fastapi import Depends, Request, status
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


def validate_form_parameters(form):
    """Validate and convert form parameters to their expected types."""
    try:
        audio = []
        if "audio[]" in form:
            audio += form.getlist("audio[]")
        elif "audio" in form:
            audio += form.getlist("audio")

        params = {
            "prompt": form.get("prompt"),
            "input_reference": form.get("input_reference"),
            "audio": audio,
            "audio_guide_scale": float(form.get("audio_guide_scale", 5.0)),
            "audio_type": form.get("audio_type", "add"),
            "model": form.get("model"),
            "seconds": int(form.get("seconds", 4)),
            "fps": int(form.get("fps", 24)),
            "shift": float(form.get("shift", 5.0)),
            "steps": int(form.get("steps", 50)),
            "seed": int(form.get("seed", 42)),
            "guide_scale": float(form.get("guide_scale", 5.0)),
            "size": form.get("size", "720x1280"),
        }
        # Validate size format
        width, height = params["size"].split("x")
        if not (width.isdigit() and height.isdigit()):
            raise ValueError("Invalid size format. Expected 'widthxheight'.")

        if not params["input_reference"] or len(params["audio"]) == 0:
            raise ValueError("'input_reference' and 'audio' must be provided.")

        return params, None
    except (ValueError, TypeError) as e:
        error_content = {"error": {"message": f"Invalid parameter type: {e}", "code": "400"}}
        return None, JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_content)


async def resolve_request(request: Request):
    form = await request.form()
    validated_params, error_response = validate_form_parameters(form)
    if error_response:
        return error_response
    return Text2VideoInput(**validated_params)


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
async def text2video(input_data: Text2VideoInput = Depends(resolve_request)) -> Text2VideoOutput:
    """
    Process a text-to-video generation request.

    Args:
        input_data (Text2VideoInput): The input data containing the prompt.

    Returns:
        Text2VideoOutput: The result of the video generation.
    """
    if isinstance(input_data, JSONResponse):
        return input_data
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
        sep = os.getenv("SEP")
        with open(job_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                job = line.strip().split(sep)
                if job[0] == video_id:
                    return Text2VideoOutput(
                        id=job[0],
                        model=os.getenv("MODEL"),
                        status=job[1],
                        progress=100 if job[1] == "completed" else 0,
                        created_at=int(job[2]),
                        seconds=job[4],
                        size=job[5],
                        quality=job[6],
                        error=job[13] if job[1] == "error" else ""
                    )

    content = {
        "error": {
            "message": f"Video with id {video_id} not found.",
            "code": "404"
        }
    }
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)


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
    job_file = os.path.join(os.getenv("VIDEO_DIR"), "job.txt")
    if os.path.exists(job_file):
        sep = os.getenv("SEP")
        with open(job_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                job = line.strip().split(sep)
                if job[0] == video_id:
                    if job[1] != "completed":
                        return Text2VideoOutput(
                            id=job[0],
                            model=os.getenv("MODEL"),
                            status=job[1],
                            progress=0,
                            created_at=int(job[2]),
                            seconds=job[4],
                            size=job[5],
                            quality=job[6],
                        )
                    else:
                        video_file = os.path.join(os.getenv("VIDEO_DIR"), video_id, "output.mp4")
                        if os.path.exists(video_file):
                            return FileResponse(video_file, media_type="video/mp4", filename=f"{video_id}.mp4")

    content = {
        "error": {
            "message": f"Video with id {video_id} not found.",
            "code": "404"
        }
    }
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)


def main():
    """
    Main function to set up and run the text-to-video microservice.
    """
    global component_loader

    parser = argparse.ArgumentParser(description="Text-to-Video Microservice")
    parser.add_argument("--model_name_or_path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="Model name or path.")
    parser.add_argument("--video_dir", type=str, default="/home/user/video", help="Video output directory.")

    args = parser.parse_args()

    if not os.environ.get("MODEL"):
        os.environ["MODEL"] = args.model_name_or_path
    os.environ["VIDEO_DIR"] = args.video_dir
    os.environ["SEP"] = "$###$"
    text2video_component_name = os.getenv("TEXT2VIDEO_COMPONENT_NAME", "OPEA_TEXT2VIDEO")

    try:
        component_loader = OpeaComponentLoader(
            component_name=text2video_component_name,
            description=f"OPEA IMAGES_GENERATIONS Component: {text2video_component_name}",
            config=args.__dict__,
            video_dir=args.video_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize component loader: {e}")
        exit(1)

    logger.info("Text-to-video server started.")
    opea_microservices["opea_service@text2video"].start()


if __name__ == "__main__":
    main()
