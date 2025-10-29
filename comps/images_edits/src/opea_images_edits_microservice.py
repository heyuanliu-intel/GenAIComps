# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse

from typing import List, Union
from fastapi import Form, File, UploadFile, Depends, Request

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    SDOutputs,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.images_edits.src.integrations.native import OpeaImagesEdits
from comps.cores.proto.api_protocol import (
    ImagesEditsInput,
)

args = None
logger = CustomLogger("images_edits")
component_loader = None

async def resolve_images_edits_request(request: Request):
    form = await request.form()

    common_args = {
        "image": form.getlist("image[]"),
        "prompt": form.get("prompt", None),
        "background": form.get("background", None),
        "input_fidelity": form.get("input_fidelity", None),
        "mask": form.get("mask", None),
        "model": form.get("model", None),
        "n": form.get("n", 1),
        "output_compression": form.get("output_compression", None),
        "output_format": form.get("output_format", None),
        "partial_images": form.get("partial_images", False),
        "quality": form.get("quality", None),
        "response_format": form.get("response_format", "url"),
        "size": form.get("size", None),
        "stream": form.get("stream", False),
        "user": form.get("user", None),
    }

    return ImagesEditsInput(**common_args)


@register_microservice(
    name="opea_service@images_edits",
    service_type=ServiceType.IMAGES_EDITS,
    endpoint="/v1/images/edits",
    host="0.0.0.0",
    port=9390,
)

@register_statistics(names=["opea_service@images_edits"])
async def images_edits(input: ImagesEditsInput = Depends(resolve_images_edits_request)) -> SDOutputs:
    start = time.time()
    results = await component_loader.invoke(input)
    statistics_dict["opea_service@images_edits"].append_latency(time.time() - start, None)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()
    images_edits_component_name = os.getenv("IMAGES_EDITS_COMPONENT_NAME", "OPEA_IMAGES_EDITS")
    # Register components
    try:
        # Initialize OpeaComponentLoader
        component_loader = OpeaComponentLoader(
            images_edits_component_name,
            description=f"OPEA IMAGES_EDITS Component: {images_edits_component_name}",
            config=args.__dict__,
            seed=args.seed,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            token=args.token,
            bf16=args.bf16,
            use_hpu_graphs=args.use_hpu_graphs,
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        exit(1)

    logger.info("images_edits server started.")
    opea_microservices["opea_service@images_edits"].start()
