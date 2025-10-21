# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import os
import threading
import time
import tempfile

from typing import List, Union, Optional
from fastapi import Form, UploadFile

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

from comps import (
    CustomLogger,
    SDImg2ImgInputs,
    SDOutputs,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("image2image")
pipe = None
args = None
initialization_lock = threading.Lock()
initialized = False


def initialize():
    global pipe, args, initialized
    with initialization_lock:
        if not initialized:
            # initialize model and tokenizer
            if os.getenv("MODEL", None):
                args.model_name_or_path = os.getenv("MODEL")
            kwargs = {}
            if args.bf16:
                kwargs["torch_dtype"] = torch.bfloat16
            if not args.token:
                args.token = os.getenv("HF_TOKEN")
            if args.device == "hpu":
                kwargs.update(
                    {
                        "use_habana": True,
                        "use_hpu_graphs": args.use_hpu_graphs,
                        "gaudi_config": "Habana/stable-diffusion",
                        "token": args.token,
                    }
                )
                if "stable-diffusion-xl" in args.model_name_or_path:
                    from optimum.habana.diffusers import GaudiStableDiffusionXLImg2ImgPipeline

                    pipe = GaudiStableDiffusionXLImg2ImgPipeline.from_pretrained(
                        args.model_name_or_path,
                        **kwargs,
                    )
                else:
                    raise NotImplementedError(
                        "Only support stable-diffusion-xl now, " + f"model {args.model_name_or_path} not supported."
                    )
            elif args.device == "cpu":
                pipe = AutoPipelineForImage2Image.from_pretrained(args.model_name_or_path, token=args.token, **kwargs)
            else:
                raise NotImplementedError(f"Only support cpu and hpu device now, device {args.device} not supported.")
            logger.info("Stable Diffusion model initialized.")
            initialized = True


@register_microservice(
    name="opea_service@image2image",
    service_type=ServiceType.IMAGE2IMAGE,
    endpoint="/images/generations",
    host="0.0.0.0",
    port=9389,
    input_datatype=SDImg2ImgInputs,
    output_datatype=SDOutputs,
)
@register_statistics(names=["opea_service@image2image"])
async def image2image(input: SDImg2ImgInputs) -> SDOutputs:
    initialize()
    start = time.time()

    if input.image.startswith("data:image"):
        image_content = base64.b64decode(image)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_content)
            temp_file_path = temp_file.name
        image = load_image(temp_file_path).convert("RGB")
        os.unlink(temp_file_path)
    else:
        image = load_image(input.image).convert("RGB")

    generator = torch.manual_seed(input.seed)
    prompt = input.prompt
    guidance_scale = input.guidance_scale
    true_cfg_scale = input.cfg
    num_inference_steps = input.num_inference_steps
    num_images_per_prompt = input.batch_size
    images = pipe(image=image,
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
    statistics_dict["opea_service@image2image"].append_latency(time.time() - start, None)

    return SDOutputs(seed=input.seed, images=results, timings={"inference": time.time() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    logger.info("Image2image server started.")
    opea_microservices["opea_service@image2image"].start()
