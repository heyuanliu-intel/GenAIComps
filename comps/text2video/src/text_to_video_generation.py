#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Adapted from ../stable-diffusion/text_to_image_generation.py

import argparse
import logging
import sys
import time
import os
import torch

from diffusers.utils.export_utils import export_to_video
from optimum.habana.diffusers import GaudiWanPipeline
from optimum.habana.distributed import parallel_state
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.utils import set_seed

try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:
    def check_optimum_habana_min_version(*a, **b):
        return ()
# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.19.0.dev0")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name_or_path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="Model name or path.")
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Enable HPU graphs.")
    parser.add_argument("--device", type=str, default="hpu", help="Device to run the model on (e.g., 'cpu', 'hpu').")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "autocast_bf16"], help="Which runtime dtype to perform generation in.")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Determines how many ranks are divided into context parallel group.")
    parser.add_argument("--video_dir", type=str, default="/home/user/video", help="Video output directory.")
    parser.add_argument("--sep", type=str, default="$###$", help="Video output directory.")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Arguments: {args}")

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    if args.dtype == "autocast_bf16":
        gaudi_config_kwargs["use_torch_autocast"] = True

    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    logger.info(f"Gaudi Config: {gaudi_config}")

    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }

    if args.dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "fp32":
        kwargs["torch_dtype"] = torch.float32

    if args.context_parallel_size > 1 and parallel_state.is_unitialized():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="hccl")
        parallel_state.initialize_model_parallel(sequence_parallel_size=args.context_parallel_size, use_fp8=False)

    pipeline: GaudiWanPipeline = GaudiWanPipeline.from_pretrained(args.model_name_or_path, **kwargs)
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    job_file = os.path.join(args.video_dir, "job.txt")

    while True:
        try:
            if not os.path.exists(job_file):
                time.sleep(1.0)
                continue

            # Read all jobs
            with open(job_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            updated_lines = list(lines)  # Make a mutable copy
            sep = args.sep
            job_processed = False
            for i, line in enumerate(lines):
                parts = line.split(sep)
                if len(parts) < 16:
                    continue

                id, status, created_str, prompt, seconds, size, quality, fps, shift, steps, guide_scale, audio_guide_scale, audio_type, seed, input_reference, audio = parts[:16]
                if status == "queued":
                    # Process the first queued job found
                    fps = int(fps)
                    num_frames = int(seconds) * fps + 1
                    width, height = size.split("x")
                    set_seed(int(seed))
                    generator = torch.manual_seed(int(seed))
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        generator=generator,
                        width=int(width),
                        height=int(height),
                        guidance_scale=float(guide_scale),
                        num_inference_steps=int(steps),
                        num_frames=num_frames,
                    ).frames[0]

                    export_to_video(output, os.path.join(args.video_dir, f"{id}.mp4"), fps=fps)
                    logger.info(f"Exported video for job {id} to {args.video_dir}/{id}.mp4")

                    status = "completed"
                    updated_job = [id, status, created_str, prompt, seconds, size, quality, fps, shift, steps, guide_scale, audio_guide_scale, audio_type, seed, input_reference, audio]
                    updated_lines[i] = sep.join(map(str, updated_job))
                    job_processed = True
                    break  # Exit after processing one job to rewrite the file

            # If a job was processed, rewrite the entire job file
            if job_processed:
                with open(job_file, "w") as f:
                    for l in updated_lines:
                        f.write(f"{l}\n")

        except Exception as e:
            logger.error(f"Job worker encountered an error: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
