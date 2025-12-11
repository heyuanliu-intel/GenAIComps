# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from fastapi import Depends, Request
from comps.cores.proto.api_protocol import AudioSpeechRequest
from comps.text2audio.src.integrations.native import OpeaText2audio

logger = CustomLogger("opea_text2audio_microservice")


async def resolve_request(request: Request):
    form = await request.form()
    common_args = {
        "input": form.get("input"),
        "sample": form.get("sample", None),
        "sample_input": form.get("sample_input", None),
        "model": form.get("model", "iic/CosyVoice2-0.5B"),
        "voice": form.get("voice", "default"),
        "speed": float(form.get("speed", 1.0)),
        "seed": int(form.get("seed", 0)),
        "response_format": form.get("response_format", "wav"),
    }
    return AudioSpeechRequest(**common_args)


@register_microservice(
    name="opea_service@text2audio",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/speech",
    host="0.0.0.0",
    port=9380,
    input_datatype=AudioSpeechRequest,
    output_datatype=bytes,
)
@register_statistics(names=["opea_service@text2audio"])
async def text2audio(input: AudioSpeechRequest = Depends(resolve_request)):
    start = time.time()
    try:
        # Use the loader to invoke the active component
        results = await loader.invoke(input)
        statistics_dict["opea_service@text2audio"].append_latency(time.time() - start, None)
        return results
    except Exception as e:
        logger.error(f"Error during text2audio invocation: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="iic/CosyVoice2-0.5B")
    parser.add_argument("--device", type=str, default="hpu")
    parser.add_argument("--root_dir", type=str, default="/home/user/CosyVoice")
    parser.add_argument("--audio_dir", type=str, default="/home/user/audio", help="Audio output directory.")

    args = parser.parse_args()
    os.environ["MODEL"] = args.model_name_or_path
    os.environ["ROOT_DIR"] = args.root_dir
    os.environ["AUDIO_DIR"] = args.audio_dir

    text2audio_component_name = os.getenv("TEXT2AUDIO_COMPONENT_NAME", "OPEA_TEXT2AUDIO")
    # Initialize OpeaComponentLoader
    loader = OpeaComponentLoader(
        text2audio_component_name,
        description=f"OPEA TEXT2AUDIO Component: {text2audio_component_name}",
        config=args.__dict__,
    )

    logger.info("Text2audio server started.")
    opea_microservices["opea_service@text2audio"].start()
