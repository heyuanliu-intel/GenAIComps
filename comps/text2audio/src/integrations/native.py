# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import time
import threading
import torch
import torchaudio

from fastapi.responses import FileResponse
from cosyvoice.utils.file_utils import load_wav
from comps.cores.proto.api_protocol import AudioSpeechRequest
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry

logger = CustomLogger("opea_text2audio")

# Global variables for the model pipeline and initialization state
cosyvoice = None
initialization_lock = threading.Lock()
initialized = False


def initialize(model_name_or_path: str = "iic/CosyVoice2-0.5B", device: str = "hpu"):
    """Initialize the model pipeline in a thread-safe manner."""
    global cosyvoice, initialized
    with initialization_lock:
        if initialized:
            return

        model_name = os.getenv("MODEL", model_name_or_path)
        if device == "hpu":
            import habana_frameworks.torch as ht_torch
            import habana_frameworks.torch.core as htcore
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
            adapt_transformers_to_gaudi()
            root_dir = os.getenv("ROOT_DIR", "/home/user/CosyVoice")
            sys.path.append(f"{root_dir}/third_party/Matcha-TTS")
            from cosyvoice.cli.cosyvoice import CosyVoice2
            cosyvoice = CosyVoice2(model_name, load_jit=False, load_trt=False, fp16=False)
            torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)
            model = cosyvoice.model.llm.llm.model.bfloat16().eval().to(device)
            cosyvoice.model.llm.llm.model = wrap_in_hpu_graph(model)
            model = cosyvoice.model.llm.llm_decoder.bfloat16().eval().to(device)
            cosyvoice.model.llm.llm_decoder = wrap_in_hpu_graph(model)
            cosyvoice.model.flow = cosyvoice.model.flow.bfloat16().eval()
            logger.info(f"Gaudi with {model_name} loaded.")
        else:
            raise NotImplementedError(f"Device '{device}' is not supported. Only 'hpu' are supported.")

        logger.info(f"Model '{model_name}' initialized on device '{device}'.")
        initialized = True


@OpeaComponentRegistry.register("OPEA_TEXT2AUDIO")
class OpeaText2audio(OpeaComponent):
    """A specialized text2audio component derived from OpeaComponent for text2audio services.

    Attributes:
        model: The loaded text-to-speech model.
        processor: The processor for the model.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, "text2audio", description, config)

        # initialize model and processor
        model_name_or_path = config["model_name_or_path"]
        device = config["device"]
        initialize(model_name_or_path=model_name_or_path, device=device)
        root_dir = config.get("root_dir", "/home/user/CosyVoice")
        self.prompt_speech_16k = load_wav(f'{root_dir}/asset/zero_shot_prompt.wav', 16000)
        logger.info(f"Loading CosyVoice2 model: {model_name_or_path}")
        self.device = device
        self.cosyvoice = cosyvoice
        logger.info("CosyVoice2 model initialized.")

    async def invoke(self, input: AudioSpeechRequest):
        """Invokes the text2audio service to generate audio for the provided input.

        Args:
            input (AudioSpeechRequest): The input for text2audio service, including text, model, voice, etc.
        """
        text = input.input
        voice = input.voice or "default"
        speed = input.speed or 1.0
        response_format = input.response_format or "wav"
        logger.info(f"Generating audio with text: {text[:50]}..., voice: {voice}, speed: {speed}, format: {response_format}")

        audio_dir = os.getenv("AUDIO_DIR", "/home/user/audio")
        os.makedirs(audio_dir, exist_ok=True)

        created = time.time()
        output_path = f"{audio_dir}/audio_{int(created)}.wav"
        with torch.no_grad():
            # The inference_instruct2 is a generator, which yields the TTS speech chunk by chunk.
            # Concatenate the chunks into a single audio.
            output = self.cosyvoice.inference_instruct2(text, voice, self.prompt_speech_16k)
            wav = torch.cat([i["tts_speech"] for i in output])
            torchaudio.save(output_path, wav, self.cosyvoice.sample_rate)

        # os.remove(output_path)
        return FileResponse(output_path, media_type="audio/wav", filename=f"audio_{int(created)}.wav")

    def check_health(self) -> bool:
        return True
