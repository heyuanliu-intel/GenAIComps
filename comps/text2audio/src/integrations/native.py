# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import time
import threading
import torch
import random
import numpy as np
import torchaudio
import librosa

from fastapi.responses import FileResponse
from cosyvoice.utils.file_utils import load_wav
from comps.cores.proto.api_protocol import AudioSpeechRequest
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry

logger = CustomLogger("opea_text2audio")

# Global variables for the model pipeline and initialization state
cosyvoice = None
initialization_lock = threading.Lock()
initialized = False


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_wav(wav, target_sr: int = 16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


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
        logger.info(f"Loading CosyVoice2 model: {model_name_or_path}")
        self.device = device
        self.cosyvoice = cosyvoice
        logger.info("CosyVoice2 model initialized.")

    async def invoke(self, input: AudioSpeechRequest):
        """Invokes the text2audio service to generate audio for the provided input.

        Args:
            input (AudioSpeechRequest): The input for text2audio service, including text, model, voice, etc.
        """
        set_all_random_seed(input.seed or 0)
        text = input.input
        voice = input.voice or "default"
        speed = input.speed or 1.0
        response_format = input.response_format or "wav"
        logger.info(f"Generating audio with text: {text[:50]}..., voice: {voice}, speed: {speed}, format: {response_format}")

        audio_dir = os.getenv("AUDIO_DIR", "/home/user/audio")
        os.makedirs(audio_dir, exist_ok=True)

        # logging.info('get zero_shot inference request')
        # prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        # set_all_random_seed(seed)
        # for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
        #     yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        #     if not stream:
        #         yield (cosyvoice.sample_rate, wa_data)
        # if input.input_reference:
        #     image_file = os.path.join(self.video_dir, f"{job_id}_input_reference")
        #     contents = await input.input_reference.read()
        #     with open(image_file, "wb") as img_f:
        #         img_f.write(contents)
        #     job.append(image_file)
        # else:
        #     job.append("N/A")

        created = time.time()
        output_path = f"{audio_dir}/audio_{int(created)}.wav"
        with torch.no_grad():
            # The inference_instruct2 is a generator, which yields the TTS speech chunk by chunk.
            # Concatenate the chunks into a single audio.
            output = self.cosyvoice.inference_instruct2(text, voice, self.prompt_speech_16k, speed=speed)
            wav = torch.cat([i["tts_speech"] for i in output])
            torchaudio.save(output_path, wav, self.cosyvoice.sample_rate)

        # os.remove(output_path)
        return FileResponse(output_path, media_type="audio/wav", filename=f"audio_{int(created)}.wav")

    def check_health(self) -> bool:
        return True
