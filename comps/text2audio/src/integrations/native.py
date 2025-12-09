# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, AudioSpeechRequest

logger = CustomLogger("opea")


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
        
        logger.info(f"Loading CosyVoice2 model: {model_name_or_path}")
        
        # Load the CosyVoice2 model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        # Set generation config
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        
        self.device = device
        logger.info("CosyVoice2 model initialized.")

    async def invoke(self, input: AudioSpeechRequest) -> bytes:
        """Invokes the text2audio service to generate audio for the provided input.

        Args:
            input (AudioSpeechRequest): The input for text2audio service, including text, model, voice, etc.

        Returns:
            bytes: The generated audio bytes.
        """
        text = input.input
        voice = input.voice or "default"
        speed = input.speed or 1.0
        response_format = input.response_format or "mp3"
        
        logger.info(f"Generating audio with text: {text[:50]}..., voice: {voice}, speed: {speed}, format: {response_format}")
        
        # Process input text
        inputs = self.processor(
            text=text,
            voice=voice,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate audio
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        
        # Post-process the outputs
        audio_values = outputs["audio_values"][0]
        sampling_rate = self.model.config.sampling_rate
        
        # Resample if needed based on speed
        if speed != 1.0:
            # Adjust sampling rate based on speed
            new_sampling_rate = int(sampling_rate * speed)
            from torchaudio.transforms import Resample
            resampler = Resample(orig_freq=sampling_rate, new_freq=new_sampling_rate)
            audio_values = resampler(audio_values)
            sampling_rate = new_sampling_rate
        
        # Save to temporary file and read as bytes
        with tempfile.NamedTemporaryFile(suffix="." + response_format, delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Use torchaudio to save the audio
            import torchaudio
            torchaudio.save(
                temp_path,
                audio_values.unsqueeze(0),
                sampling_rate=sampling_rate,
                format=response_format.upper(),
            )
            
            # Read the audio file as bytes
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return audio_bytes

    def check_health(self) -> bool:
        return True
