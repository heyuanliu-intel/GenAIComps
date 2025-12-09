# OPEA Text-to-Audio (TTS) Service

## Overview

This service provides a text-to-audio (TTS) microservice using the `iic/CosyVoice2-0.5B` model, following the OpenAI TTS API interface.

## API Endpoint

- **URL**: `/v1/audio/speech`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Response**: Audio file (default: mp3)

## Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `string` | Yes | The text to generate audio for. Maximum length: 4096 characters. |
| `model` | `string` | Yes | One of the available TTS models. Default: `iic/CosyVoice2-0.5B` |
| `voice` | `string` | Yes | The voice to use. Supported voices: `default`, `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse` |
| `instructions` | `string` | No | Additional voice control instructions. |
| `response_format` | `string` | No | Audio format. Supported: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`. Default: `mp3` |
| `speed` | `number` | No | Audio speed. Range: 0.25-4.0. Default: 1.0 |
| `stream_format` | `string` | No | Streaming format. Supported: `sse`, `audio`. Default: `audio` |

## Usage Examples

### Using curl

```bash
curl https://localhost:9380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "iic/CosyVoice2-0.5B",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

### Using Python

```python
import requests

url = "https://localhost:9380/v1/audio/speech"
headers = {
    "Content-Type": "application/json"
}
payload = {
    "model": "iic/CosyVoice2-0.5B",
    "input": "Hello, this is a test of the text-to-audio service.",
    "voice": "ash",
    "speed": 1.2
}

response = requests.post(url, headers=headers, json=payload)
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd comps/text2audio/src
   ```

2. Install dependencies:
   ```bash
   # For CPU
   pip install -r requirements-cpu.txt
   
   # For GPU with CUDA
   pip install -r requirements-gpu.txt
   ```

3. Run the service:
   ```bash
   python opea_text2audio_microservice.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   cd comps/text2audio/src
   docker build -t opea/text2audio:latest .
   ```

2. Run the container:
   ```bash
   docker run -p 9380:9380 opea/text2audio:latest
   ```

### Docker Compose

```bash
cd comps/text2audio/deployment/docker_compose
docker-compose up -d
```

## Configuration

The service can be configured using command-line arguments or environment variables:

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--model_name_or_path` | - | `iic/CosyVoice2-0.5B` | Model name or path |
| `--device` | - | `cpu` | Device to use: `cpu` or `cuda` |
| - | `TEXT2AUDIO_COMPONENT_NAME` | `OPEA_TEXT2AUDIO` | Component name |

## Supported Models

- `iic/CosyVoice2-0.5B` (default)

## Supported Voices

- `default`
- `alloy`
- `ash`
- `ballad`
- `coral`
- `echo`
- `fable`
- `onyx`
- `nova`
- `sage`
- `shimmer`
- `verse`

## Audio Formats

- `mp3` (default)
- `opus`
- `aac`
- `flac`
- `wav`
- `pcm`

## Performance

- The service supports both CPU and GPU inference
- For best performance, use a GPU with CUDA support
- Audio generation time depends on text length and complexity

## Health Check

The service includes a built-in health check mechanism.

## Logs

Logs are generated using the OPEA CustomLogger and can be configured as needed.

## License

This project is licensed under the Apache 2.0 License.
