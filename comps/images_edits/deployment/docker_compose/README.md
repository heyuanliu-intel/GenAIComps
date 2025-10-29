# Deploying Image-to-Image Service

This document provides a comprehensive guide to deploying the Image-to-Image microservice pipeline on Intel platforms.

This guide covers two deployment methods:

- [🚀 1. Quick Start with Docker Compose](#-1-quick-start-with-docker-compose): The recommended method for a fast and easy setup.
- [🚀 2. Manual Step-by-Step Deployment (Advanced)](#-2-manual-step-by-step-deployment-advanced): For users who want to build and run the container individually.

## 🚀 1. Quick Start with Docker Compose

This method uses Docker Compose to start the service with a single command. It is the fastest and easiest way to get the service running.

### 1.1. Access the Code

Clone the repository and navigate to the deployment directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps/comps/images_edits/deployment/docker_compose
```

### 1.2. Set Hugging Face Token

Set your Hugging Face token as an environment variable. This is required to download the Stable Diffusion model.

```bash
export HF_TOKEN=<your huggingface token>
```

### 1.3. Deploy the Service

Choose the command corresponding to your target platform.

- **For Intel® Xeon® CPU:**

  ```bash
  docker compose -f compose.yaml up images_edits -d
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker compose -f compose.yaml up images_edits-gaudi -d
  ```

### 1.4. Validate the Service

Once the container is running, you can validate the service. **Note:** Run this command from the root of the `GenAIComps` repository.

```bash
export ip_address=$(hostname -I | awk '{print $1}')
curl http://${ip_address}:9389/v1/images_edits -XPOST \
-d '{"image": "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png", "prompt":"a photo of an astronaut riding a horse on mars", "num_images_per_prompt":1}' \
-H 'Content-Type: application/json'
```

The expected output will be a JSON object containing the generated image data.

### 1.5. Clean Up the Deployment

To stop and remove the containers, run the following command from the `comps/images_edits/deployment/docker_compose` directory:

```bash
docker compose down
```

---

## 🚀 2. Manual Step-by-Step Deployment (Advanced)

This section provides detailed instructions for building the Docker image and running the microservice container individually.

### 2.1. Clone the Repository

If you haven't already, clone the repository and navigate to the root directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
```

### 2.2. Build the Docker Image

- **For Intel® Xeon® CPU:**
  ```bash
  docker build -t opea/images_edits:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/images_edits/src/Dockerfile .
  ```
- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker build -t opea/images_edits-gaudi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/images_edits/src/Dockerfile.intel_hpu .
  ```

### 2.3. Configure Environment Variables

Set the necessary environment variables for the container.

- **For both Intel® Xeon® and Intel® Gaudi® 2 HPU:**

  ```bash
  export ip_address=$(hostname -I | awk '{print $1}')
  export HF_TOKEN=<your huggingface token>
  # SDXL is an example, you can use other compatible models.
  export MODEL=stabilityai/stable-diffusion-xl-refiner-1.0
  ```

### 2.4. Run the Microservice Container

#### 2.4.1. Run Image-to-Image Microservice on Xeon

```bash
docker run -d --name "images_edits-service" --ipc=host -p 9389:9389 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HF_TOKEN=$HF_TOKEN -e MODEL=$MODEL opea/images_edits:latest
```

#### 2.4.2. Run Image-to-Image Microservice on Gaudi

```bash
docker run -d --name "images_edits-gaudi-service" -p 9389:9389 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HF_TOKEN=$HF_TOKEN -e MODEL=$MODEL opea/images_edits-gaudi:latest
```

### 2.5. Validate the Service

After starting the container, test the service endpoint. Make sure you are in the root directory of the `GenAIComps` repository.

```bash
# The ip_address variable should be set from step 2.3
curl http://${ip_address}:9389/v1/images_edits -XPOST \
-d '{"image": "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png", "prompt":"a photo of an astronaut riding a horse on mars", "num_images_per_prompt":1}' \
-H 'Content-Type: application/json'
```

You should see a successful response containing the generated image data.

### 2.6. Clean Up the Deployment

To stop and remove the container you started manually, use the `docker stop` and `docker rm` commands.

- **For Intel® Xeon® CPU:**

  ```bash
  docker stop images_edits-service
  docker rm images_edits-service
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker stop images_edits-gaudi-service
  docker rm images_edits-gaudi-service
  ```
