#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import requests
import argparse

def test_text2audio():
    parser = argparse.ArgumentParser(description="Test Text2Audio service")
    parser.add_argument("--url", type=str, default="http://localhost:9380/v1/audio/speech")
    parser.add_argument("--output", type=str, default="test_output.mp3")
    
    args = parser.parse_args()
    
    # Test payload
    payload = {
        "model": "iic/CosyVoice2-0.5B",
        "input": "Hello, this is a test of the OPEA Text2Audio service using the CosyVoice2 model.",
        "voice": "default",
        "speed": 1.0,
        "response_format": "mp3"
    }
    
    try:
        print(f"Testing Text2Audio service at {args.url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Send POST request
        response = requests.post(args.url, json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            # Save the audio file
            with open(args.output, "wb") as f:
                f.write(response.content)
            print(f"✓ Audio generated successfully! Saved to {args.output}")
            print(f"✓ Audio size: {len(response.content)} bytes")
            print(f"✓ Response format: {response.headers.get('Content-Type')}")
        else:
            print(f"✗ Failed to generate audio. Status code: {response.status_code}")
            print(f"✗ Response: {response.text}")
            
    except Exception as e:
        print(f"✗ Error during test: {e}")

if __name__ == "__main__":
    test_text2audio()
