"""
Provide a lightweight server/client implementation template for deploying your generalist policy over a
REST API. This template implements *just* the server.
See auto_eval/robot/policy_clients.py:OpenWebClient for an example of how the client is handled.

Dependencies:
pip install uvicorn fastapi json-numpy draccus

Usage:
python policy_server.py --port 8000

To make your server accessible on the open web, you can use ngrok or bore.pub
With ngrok:
  ngrok http 8000
With bore.pub:
  bore local 8000 --to bore.pub

Note that if you aren't able to resolve bore.pub's DNS (test this with `ping bore.pub`), you can use their actual IP: 159.223.171.199

Adapted from: https://github.com/openvla/openvla/blob/main/vla-scripts/deploy.py
"""

import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import time

import draccus
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# === Server Interface ===
class PolicyServer:
    """
    A simple server for your robot policy; exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image": np.ndarray, "instruction": str, "proprio": Optional[np.ndarray]}
        => Returns  {"action": np.ndarray}
    """

    def __init__(self, cfg):
        """Load policy model following logic in `run_libero_eval.py`."""
        from experiments.robot.robot_utils import get_model
        from experiments.robot.openvla_utils import get_processor, hf_to_vllm

        self.cfg = cfg
        # Load model
        self.model = get_model(self.cfg)

        # If openvla, adjust unnorm_key if suffixed version exists
        if self.cfg.model_family == "openvla":
            if self.cfg.unnorm_key not in self.model.norm_stats and f"{self.cfg.unnorm_key}_no_noops" in self.model.norm_stats:
                self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
            assert (
                self.cfg.unnorm_key in self.model.norm_stats
            ), f"Action un-norm key {self.cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Processor (OpenVLA only)
        self.processor = None
        if self.cfg.model_family == "openvla":
            self.processor = get_processor(self.cfg)
            if getattr(self.cfg, "use_vllm", False):
                self.model = hf_to_vllm(self.model, self.processor, self.cfg)
                self.model.use_vllm = True
            else:
                self.model.use_vllm = False

        logging.info("Model and processor loaded for PolicyServer")

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        """
        Predict a 7-dim action given an image + proprio + instruction
        """
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            if "image" not in payload or "instruction" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: image and instruction",
                )
            image, instruction = payload["image"], payload["instruction"]
            proprio = payload.get("proprio", None)  # proprio is optional

            # Build observation dict consistent with eval_libero (proprio optional)
            if proprio is None:
                proprio = np.zeros(7)  # placeholder if not provided (eef pos(3)+eef axisangle(3)+grip(1))
            observation = {
                "full_image": image,
                "state": proprio,
            }

            # Invoke model inference using shared helper
            from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action

            start_time = time.perf_counter()
            infer_time, action, generated_ids = get_action(
                self.cfg,
                self.model,
                observation,
                instruction,
                processor=self.processor,
                max_new_tokens=self.cfg.max_new_tokens,
            )
            latency = time.perf_counter() - start_time
            if type(action) is tuple:
                action = action[0]
            print(f"Predicted action: {action} (in {latency:.2f}s, infer {infer_time:.2f}s)")

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=(
                    "Error processing request."
                    "Make sure your request complies with the expected format:\n"
                    "{'image': np.ndarray, 'instruction': str}\n"
                ),
            )
        
    def reset(self) -> None:
            """
            Reset the server state (observation history and action history).
            """
            return  # No state to reset for stateless model
    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Add health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        self.app.post("/act")(self.predict_action)

        # Add reset endpoint
        @self.app.post("/reset")
        async def reset_server():
            self.reset()
            return {"status": "reset successful"}

        # Configure server with increased timeout and request size limits
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            timeout_keep_alive=120,
            # limit_concurrency=2,
        )
        server = uvicorn.Server(config)
        server.run()


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 8000  # Host Port

    # Model-specific parameters (mirroring GenerateConfig subset)
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "Embodied-CoT/ecot-openvla-7b-bridge"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    norm_stats: Optional[str] = None

    # Inference options
    unnorm_key: str = "bridge_orig"
    use_vllm: bool = False
    async_engine: bool = False
    max_new_tokens: int = 1024


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    if not cfg.pretrained_checkpoint:
        raise ValueError("--pretrained_checkpoint must be provided")
    server = PolicyServer(cfg)
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()