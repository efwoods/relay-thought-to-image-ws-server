import asyncio
import base64
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import websockets
from models.waveform_encoder import WaveformEncoder
from models.image_decoder import ImageDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained models
waveform_encoder = WaveformEncoder().to(device).eval()
image_decoder = ImageDecoder().to(device).eval()

clients = set()


def tensor_to_base64(tensor):
    tensor = tensor.squeeze().cpu().clamp(0, 1)
    image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode(
        "utf-8"
    )


async def relay(websocket):
    clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "waveform_latent":
                latent = torch.tensor(data["payload"]).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    z_waveform = waveform_encoder(latent)
                    reconstructed = image_decoder(z_waveform)

                image_b64 = tensor_to_base64(reconstructed)

                response = {
                    "type": "reconstructed_image",
                    "session_id": data["session_id"],
                    "image_base64": image_b64,
                }

                await websocket.send(json.dumps(response))
    finally:
        clients.remove(websocket)


start_server = websockets.serve(relay, "localhost", 8766)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
