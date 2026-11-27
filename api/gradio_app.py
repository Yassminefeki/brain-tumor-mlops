"""Gradio demo: upload an MRI slice, see the predicted class and the Grad-CAM heatmap.

Run::

    MODEL_CHECKPOINT=models/best.pt python -m api.gradio_app

Deployable as-is to a Hugging Face Space (add the checkpoint to the Space).
"""

from __future__ import annotations

import os

import gradio as gr
import numpy as np

from brain_tumor.inference import Predictor, load_image_bytes

predictor = Predictor(os.environ.get("MODEL_CHECKPOINT", "models/best.pt"))

# One test-set slice per class (patient-level split: never seen in training), shown as
# clickable examples when the dataset is available locally (not in the Docker image).
EXAMPLE_IDS = {"meningioma": 132, "glioma": 2879, "pituitary": 1217}
EXAMPLES = [
    [path]
    for path in (os.path.join("data", "interim", f"{i}.mat") for i in EXAMPLE_IDS.values())
    if os.path.exists(path)
] or None


def analyse(path: str) -> tuple[dict[str, float], np.ndarray]:
    if path is None:
        raise gr.Error("Upload an image first")
    with open(path, "rb") as f:
        image = load_image_bytes(f.read(), path)
    result = predictor.predict(image)
    return result["probabilities"], predictor.explain(image)


demo = gr.Interface(
    fn=analyse,
    inputs=gr.File(label="MRI slice (.mat, .png or .jpg)", type="filepath"),
    outputs=[
        gr.Label(label="Prediction", num_top_classes=3),
        gr.Image(label="Grad-CAM (where the model looks)"),
    ],
    title="Brain tumour MRI classifier",
    description=f"Model: {predictor.model_name}. Classes: meningioma, glioma, pituitary tumour. "
    "Educational project — not a medical diagnosis.",
    examples=EXAMPLES,
    example_labels=list(EXAMPLE_IDS) if EXAMPLES and len(EXAMPLES) == len(EXAMPLE_IDS) else None,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
