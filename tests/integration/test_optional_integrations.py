"""Runtime checks enabled only in the packaged container."""
import os
import shutil
import pytest

pytestmark = pytest.mark.skipif(not os.getenv("ARTIFACT_CONTAINER"), reason="container runtime checks")

def test_task_runtimes():
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter
    import cv2
    import sklearn
    import numpy as np
    assert shutil.which("arduino-cli")
    assert shutil.which("ssh")
    model = tf.keras.models.load_model("inputs/convert/model.keras")
    calibration = np.load("inputs/convert/calibration.npy")
    assert model.input_shape[-1] == calibration.shape[-1] == 3
    runtime = Interpreter(model_path="inputs/pysketch/detect.tflite")
    runtime.allocate_tensors()
