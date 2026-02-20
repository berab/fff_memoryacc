import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import os

def preprocess_func(images_folder, height, width, size_limit=0):
    # create dummy batch and return sanitized list of 4D batches
    batch_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    out = []
    for i in range(batch_data.shape[0]):
        x = batch_data[i:i+1].astype(np.float32)
        # replace NaN/Inf with finite numbers
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        rmin, rmax = float(x.min()), float(x.max())
        # ensure a non-zero dynamic range (avoid rmax <= rmin which breaks quantization)
        if rmax <= rmin:
            x = x.copy()
            x.flat[0] = x.flat[0] + 1e-6
        out.append(x)
    return out


class FFFDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, 28, 28, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'actual_input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


# change it to your real calibration data set
calibration_data_folder = "calibration_imagenet"
dr = FFFDataReader(calibration_data_folder)

quantize_static('fff_float.onnx',
                'fff_uint8.onnx',
                dr)

print('ONNX full precision model size (MB):', os.path.getsize("fff_float.onnx")/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize("fff_uint8.onnx")/(1024*1024))
