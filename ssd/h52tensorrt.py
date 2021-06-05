import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

def allocate_buffers(engine):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):

    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class TrtYOLO(object):

    def _load_engine(self):
        TRTbin = self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def __init__(self, model, input_shape, category_num=80, letter_box=False):

        self.model = model
        self.input_shape = input_shape
        self.catrgory_num = category_num
        self.letter_box = letter_box
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        input_volume = trt.volume((3, 416, 416))
        self.numpy_array = np.zeros((self.engine.max_batch_size, input_volume))

    def infer_webcam(self, arr):
        img = self._load_img_webcam(arr)
        np.copyto(self.inputs[0].host, img.ravel())

        detection_out = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)
        
        return detection_out

    def _load_img_webcam(self, arr):
        image = Image.fromarray(np.uint8(arr))
        model_input_width = 416
        model_input_height = 416
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np
    
    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, 3)
        ).astype(np.uint8)


if __name__ == "__main__":

    trt_inference_wrapper = TrtYOLO("ssdlite_mobilenetv1.trt", (300, 300), 31)

    predicts = []
    with open("test.txt") as f:
        line = f.readline()
        while line:
            l = line.split()
            boxes = l[1].split(",")
            predicts.append([l[0], boxes[0], boxes[1], boxes[2], boxes[3]])
            line = f.readline()
    
    for index, pre in enumerate(predicts):

        #print(pre[0])
        img = cv2.imread(pre[0])
        output, image = trt_inference_wrapper.infer_webcam(img)
        print(output)