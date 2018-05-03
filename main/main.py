#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import yoloparser

import cv2

try:
    from PIL import Image
    from PIL import ImageDraw
    import pycuda.driver as cuda
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have pycuda and the example dependencies installed. 
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

try:
    import uff
except ImportError:
    raise ImportError("""Please install the UFF Toolkit""")

try:
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have the TensorRT Library installed 
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1)

class Profiler(trt.infer.Profiler):
    """
    Example Implimentation of a Profiler
    Is identical to the Profiler class in trt.infer so it is possible
    to just use that instead of implementing this if further
    functionality is not needed
    """
    def __init__(self, timing_iter):
        trt.infer.Profiler.__init__(self)
        self.timing_iterations = timing_iter
        self.profile = []

    def report_layer_time(self, layerName, ms):
        record = next((r for r in self.profile if r[0] == layerName), (None, None))
        if record == (None, None):
            self.profile.append((layerName, ms))
        else:
            self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

    def print_layer_times(self):
        totalTime = 0
        for i in range(len(self.profile)):
            print("{:40.40} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
            totalTime += self.profile[i][1]
        print("Time over all layers: {:4.3f}".format(totalTime / self.timing_iterations))

TIMING_INTERATIONS = 1000

G_PROFILER = Profiler(TIMING_INTERATIONS)       
# create a logger
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

# define some constants
INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['conv9']
INPUT_C = 3
INPUT_H = 416
INPUT_W = 416
MAX_BATCH_SIZE = 1 # is the size for which the engine will be tuned. At execution time, smaller batches may be used, but not larger
# the execution of smaller batch sizes may be slower than with a TensorRT engine optimized for that size
MAX_WORKSPACE = 1 << 20 # default from MNIST example 

# constants specifically for YOLO
output_wd = 13
nclass = 20
class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
threshold = 0.2
nms = 0.2
nbox = 5
biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
OUTPUT_SIZE = output_wd * output_wd * nbox * (nclass+5)

PARSER = argparse.ArgumentParser(description="TensorFlow based TensorRT Engine")
PARSER.add_argument('datadir',help='Path to Python TensorRT data directory (realpath)')
PARSER.add_argument('--timeit',help='(optional) to time the inference process without actually output the result (using context execute)',action='store_true')
PARSER.add_argument('--validate',help='validation flag, will run inference process against the validation data sets specified beforehand. will be ignored if timeit are True.',action='store_true')
ARGS = PARSER.parse_args()

# define some paths
DATA_DIR = ARGS.datadir # /tiny-yolo-voc
TIMEIT = ARGS.timeit 
VALIDATE = ARGS.validate
DATA = DATA_DIR + '/VOCdevkit/'

# input preprocessing
def convert2ppm(filepath,img_id):
	"""
	Convert the original image to PPM format using PIL

	Parameters
	----------
	filepath: path to input image
	img_id: input image id/name (for saving the result)

	Returns
	-------
	PPM image array 
	"""
	# TensorRT does not depend on any computer vision libraris, the images
	# are represented in binary RGB values for each pixels.
	# The format is PPM.
	# Faster RCNN have trained the network such that the first conv layer sees the image data in BGR, need to reverse
	output_path="/tiny-yolo-voc/ppm_imgs/"
	with Image.open(filepath) as img:
		img.convert('RGB').save(output_path+img_id+".ppm","PPM")
		print ("PPM image is saved at {}".format(output_path+img_id))

	return Image.open(output_path+img_id+".ppm")

# image to array as testcase
def get_testcase(filepath):
	"""
	Get the original image information and preprocess it to match TensorRT need

	Parameters
	----------
	filepath: path to input image

	Returns
	-------
	PPM image array, original image id and original image size
	"""
	#parse the filepath to filename and image id
	filename = filepath.split("/")[-1]
	img_id = filename.split(".")[0]

	#read image and reshape the image if it's not the same as our expected input dimension
	img_jpg = cv2.imread(filepath)
	output_path="/tiny-yolo-voc/resized_jpgs/"
	if (img_jpg.shape[:2] != (INPUT_H,INPUT_W)):
		img_res = cv2.resize(img_jpg,dsize=(INPUT_H,INPUT_W),interpolation=cv2.INTER_LANCZOS4)
		print ("New Image shape: {}".format(img_res.shape))
		cv2.imwrite(output_path+img_id+".jpg",img_res)

	#convert jpg to ppm
	img_ppm = convert2ppm(output_path+img_id+".jpg",img_id)
	assert(img_ppm)
	print("Test image: {} \nFormat: {} Size:{} Mode: {}".format(filepath,img_ppm.format,img_ppm.size,img_ppm.mode)) #im.size(w,h)

	#put the image to array
	arr = np.array(img_ppm)
	print ("Image array shape: {}".format(arr.shape)) #shape (h,w,c)

	#make array 1D
	img_resized_arr = arr.ravel()
	print("1D array size: {}".format(img_resized_arr.size))
	return img_resized_arr,img_id,img_ppm.size[0],img_ppm.size[1]

# run inference on device (GPU)
def infer(context, input_img, output_size, batch_size):
    #load engine
    engine = context.get_engine()
    #print("Bindings: {}").format(engine.get_nb_bindings())
    assert(engine.get_nb_bindings() == 2)
    #convert input data to Float32
    input_img = input_img.astype(np.float32)

    #create output array to receive data 
    #output = np.empty(output_size, dtype = np.float32)

    #alocate pagelocked memory
    output = cuda.pagelocked_empty(output_size, dtype=np.float32)

    # #alocate device memory
    # print(input_img.size)
    # print(input_img.dtype.itemsize) #itemsize in byte
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    #transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream) #likely copy from here(pc) to device(gpu)
    #execute model 
    context.enqueue(batch_size, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    #synchronise threads
    stream.synchronize()

    #save our engine to a file to use later
    #trt.utils.write_engine_to_file("/root/tensorrt/tiny-yolo.engine", engine.serialize())
    return output

#draw the resulted bounding boxes
def draw_bbox(img,box,color): 
	"""
	Draw bbox on top of an image using PIL.ImageDraw

	Parameters
	----------
	img: source image (array in RGB)
	box : list of DetectedObject / bounding box(s) 
	color : box outline color
	"""
	if (color =="green"):
	    color = (0,255,0)
	elif (color == "red"):
	    color = (0,0,255)
	else:
	    color = (255,0,0)

	draw = ImageDraw.Draw(img)
	draw.rectangle(box,outline=color) # Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1] - top-left corner and bottom-right corner

#save the resulted bounding boxes
def save_results(filepath,results,img_width,img_height,img_id,output_dir,draw=True):
	"""
	Save the resulted bounding boxes to a text file with image id information
	and draw rectangles to the original images

	Parameters
	----------
	filepath: source image (original jpg/png, before it converted to ppm)
	results: list of DetectedObject
	img_width: original image width
	img_height: original image height
	img_id: original image id/name
	output_dir: absolute path to the directory where the result text files will be saved into
	"""
	default_yolo_header = "comp4_det_test_"

	#copy the image to temporary variable
	with Image.open(filepath) as img:
		img_copy = img.convert('RGB').copy()

	for i in range(len(results)):
		conf = results[i].conf
		xmin = results[i].xmin
		xmax = results[i].xmax
		ymin = results[i].ymin
		ymax = results[i].ymax
		if xmin<0.0:
			xmin = 0.0
		if ymin<0.0:
			ymin = 0.0
		if xmax>img_width:
			xmax=img_width
		if ymax>img_height:
			ymax=img_height
		fo = open(output_dir+default_yolo_header+results[i].object_class+".txt","a")
		fo.write("{} {:f} {:f} {:f} {:f} {:f}\n".format(img_id,conf,xmin,ymin,xmax,ymax))
		if draw:
			box = [results[i].xmin,results[i].ymin,results[i].xmax,results[i].ymax]
			draw_bbox(img_copy,box,"green")

	output_path="/tiny-yolo-voc/out_imgs/"
	img_copy.save(output_path+img_id+".jpg","JPEG")

#Run inference on device
def time_inference(context,engine, batch_size):
    assert(engine.get_nb_bindings() == 2)

    input_index = engine.get_binding_index(INPUT_LAYERS[0])
    output_index = engine.get_binding_index(OUTPUT_LAYERS[0])

    input_dim = engine.get_binding_dimensions(input_index).to_DimsCHW()
    output_dim = engine.get_binding_dimensions(output_index).to_DimsCHW()

    insize = batch_size * input_dim.C() * input_dim.H() * input_dim.W() * 4
    outsize = batch_size * output_dim.C() * output_dim.H() * output_dim.W() * 4

    d_input = cuda.mem_alloc(insize)
    d_output = cuda.mem_alloc(outsize)

    bindings = [int(d_input), int(d_output)]

    cuda.memset_d32(d_input, 0, insize // 4)

    for i in range(TIMING_INTERATIONS):
        context.execute(batch_size, bindings)

    return

def main():
	# generate test case for our engine
	img_input = DATA + '/VOC2012/JPEGImages/2008_000016.jpg'
	img, img_id, img_w, img_h = get_testcase(img_input) #img in ppm format

	# convert model to UFF
	uff_model = uff.from_tensorflow_frozen_model('/tiny-yolo-voc/tiny-yolo-graph-tf17.pb',["22-convolutional"])

	# convert model to TensorRT model
	model_parser = uffparser.create_uff_parser()
	model_parser.register_input("input",(3,416,416),0) #input name, input dims, input order
	model_parser.register_output("22-convolutional")

	# create engine, context, and runtime
	engine = trt.utils.uff_to_trt_engine(G_LOGGER,
        uff_model,
        model_parser,
        MAX_BATCH_SIZE,
        MAX_WORKSPACE)

	assert(engine)

	runtime = trt.infer.create_infer_runtime(G_LOGGER)
	context = engine.create_execution_context()
	context.set_profiler(G_PROFILER)

	if (TIMEIT):
		time_inference(context,engine,1)
	else:
		if (VALIDATE):
			f = open("/tiny-yolo-voc/2012_val.txt","r")
			for image_path in f:
				image_path = image_path.strip()
				image_jpg = image_path.split("/")[-1]
				img_input = DATA + '/VOC2012/JPEGImages/' + image_jpg
				img, img_id, img_w, img_h = get_testcase(img_input)
				out = infer(context, img, OUTPUT_SIZE,1) # infer use context.enqueue(): asynchronous process with cuda stream. TensorRT does not support profiling on this at the moment
				
				# parse output
				output_parser = yoloparser.yolov2parser(out,output_wd,nclass,nbox,class_name,biases) 
				result = output_parser.interpret(threshold,nms,img_w,img_h)
				save_results(img_input,result,img_w,img_h,img_id,"/tiny-yolo-voc/results/")
		else:
			out = infer(context, img, OUTPUT_SIZE,1) # infer use context.enqueue(): asynchronous process with cuda stream. TensorRT does not support profiling on this at the moment
				
			# parse output
			output_parser = yoloparser.yolov2parser(out,output_wd,nclass,nbox,class_name,biases) 
			result = output_parser.interpret(threshold,nms,img_w,img_h)
			save_results(img_input,result,img_w,img_h,img_id,"/tiny-yolo-voc/results/")
	
	context.destroy()
	engine.destroy()
	runtime.destroy()

	#G_PROFILER.print_layer_times() # not include pre-process and post-process time

if __name__ == "__main__":
    main()
