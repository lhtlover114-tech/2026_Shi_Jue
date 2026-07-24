
[basic]
type = axmodel
model_npu = model_296595_npu.axmodel
model_vnpu = model_296595_vnpu.axmodel

[extra]
model_type = yolov5
type=detector
input_type = rgb

input_cache = true
output_cache = true
input_cache_flush = false
output_cache_inval = true

anchors = 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
labels = gangzhu

mean = 123.5, 123.5, 123.5
scale = 0.017124753831663668, 0.017124753831663668, 0.017124753831663668

