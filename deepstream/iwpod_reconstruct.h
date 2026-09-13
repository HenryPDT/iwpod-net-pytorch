// iwpod_reconstruct.h — decoder declaration for the IWPOD preprocess TU.
// See iwpod_reconstruct.cpp for the documented old-decoder bug fixes.
#pragma once

#include <opencv2/opencv.hpp>

namespace iwpod {

// pred: raw output pointer (host). Layout selected by is_nchw:
//   NCHW: [C,Gh,Gw], C==7 (logit + affine6). from_logits=1.
//   NHWC legacy: [Gh,Gw,C], C==8 (prob + bg + affine6). from_logits=0.
// in_w/in_h: SGIE input (infer-dims) in pixels. out_w/out_h: plate size.
// Returns detection confidence (0.0 = none). tensor_output[0..3]=xs,
// [4..7]=ys of the winning quad in vehicle-crop pixels.
float reconstructIwpod(cv::Mat* output, float* tensor_output, const cv::Mat& image,
                       const float* pred, int C, int Gh, int Gw, bool is_nchw,
                       int in_w, int in_h, int out_w, int out_h, double net_stride,
                       double side, double min_probability, bool from_logits,
                       int topk, double nms_iou);

}  // namespace iwpod
