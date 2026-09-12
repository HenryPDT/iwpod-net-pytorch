/*
 * nvdspreprocess_iwpod_impl.cpp — fast reimplementation of the WPOD
 * nvdspreprocess TU for IWPOD-v2.
 *
 * DROP-IN RULE: in ocr_preprocessor/Makefile replace
 *     SRCS:= ... nvdspreprocess_impl.cpp ...
 * with
 *     SRCS:= ... nvdspreprocess_iwpod_impl.cpp iwpod_reconstruct_v2.cpp ...
 * (copy this file + iwpod_reconstruct_v2.{h,cpp} from
 * iwpod-net-pytorch/deepstream/ into ocr_preprocessor/ first).
 * No changes to nvdspreprocess_lib.cpp, the Makefile otherwise, any config
 * (except the 3-line SGIE swap in DOWNSTREAM_GUIDE.md §2), LPR.cpp, or OCR.
 *
 * REUSE vs REWRITE split:
 *  - Reused verbatim (NVIDIA boilerplate / pipeline contracts): CudaStream,
 *    CudaDeviceBuffer, setters, mean-file/resource/stream handling, initLib
 *    entry, tensor-meta lookup, cpuPtr reuse, image_info_map send gating,
 *    JPEG telemetry blocks, buffer-hijack layout (wpod[0..10]), rect flash.
 *  - Rewritten (hot path, runs per vehicle per frame): grid decode
 *    (via iwpod_v2::reconstructIwpod: no full-grid Affines copy, no VLA,
 *    correct NCHW indexing, top-k + NMS) and the CHW interleave write
 *    (row-pointer planar memcpy instead of per-pixel .at<Vec3f>).
 *
 * Contracts preserved byte-for-byte: tensor_output[0..7] quad pixels,
 * wpod[8]=confidence, wpod[9]=jpeg bytes, wpod[10]=is-plate, OCR tensor
 * NCHW float [0,1] via pixel_val*255*m_Scale (fused to one multiply).
 */

#include <cuda.h>
#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <opencv2/opencv.hpp>

#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvdspreprocess_conversion.h"
#include "nvtx3/nvToolsExtCudaRt.h"

#include "nvdspreprocess_impl.h"
#include "iwpod_reconstruct_v2.h"

using namespace cv;
using namespace std;

// ---- IWPOD-v2 decode parameters (must match training + export) ------------
static constexpr double kNetStride = 16.0;
static constexpr double kSide = ((208.0 + 40.0) / 2.0) / kNetStride;  // 7.75
static constexpr float kMinConfidence = 0.3f;  // mirrors runtime wpod_threshold default
static constexpr int kTopK = 1;                // 1 reproduces old max-only behavior
static constexpr double kNmsIou = 0.25;
// ---------------------------------------------------------------------------

CudaStream::CudaStream(uint flag, int priority) {
    cudaError_t err = cudaStreamCreateWithPriority(&m_Stream, flag, priority);
    if (err != cudaSuccess) {
        printf("cudaStreamCreateWithPriority failed with err %d : %s\n",
               (int)err,
               cudaGetErrorName(err));
    }
}

CudaStream::~CudaStream() {
    if (m_Stream != nullptr) {
        cudaError_t err = cudaStreamDestroy(m_Stream);
        if (err != cudaSuccess) {
            printf("cudaStreamDestroy failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
        }
    }
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t size)
    : CudaBuffer(size) {
    cudaError_t err = cudaMalloc(&m_Buf, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
    }

    m_Size = size;
}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    if (m_Buf != nullptr) {
        cudaError_t err = cudaFree(m_Buf);
        if (err != cudaSuccess) {
            printf("cudaFree failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
        }
    }
}

NvDsPreProcessTensorImpl::NvDsPreProcessTensorImpl(
        const NvDsPreProcessNetworkSize& size,
        NvDsPreProcessFormat format,
        int id)
    : m_UniqueID(id),
      m_NetworkSize(size),
      m_NetworkInputFormat(format) {}

bool NvDsPreProcessTensorImpl::setScaleOffsets(float scale, const std::vector<float>& offsets) {
    if (!offsets.empty() && m_NetworkSize.channels != (uint32_t)offsets.size()) {
        return false;
    }

    m_Scale = scale;
    if (!offsets.empty()) {
        m_ChannelMeans.assign(offsets.begin(), offsets.begin() + m_NetworkSize.channels);
    }
    return true;
}

bool NvDsPreProcessTensorImpl::setMeanFile(const std::string& file) {
    if (!file_accessible(file)) return false;
    m_MeanFile = file;
    return true;
}

bool NvDsPreProcessTensorImpl::setInputOrder(const NvDsPreProcessNetworkInputOrder order) {
    m_InputOrder = order;
    return true;
}

/* Read the mean image ppm file and copy the mean image data to the mean
 * data buffer allocated on the device memory.
 */
NvDsPreProcessStatus NvDsPreProcessTensorImpl::readMeanImageFile() {
    std::ifstream infile(m_MeanFile, std::ifstream::binary);
    size_t size = m_NetworkSize.width * m_NetworkSize.height * m_NetworkSize.channels;
    uint8_t tempMeanDataChar[size];
    float tempMeanDataFloat[size];
    cudaError_t cudaReturn;

    if (!infile.good()) {
        printf("Could not open mean image file '%s\n'", safeStr(m_MeanFile));
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    std::string magic, max;
    unsigned int h, w;
    infile >> magic >> w >> h >> max;

    if (magic != "P3" && magic != "P6") {
        printf("Magic PPM identifier check failed\n");
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    if (w != m_NetworkSize.width || h != m_NetworkSize.height) {
        printf("Mismatch between ppm mean image resolution(%d x %d) and "
               "network resolution(%d x %d)\n",
               w,
               h,
               m_NetworkSize.width,
               m_NetworkSize.height);
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    infile.get();
    infile.read((char*)tempMeanDataChar, size);
    if (infile.gcount() != (int)size || infile.fail()) {
        printf("Failed to read sufficient bytes from mean file\n");
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    for (size_t i = 0; i < size; i++) {
        tempMeanDataFloat[i] = (float)tempMeanDataChar[i];
    }

    assert(m_MeanDataBuffer);
    cudaReturn = cudaMemcpy(
            m_MeanDataBuffer->ptr(),
            tempMeanDataFloat,
            size * sizeof(float),
            cudaMemcpyHostToDevice);
    if (cudaReturn != cudaSuccess) {
        printf("Failed to copy mean data to mean data buffer (%s)\n", cudaGetErrorName(cudaReturn));
        return NVDSPREPROCESS_CUDA_ERROR;
    }

    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus NvDsPreProcessTensorImpl::allocateResource() {
    if (!m_MeanFile.empty() || m_ChannelMeans.size() > 0) {
        /* Mean Image File specified. Allocate the mean image buffer on device
         * memory. */
        m_MeanDataBuffer = std::make_unique<CudaDeviceBuffer>(
                (size_t)m_NetworkSize.width * m_NetworkSize.height * m_NetworkSize.channels *
                sizeof(float));

        if (!m_MeanDataBuffer || !m_MeanDataBuffer->ptr()) {
            printf("Failed to allocate cuda buffer for mean image");
            return NVDSPREPROCESS_CUDA_ERROR;
        }
    }

    /* Read the mean image file (PPM format) if specified and copy the
     * contents into the buffer. */
    if (!m_MeanFile.empty()) {
        if (!file_accessible(m_MeanFile)) {
            printf("Cannot access mean image file '%s'", safeStr(m_MeanFile));
            return NVDSPREPROCESS_CONFIG_FAILED;
        }
        NvDsPreProcessStatus status = readMeanImageFile();
        if (status != NVDSPREPROCESS_SUCCESS) {
            printf("Failed to read mean image file\n");
            return status;
        }
    }
    /* Create the mean data buffer from per-channel offsets. */
    else if (m_ChannelMeans.size() > 0) {
        /* Make sure the number of offsets are equal to the number of input
         * channels. */
        if ((uint32_t)m_ChannelMeans.size() != m_NetworkSize.channels) {
            printf("Number of offsets(%d) not equal to number of input "
                   "channels(%d)",
                   (int)m_ChannelMeans.size(),
                   m_NetworkSize.channels);
            return NVDSPREPROCESS_CONFIG_FAILED;
        }

        std::vector<float> meanData(
                m_NetworkSize.channels * m_NetworkSize.width * m_NetworkSize.height);
        for (size_t j = 0; j < m_NetworkSize.width * m_NetworkSize.height; j++) {
            for (size_t i = 0; i < m_NetworkSize.channels; i++) {
                meanData[j * m_NetworkSize.channels + i] = m_ChannelMeans[i];
            }
        }
        cudaError_t cudaReturn = cudaMemcpy(
                m_MeanDataBuffer->ptr(),
                meanData.data(),
                meanData.size() * sizeof(float),
                cudaMemcpyHostToDevice);
        if (cudaReturn != cudaSuccess) {
            printf("Failed to copy mean data to mean data cuda buffer(%s)",
                   cudaGetErrorName(cudaReturn));
            return NVDSPREPROCESS_CUDA_ERROR;
        }
    }

    /* Create the cuda stream on which pre-processing jobs will be executed. */
    m_PreProcessStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);
    if (!m_PreProcessStream || !m_PreProcessStream->ptr()) {
        printf("Failed to create preprocessor cudaStream");
        return NVDSPREPROCESS_CUDA_ERROR;
    }

    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus NvDsPreProcessTensorImpl::syncStream() {
    if (m_PreProcessStream) {
        if (cudaSuccess != cudaStreamSynchronize(*m_PreProcessStream))
            return NVDSPREPROCESS_CUDA_ERROR;
    }
    return NVDSPREPROCESS_SUCCESS;
}

// Convert Phase-1 passthrough (NCHW [3,H,W] or legacy HWC [H,W,3]) to HWC
// float interleaved for cv::Mat(H, W, CV_32FC3). Fail closed on unknown layout.
static bool passthrough_to_hwc(const float* src, const unsigned* d, int num_dims,
                               std::vector<float>& hwc, int& out_h, int& out_w) {
    if (src == nullptr || d == nullptr || num_dims < 3) return false;
    const int d0 = (int)d[0], d1 = (int)d[1], d2 = (int)d[2];
    const bool nchw = (d0 == 3 && d2 != 3);  // [3, H, W]
    const bool hwc_layout = (d2 == 3);       // [H, W, 3]
    if (nchw) {
        out_h = d1;
        out_w = d2;
        if (out_h <= 0 || out_w <= 0) return false;
        hwc.resize((size_t)out_h * (size_t)out_w * 3);
        const size_t plane = (size_t)out_h * (size_t)out_w;
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    hwc[((size_t)y * out_w + x) * 3 + c] =
                            src[(size_t)c * plane + (size_t)y * out_w + x];
                }
            }
        }
        return true;
    }
    if (hwc_layout) {
        out_h = d0;
        out_w = d1;
        if (out_h <= 0 || out_w <= 0) return false;
        hwc.assign(src, src + (size_t)out_h * (size_t)out_w * 3);
        return true;
    }
    return false;
}

// Fast HWC-float32 -> CHW-planar write with fused scale.
// Old code: triple loop with .at<Vec3f>(h,w)[c] and (v*255)*m_Scale per pixel.
// New: row pointers, one fused multiply k = 255*m_Scale, memcpy-friendly stride.
static inline void write_planar_fast(const cv::Mat& src, float* dst, int channels, float k) {
    const int H = src.rows, W = src.cols;
    for (int c = 0; c < channels; ++c) {
        float* d = dst + (size_t)c * H * W;
        for (int h = 0; h < H; ++h) {
            const float* s = src.ptr<float>(h);
            size_t base = (size_t)h * W;
            for (int w = 0; w < W; ++w) {
                d[base + w] = s[(size_t)w * 3 + c] * k;
            }
        }
    }
}

NvDsPreProcessStatus NvDsPreProcessTensorImpl::prepare_tensor(
        NvDsPreProcessBatch* batch,
        void*& devBuf) {
    unsigned int batch_size = batch->units.size();

    unsigned out_size[3] = {m_NetworkSize.width, m_NetworkSize.height, m_NetworkSize.channels};
    const float norm_k = 255.0f * m_Scale;  // fused: old (v*255)*m_Scale == v*k

    /* For each frame in the input batch convert/copy to the input binding
     * buffer. */
    for (unsigned int i = 0; i < batch_size; i++) {
        float* outPtr = (float*)devBuf +
                        i * m_NetworkSize.channels * m_NetworkSize.width * m_NetworkSize.height;
        // Set to zeros incase previous usage still has some data there
        cudaMemset(outPtr, 0, out_size[0] * out_size[1] * out_size[2] * sizeof(float));
        /* Input needs to be pre-processed. */
        if (batch->units[i].obj_meta) {
            for (NvDsMetaList* l_user = batch->units[i].obj_meta->obj_user_meta_list;
                 l_user != NULL;
                 l_user = l_user->next) {
                NvDsUserMeta* user_meta = (NvDsUserMeta*)l_user->data;
                if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                    NvDsInferTensorMeta* tensor_meta =
                            (NvDsInferTensorMeta*)user_meta->user_meta_data;
                    if (tensor_meta->unique_id != (guint)m_WpodUniqueID) {
                        continue;
                    }
                    if (tensor_meta->numOutputLayers < 2) {
                        printf("IWPOD Phase-1 requires passthrough output[1]; skipping object\n");
                        break;
                    }
                    const NvDsInferDims& pass_dims = tensor_meta->output_layers_info[1].inferDims;
                    const NvDsInferDims& pred_inf = tensor_meta->output_layers_info[0].inferDims;
                    if (pass_dims.numDims < 3 || pred_inf.numDims < 3) {
                        printf("IWPOD unexpected rank pred=%u pass=%u; skipping\n",
                               pred_inf.numDims, pass_dims.numDims);
                        break;
                    }
                    unsigned* in_size = tensor_meta->output_layers_info[1].inferDims.d;
                    unsigned* pred_dims = tensor_meta->output_layers_info[0].inferDims.d;
                    int vC, vGh, vGw;
                    bool vNCHW, vLogits;
                    if (pred_dims[2] == 8 && pred_dims[0] != 8 && pred_dims[0] != 7) {
                        vNCHW = false;
                        vC = 8;
                        vGh = (int)pred_dims[0];
                        vGw = (int)pred_dims[1];
                        vLogits = false;
                    } else if (pred_dims[0] == 7 || pred_dims[0] == 8) {
                        vNCHW = true;
                        vC = (int)pred_dims[0];
                        vGh = (int)pred_dims[1];
                        vGw = (int)pred_dims[2];
                        vLogits = (vC == 7);
                    } else {
                        printf("IWPOD unknown lpd_pred layout [%u,%u,%u]; skipping\n",
                               pred_dims[0], pred_dims[1], pred_dims[2]);
                        break;
                    }
                    float* wpod = (float*)tensor_meta->out_buf_ptrs_host[0];
                    float* cropped_image = (float*)tensor_meta->out_buf_ptrs_host[1];
                    if (!wpod || !cropped_image) {
                        printf("IWPOD missing host tensor pointers; skipping\n");
                        break;
                    }
                    std::vector<float> hwc;
                    int pass_h = 0, pass_w = 0;
                    if (!passthrough_to_hwc(cropped_image, in_size, (int)pass_dims.numDims,
                                            hwc, pass_h, pass_w)) {
                        printf("IWPOD passthrough layout not NCHW[3,H,W] or HWC[H,W,3]; skipping\n");
                        break;
                    }
                    float pts[8];
                    cv::Mat image = cv::Mat(pass_h, pass_w, CV_32FC3, hwc.data()).clone();
                    Mat image_output =
                            cv::Mat(out_size[1], out_size[0], CV_32FC3, cv::Scalar(0, 0, 0));
                    const size_t ocr_elems =
                            (size_t)out_size[0] * out_size[1] * out_size[2];
                    if (cpuPtr && ptrSize < ocr_elems) {
                        delete[] cpuPtr;
                        cpuPtr = nullptr;
                    }
                    if (!cpuPtr) {
                        cpuPtr = new float[ocr_elems];
                        ptrSize = ocr_elems;
                    }
                    if (!cpuPtr) {
                        printf("COULD NOT GET MEMORY FOR TEMP BUFFER SKIPPING PLATE READ\n");
                        break;
                    }
                    memset(cpuPtr, 0, ocr_elems * sizeof(float));

                    // WPOD output is RGB. Convert if the network requires a different
                    // format.
                    if (m_NetworkInputFormat == NvDsPreProcessFormat_BGR) {
                        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                    } else if (m_NetworkInputFormat == NvDsPreProcessFormat_GRAY) {
                        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
                    }

                    float confidence = iwpod_v2::reconstructIwpod(
                            &image_output, pts, image, wpod, vC, vGh, vGw, vNCHW,
                            pass_w, pass_h, (int)out_size[0],
                            (int)out_size[1], kNetStride, kSide, kMinConfidence,
                            vLogits, kTopK, kNmsIou);
                    if (confidence < kMinConfidence) {
                        // Set to 0
                        for (int c = 0; c <= 15; c++) {
                            wpod[c] = 0.0;
                        }
                        break;
                    }

                    // This part is about whether to send a vehicle or plate image for
                    // this object
                    int object_id = batch->units[i].obj_meta->object_id;
                    // We store whether we have already sent an image (and the confidence
                    // in that image here)
                    auto& inner_map = image_info_map[batch->units[i].frame_meta->source_id];
                    auto obj_itr = inner_map.find(object_id);
                    // Whether we want to send it
                    bool send_plate = false;
                    bool send_vehicle = false;
                    ImageInfo* info;
                    if (!disable_images) {
                        if (obj_itr != inner_map.end()) {
                            // The object_id exists within the source_id map
                            info = &(obj_itr->second);
                            // If we are far more confidence in this new image and have sent 1
                            // in the last 4 frame send it
                            if (info->plate_confidence + 0.1 < confidence &&
                                batch->units[i].frame_meta->frame_num - info->last_plate_sent > 4) {
                                send_plate = true;
                            } else if (
                                    info->vehicle_confidence + 0.1 < confidence &&
                                    batch->units[i].frame_meta->frame_num -
                                                    info->last_vehicle_sent >
                                            4) {
                                send_vehicle = true;
                            }
                        } else {
                            // Havent seen this object before
                            // Delete from our map if we have more than 16 stored (they are
                            // likely gone anyways)
                            if (inner_map.size() > 16) {
                                // Erase the first element (which has the lowest object_id)
                                inner_map.erase(inner_map.begin());
                            }

                            info = &(image_info_map[batch->units[i].frame_meta->source_id]
                                                   [object_id]);
                            info->last_plate_sent = 0;
                            info->plate_confidence = 0;
                            info->last_vehicle_sent = 0;
                            info->vehicle_confidence = 0;
                            // Send the plate on the first time we see it
                            send_plate = true;
                        }
                    }
                    std::vector<uchar> jpegData;

                    // This must be before copying the plate to the cpuPtr as it uses the
                    // same memory
                    if (send_vehicle) {
                        // Since image has black bar at the bottom we want to crop that out
                        // so we do that here
                        int blackBarStartRow = -1;
                        for (int r = image.rows - 1; r >= 0; r--) {
                            cv::Vec3b pixel = image.at<cv::Vec3b>(r, image.cols / 2);
                            if (pixel == cv::Vec3b(0, 0, 0)) {
                                blackBarStartRow = r;
                            } else {
                                break;
                            }
                        }
                        // Image pixels are 0.0-1.0 but we need them to be 0-255 so multiple
                        // that here
                        image = image * 255;
                        // Convert to 8bit instead of float
                        image.convertTo(image, CV_8UC3);
                        // Sometimes it doesnt work out cropping so we just do the whole
                        // image with black bars
                        if (blackBarStartRow <= 10) {
                            blackBarStartRow = image.rows;
                        }
                        // Crop the image
                        cv::Rect regionOfInterest(0, 0, image.cols, blackBarStartRow);
                        cv::Mat croppedImage = image(regionOfInterest);
                        // Calculate the scaling factor to resize the image to have a
                        // maximum height of 50 pixels
                        double scale = std::min(50.0 / croppedImage.rows, 1.0);
                        if (scale < 1.0) {
                            cv::Size newSize(croppedImage.cols * scale, croppedImage.rows * scale);
                            cv::resize(croppedImage, croppedImage, newSize);
                        }
                        cv::imencode(".jpg", croppedImage, jpegData);
                        info->vehicle_confidence = confidence;
                        info->last_vehicle_sent = batch->units[i].frame_meta->frame_num;
                    }

                    // The number of channels depends on the final network format (e.g., 1
                    // for GRAY, 3 for RGB/BGR)
                    int channels_to_copy = image_output.channels();
                    if (m_NetworkInputFormat == NvDsPreProcessFormat_GRAY) {
                        channels_to_copy = 1;
                    }

                    // FAST PATH (rewritten): planar write via row pointers.
                    // Old code did channels×rows×cols .at<Vec3f> lookups here.
                    if (channels_to_copy == 3 && image_output.isContinuous()) {
                        write_planar_fast(image_output, cpuPtr, 3, norm_k);
                    } else {
                        // Slow generic path (GRAY or non-continuous, same values).
                        for (int c = 0; c < channels_to_copy; c++) {
                            for (int h = 0; h < image_output.rows; h++) {
                                for (int w = 0; w < image_output.cols; w++) {
                                    int index = c * image_output.cols * image_output.rows +
                                                h * image_output.cols + w;
                                    float pixel_val;
                                    if (channels_to_copy == 1) {
                                        pixel_val = image_output.at<float>(h, w);
                                    } else {
                                        pixel_val = image_output.at<cv::Vec3f>(h, w)[c];
                                    }
                                    cpuPtr[index] = pixel_val * norm_k;
                                }
                            }
                        }
                    }
                    // For some reason this has to be after copying to the cpuPtr
                    if (send_plate) {
                        // Image pixels are 0.0-1.0 but we need them to be 0-255 so multiple
                        // that here
                        image_output = image_output * 255;
                        // Convert from float to integers
                        image_output.convertTo(image_output, CV_8UC3);
                        // Resize to a quart of the original size (256/4, 96/4) to save
                        // memory
                        cv::Size newSize(image_output.cols / 2, image_output.rows / 2);
                        cv::resize(image_output, image_output, newSize);
                        cv::imencode(".jpg", image_output, jpegData);
                        info->plate_confidence = confidence;
                        info->last_plate_sent = batch->units[i].frame_meta->frame_num;
                    }

                    // Copy the converted image to the output
                    cudaMemcpy(
                            outPtr,
                            cpuPtr,
                            out_size[0] * out_size[1] * out_size[2] * sizeof(float),
                            cudaMemcpyHostToDevice);
                    // We replace the first 8 values of the wpod raw output for display
                    for (int c = 0; c < 8; c++) {
                        wpod[c] = pts[c];
                    }

                    wpod[8] = confidence;
                    // So we know the size of the output/ whether there is data there
                    wpod[9] = jpegData.size();
                    // So we know whether this is a plate or a vehicle
                    wpod[10] = send_plate;
                    // Since the memory is just a pointer to some memory we can recast it
                    // to a uchar (1 byte) and copy the image data there
                    uchar* image_data = (uchar*)tensor_meta->out_buf_ptrs_host[1];
                    // Copy the image into the existing wpod buffer as uchar
                    // Check to see that our image will fit since we have scaled images
                    // this should always fit But good to check anyways
                    if (0 < jpegData.size() &&
                        jpegData.size() * sizeof(uchar) <
                                (in_size[0] * in_size[1] * in_size[2]) * sizeof(float)) {
                        memcpy(image_data, jpegData.data(), jpegData.size() * sizeof(uchar));
                        // Flash the bbox to white for vehicle image and blue for plate
                        // image
                        batch->units[i].obj_meta->rect_params.has_bg_color = true;
                        batch->units[i].obj_meta->rect_params.bg_color.red =
                                send_vehicle ? 1.0 : 0.0;
                        batch->units[i].obj_meta->rect_params.bg_color.blue = 1.0;
                        batch->units[i].obj_meta->rect_params.bg_color.green =
                                send_vehicle ? 1.0 : 0.0;
                        ;
                        batch->units[i].obj_meta->rect_params.bg_color.alpha = 0.3;
                    } else {
                        // Signal for whether there is no image data stored
                        wpod[9] = 0;
                    }
                    break;
                }
            }
        }
    }

    return NVDSPREPROCESS_SUCCESS;
}

extern "C" NvDsPreProcessStatus normalization_mean_subtraction_impl_initialize(
        CustomMeanSubandNormParams* custom_params,
        NvDsPreProcessTensorParams* tensor_params,
        std::unique_ptr<NvDsPreProcessTensorImpl>& m_Preprocessor,
        int unique_id) {
    if (tensor_params->network_input_order == NvDsPreProcessNetworkInputOrder_kNCHW) {
        custom_params->networkSize.channels = tensor_params->network_input_shape[1];
        custom_params->networkSize.height = tensor_params->network_input_shape[2];
        custom_params->networkSize.width = tensor_params->network_input_shape[3];
    } else if (tensor_params->network_input_order == NvDsPreProcessNetworkInputOrder_kNHWC) {
        custom_params->networkSize.height = tensor_params->network_input_shape[1];
        custom_params->networkSize.width = tensor_params->network_input_shape[2];
        custom_params->networkSize.channels = tensor_params->network_input_shape[3];
    } else {
        printf("network-input-order = %d not supported\n", tensor_params->network_input_order);
    }

    switch (tensor_params->network_color_format) {
        case NvDsPreProcessFormat_RGB:
        case NvDsPreProcessFormat_BGR:
            if (custom_params->networkSize.channels != 3) {
                printf("RGB/BGR input format specified but network input channels is not "
                       "3\n");
                return NVDSPREPROCESS_CONFIG_FAILED;
            }
            break;
        case NvDsPreProcessFormat_GRAY:
            if (custom_params->networkSize.channels != 1) {
                printf("GRAY input format specified but network input channels is not "
                       "1.\n");
                return NVDSPREPROCESS_CONFIG_FAILED;
            }
            break;
        case NvDsPreProcessFormat_Tensor:
            break;
        default:
            printf("Unknown input format\n");
            return NVDSPREPROCESS_CONFIG_FAILED;
    }

    std::unique_ptr<NvDsPreProcessTensorImpl> tensor_impl =
            std::make_unique<NvDsPreProcessTensorImpl>(
                    custom_params->networkSize,
                    tensor_params->network_color_format,
                    unique_id);
    assert(tensor_impl);
    tensor_impl->setWpodUniqueID(custom_params->wpod_unique_id);

    if (custom_params->pixel_normalization_factor > 0.0f) {
        std::vector<float> offsets = custom_params->offsets;
        if (!tensor_impl->setScaleOffsets(custom_params->pixel_normalization_factor, offsets)) {
            printf("Preprocessor set scale and offsets failed.\n");
            return NVDSPREPROCESS_CONFIG_FAILED;
        }
    }

    if (!custom_params->meanImageFilePath.empty() &&
        !tensor_impl->setMeanFile(custom_params->meanImageFilePath)) {
        printf("Cannot access mean image file %s\n", custom_params->meanImageFilePath.c_str());
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    if (!tensor_impl->setInputOrder(tensor_params->network_input_order)) {
        printf("Cannot set network order %s\n",
               (tensor_params->network_input_order == 0) ? "NCHW" : "NHWC");
    }

    NvDsPreProcessStatus status = tensor_impl->allocateResource();
    if (status != NVDSPREPROCESS_SUCCESS) {
        printf("preprocessor allocate resource failed\n");
        return status;
    }

    m_Preprocessor = std::move(tensor_impl);
    return NVDSPREPROCESS_SUCCESS;
}
