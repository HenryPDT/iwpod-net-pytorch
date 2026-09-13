/*
 * nvdspreprocess_iwpod_impl.cpp — IWPOD letterbox preprocess plugin TU.
 *
 * Linked into libcustom_iwpod_ocr_preprocess.so (Makefile). Classic WPOD
 * passthrough is libcustom_wpod_ocr_preprocess.so from nvdspreprocess_impl.cpp.
 * OCR still reads the NCHW plate tensor written here. LPD SGIE emits a single
 * lpd_pred; vehicle pixels come from NvBufSurface. JPEG telemetry is packed
 * after the first 16 floats of lpd_pred.
 *
 * Contracts: tensor_output[0..7] quad pixels in letterbox space,
 * wpod[8]=confidence, wpod[9]=jpeg bytes, wpod[10]=is-plate, OCR tensor
 * NCHW float [0,1] via pixel_val*255*m_Scale (fused to one multiply).
 */

#include "nvdspreprocess_impl.h"

#include <cuda.h>
#include <dlfcn.h>
#include <unistd.h>
#include <gst/gst.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvdspreprocess_conversion.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "iwpod_reconstruct.h"

using namespace cv;
using namespace std;

// ---- IWPOD decode parameters (must match training + export) ---------------
static constexpr double kNetStride = 16.0;
static constexpr double kSide = ((208.0 + 40.0) / 2.0) / kNetStride;  // 7.75
static constexpr float kMinConfidence = 0.3f;  // mirrors runtime wpod_threshold default
static constexpr int kTopK = 1;                // 1 reproduces old max-only behavior
static constexpr double kNmsIou = 0.25;
static constexpr int kJpegFloatOffset = 16;
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

// Match nvinfer GST_ROUND_UP_2 / GST_ROUND_DOWN_2 used for SGIE object crops.
static inline unsigned round_up2(unsigned v) {
    return (v + 1u) & ~1u;
}
static inline unsigned round_down2(unsigned v) {
    return v & ~1u;
}

static bool ensure_letterbox_surf(
        NvBufSurface*& surf,
        int& cur_w,
        int& cur_h,
        int net_w,
        int net_h,
        uint32_t gpu_id) {
    if (surf && cur_w == net_w && cur_h == net_h) {
        return true;
    }
    if (surf) {
        NvBufSurfaceDestroy(surf);
        surf = nullptr;
        cur_w = 0;
        cur_h = 0;
    }
    NvBufSurfaceCreateParams create_params = {};
    create_params.gpuId = gpu_id;
    create_params.width = net_w;
    create_params.height = net_h;
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;
#ifdef __aarch64__
    create_params.memType = NVBUF_MEM_DEFAULT;
#else
    create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
#endif
    if (NvBufSurfaceCreate(&surf, 1, &create_params) != 0 || !surf) {
        printf("IWPOD failed to allocate %dx%d letterbox surface\n", net_w, net_h);
        surf = nullptr;
        return false;
    }
    cur_w = net_w;
    cur_h = net_h;
    return true;
}

// Crop vehicle bbox from the frame surface and letterbox top-left into net_w x
// net_h RGB float [0,1], matching nvinfer maintain-aspect-ratio without
// symmetric-padding (right/bottom pad).
static bool letterbox_vehicle_rgb(
        NvBufSurface* in_surf,
        NvBufSurfaceParams* src_params,
        const NvOSD_RectParams& bbox,
        NvBufSurface*& letterbox_surf,
        int& letterbox_w,
        int& letterbox_h,
        int net_w,
        int net_h,
        cudaStream_t stream,
        cv::Mat& rgb01) {
    if (!in_surf || !src_params || net_w <= 0 || net_h <= 0) {
        return false;
    }
    if (!ensure_letterbox_surf(
                letterbox_surf,
                letterbox_w,
                letterbox_h,
                net_w,
                net_h,
                in_surf->gpuId)) {
        return false;
    }

    unsigned src_left = round_up2((unsigned)bbox.left);
    unsigned src_top = round_up2((unsigned)bbox.top);
    unsigned src_width = round_down2((unsigned)bbox.width);
    unsigned src_height = round_down2((unsigned)bbox.height);
    if (src_left >= src_params->width || src_top >= src_params->height) {
        return false;
    }
    if (src_left + src_width > src_params->width) {
        src_width = src_params->width - src_left;
        src_width = round_down2(src_width);
    }
    if (src_top + src_height > src_params->height) {
        src_height = src_params->height - src_top;
        src_height = round_down2(src_height);
    }
    if (src_width < 2 || src_height < 2) {
        return false;
    }

    unsigned dest_width = (unsigned)net_w;
    unsigned dest_height = (unsigned)net_h;
    const double hdest = (double)net_w * src_height / (double)src_width;
    const double wdest = (double)net_h * src_width / (double)src_height;
    if (hdest <= (double)net_h) {
        dest_width = (unsigned)net_w;
        dest_height = (unsigned)hdest;
    } else {
        dest_width = (unsigned)wdest;
        dest_height = (unsigned)net_h;
    }
    if (dest_width < 1) dest_width = 1;
    if (dest_height < 1) dest_height = 1;
    if (dest_width > (unsigned)net_w) dest_width = (unsigned)net_w;
    if (dest_height > (unsigned)net_h) dest_height = (unsigned)net_h;

    NvBufSurface src_wrap = {};
    src_wrap.gpuId = in_surf->gpuId;
    src_wrap.batchSize = 1;
    src_wrap.numFilled = 1;
    src_wrap.memType = in_surf->memType;
    src_wrap.isContiguous = in_surf->isContiguous;
    src_wrap.surfaceList = src_params;

    NvBufSurfTransformConfigParams cfg = {};
    cfg.compute_mode = NvBufSurfTransformCompute_GPU;
    cfg.gpu_id = in_surf->gpuId;
    cfg.cuda_stream = stream;
    NvBufSurfTransform_Error err = NvBufSurfTransformSetSessionParams(&cfg);
    if (err != NvBufSurfTransformError_Success) {
        printf("IWPOD NvBufSurfTransformSetSessionParams failed (%d)\n", (int)err);
        return false;
    }
    NvBufSurfaceMemSet(letterbox_surf, 0, 0, 0);

    NvBufSurfTransformRect src_rect = {src_top, src_left, src_width, src_height};
    NvBufSurfTransformRect dst_rect = {0, 0, dest_width, dest_height};
    NvBufSurfTransformParams tp = {};
    tp.src_rect = &src_rect;
    tp.dst_rect = &dst_rect;
    tp.transform_flag = NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
                        NVBUFSURF_TRANSFORM_CROP_DST;
    tp.transform_flip = NvBufSurfTransform_None;
    tp.transform_filter = NvBufSurfTransformInter_Default;
    err = NvBufSurfTransform(&src_wrap, letterbox_surf, &tp);
    if (err != NvBufSurfTransformError_Success) {
        printf("IWPOD NvBufSurfTransform failed (%d)\n", (int)err);
        return false;
    }

    if (NvBufSurfaceMap(letterbox_surf, 0, 0, NVBUF_MAP_READ) != 0) {
        printf("IWPOD letterbox NvBufSurfaceMap failed\n");
        return false;
    }
    NvBufSurfaceSyncForCpu(letterbox_surf, 0, 0);
    NvBufSurfaceParams& dst = letterbox_surf->surfaceList[0];
    if (!dst.mappedAddr.addr[0]) {
        NvBufSurfaceUnMap(letterbox_surf, 0, 0);
        printf("IWPOD letterbox mapped address is null\n");
        return false;
    }
    cv::Mat rgba(net_h, net_w, CV_8UC4, dst.mappedAddr.addr[0], dst.pitch);
    cv::Mat rgb8;
    cv::cvtColor(rgba, rgb8, cv::COLOR_RGBA2RGB);
    rgb8.convertTo(rgb01, CV_32FC3, 1.0 / 255.0);
    NvBufSurfaceUnMap(letterbox_surf, 0, 0);
    return !rgb01.empty();
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

    GstMapInfo in_map = {};
    NvBufSurface* in_surf = nullptr;
    bool mapped_in = false;
    if (batch->inbuf && gst_buffer_map(batch->inbuf, &in_map, GST_MAP_READ)) {
        mapped_in = true;
        in_surf = (NvBufSurface*)in_map.data;
    }

    cudaStream_t stream = m_PreProcessStream ? m_PreProcessStream->ptr() : nullptr;

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
                    if (tensor_meta->num_output_layers < 1) {
                        printf("IWPOD missing lpd_pred output; skipping object\n");
                        break;
                    }
                    const NvDsInferDims& pred_inf = tensor_meta->output_layers_info[0].inferDims;
                    if (pred_inf.numDims < 3) {
                        printf("IWPOD unexpected lpd_pred rank=%u; skipping\n", pred_inf.numDims);
                        break;
                    }
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
                               pred_dims[0],
                               pred_dims[1],
                               pred_dims[2]);
                        break;
                    }
                    float* wpod = (float*)tensor_meta->out_buf_ptrs_host[0];
                    if (!wpod) {
                        printf("IWPOD missing host tensor pointer; skipping\n");
                        break;
                    }
                    const int net_w = vGw * (int)kNetStride;
                    const int net_h = vGh * (int)kNetStride;
                    NvBufSurfaceParams* src_params = batch->units[i].input_surf_params;
                    if (!in_surf || !src_params) {
                        printf("IWPOD missing NvBufSurface for vehicle crop; skipping\n");
                        break;
                    }
                    cv::Mat image;
                    if (!letterbox_vehicle_rgb(
                                in_surf,
                                src_params,
                                batch->units[i].obj_meta->rect_params,
                                m_LetterboxSurf,
                                m_LetterboxW,
                                m_LetterboxH,
                                net_w,
                                net_h,
                                stream,
                                image)) {
                        printf("IWPOD letterbox crop failed; skipping object\n");
                        break;
                    }
                    float pts[8];
                    Mat image_output =
                            cv::Mat(out_size[1], out_size[0], CV_32FC3, cv::Scalar(0, 0, 0));
                    const size_t ocr_elems = (size_t)out_size[0] * out_size[1] * out_size[2];
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

                    // IWPOD crop is RGB. Convert if the OCR network requires a different
                    // format.
                    if (m_NetworkInputFormat == NvDsPreProcessFormat_BGR) {
                        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                    } else if (m_NetworkInputFormat == NvDsPreProcessFormat_GRAY) {
                        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
                    }

                    float confidence = iwpod::reconstructIwpod(
                            &image_output,
                            pts,
                            image,
                            wpod,
                            vC,
                            vGh,
                            vGw,
                            vNCHW,
                            net_w,
                            net_h,
                            (int)out_size[0],
                            (int)out_size[1],
                            kNetStride,
                            kSide,
                            kMinConfidence,
                            vLogits,
                            kTopK,
                            kNmsIou);
                    if (confidence < kMinConfidence) {
                        for (int c = 0; c <= 15; c++) {
                            wpod[c] = 0.0;
                        }
                        break;
                    }

                    int object_id = batch->units[i].obj_meta->object_id;
                    auto& inner_map = image_info_map[batch->units[i].frame_meta->source_id];
                    auto obj_itr = inner_map.find(object_id);
                    bool send_plate = false;
                    bool send_vehicle = false;
                    ImageInfo* info = nullptr;
                    if (!disable_images) {
                        if (obj_itr != inner_map.end()) {
                            info = &(obj_itr->second);
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
                            if (inner_map.size() > 16) {
                                inner_map.erase(inner_map.begin());
                            }
                            info = &(image_info_map[batch->units[i].frame_meta->source_id]
                                                   [object_id]);
                            info->last_plate_sent = 0;
                            info->plate_confidence = 0;
                            info->last_vehicle_sent = 0;
                            info->vehicle_confidence = 0;
                            send_plate = true;
                        }
                    }
                    std::vector<uchar> jpegData;

                    if (send_vehicle) {
                        cv::Mat vehicle8;
                        image.convertTo(vehicle8, CV_8UC3, 255.0);
                        int blackBarStartRow = vehicle8.rows;
                        for (int r = vehicle8.rows - 1; r >= 0; r--) {
                            cv::Vec3b pixel = vehicle8.at<cv::Vec3b>(r, vehicle8.cols / 2);
                            if (pixel == cv::Vec3b(0, 0, 0)) {
                                blackBarStartRow = r;
                            } else {
                                break;
                            }
                        }
                        if (blackBarStartRow <= 10) {
                            blackBarStartRow = vehicle8.rows;
                        }
                        cv::Rect regionOfInterest(0, 0, vehicle8.cols, blackBarStartRow);
                        cv::Mat croppedImage = vehicle8(regionOfInterest);
                        double scale = std::min(50.0 / croppedImage.rows, 1.0);
                        if (scale < 1.0) {
                            cv::Size newSize(croppedImage.cols * scale, croppedImage.rows * scale);
                            cv::resize(croppedImage, croppedImage, newSize);
                        }
                        cv::imencode(".jpg", croppedImage, jpegData);
                        info->vehicle_confidence = confidence;
                        info->last_vehicle_sent = batch->units[i].frame_meta->frame_num;
                    }

                    int channels_to_copy = image_output.channels();
                    if (m_NetworkInputFormat == NvDsPreProcessFormat_GRAY) {
                        channels_to_copy = 1;
                    }

                    if (channels_to_copy == 3 && image_output.isContinuous()) {
                        write_planar_fast(image_output, cpuPtr, 3, norm_k);
                    } else {
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
                    if (send_plate) {
                        image_output = image_output * 255;
                        image_output.convertTo(image_output, CV_8UC3);
                        cv::Size newSize(image_output.cols / 2, image_output.rows / 2);
                        cv::resize(image_output, image_output, newSize);
                        cv::imencode(".jpg", image_output, jpegData);
                        info->plate_confidence = confidence;
                        info->last_plate_sent = batch->units[i].frame_meta->frame_num;
                    }

                    cudaMemcpy(
                            outPtr,
                            cpuPtr,
                            out_size[0] * out_size[1] * out_size[2] * sizeof(float),
                            cudaMemcpyHostToDevice);
                    for (int c = 0; c < 8; c++) {
                        wpod[c] = pts[c];
                    }

                    wpod[8] = confidence;
                    wpod[9] = jpegData.size();
                    wpod[10] = send_plate;
                    const unsigned long pred_n =
                            (unsigned long)pred_dims[0] * pred_dims[1] * pred_dims[2];
                    const unsigned long jpeg_cap =
                            pred_n > (unsigned long)kJpegFloatOffset
                                    ? (pred_n - (unsigned long)kJpegFloatOffset) * sizeof(float)
                                    : 0;
                    uchar* image_data = (uchar*)(wpod + kJpegFloatOffset);
                    if (0 < jpegData.size() && jpegData.size() * sizeof(uchar) < jpeg_cap) {
                        memcpy(image_data, jpegData.data(), jpegData.size() * sizeof(uchar));
                        batch->units[i].obj_meta->rect_params.has_bg_color = true;
                        batch->units[i].obj_meta->rect_params.bg_color.red =
                                send_vehicle ? 1.0 : 0.0;
                        batch->units[i].obj_meta->rect_params.bg_color.blue = 1.0;
                        batch->units[i].obj_meta->rect_params.bg_color.green =
                                send_vehicle ? 1.0 : 0.0;
                        batch->units[i].obj_meta->rect_params.bg_color.alpha = 0.3;
                    } else {
                        wpod[9] = 0;
                    }
                    break;
                }
            }
        }
    }

    if (mapped_in) {
        gst_buffer_unmap(batch->inbuf, &in_map);
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
