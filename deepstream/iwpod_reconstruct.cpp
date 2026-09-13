// iwpod_reconstruct.cpp — IWPOD grid decoder for nvdspreprocess_iwpod_impl.cpp
//
// WHAT: decodes the IWPOD raw tensor (NCHW [7,Gh,Gw]: ch0 = raw logits,
//       ch1..6 = affine) into plate quads + warpPerspective rectification.
//       Also accepts the legacy WPOD layout (NHWC [Gh,Gw,8]: ch0 = prob,
//       ch1 = bg, ch2..7 = affine).
//
// WHY REPLACE reconstructSingle: the old function has 5 latent bugs this fixes:
//   1. x/y transposed indexing (wpod[x*W*D + y*D] treats NHWC as column-major;
//      harmless only because the grid is square 25x25, wrong for 384px -> 24x24
//      only if non-square ever used — this indexes both layouts correctly).
//   2. Affines copy reads Affines[x][y] but consumes Affines[y][x].
//   3. abs() on doubles (int truncation) in the degenerate-size gate -> fabs.
//   4. VLA double Affines[W][H][D-2] on stack -> std::vector.
//   5. Hardcoded stride/side/threshold -> reconstructIwpod parameters
//      (compile-time in the full TU; not DeepStream [user-configs]).
//
// Linked into libcustom_iwpod_ocr_preprocess.so with nvdspreprocess_iwpod_impl.cpp.
// tensor_output[0..7] + return value keep the LPR.cpp quad/confidence contract.
// JPEG packing lives in the impl TU (after 16 floats of lpd_pred).

#include "iwpod_reconstruct.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

namespace iwpod {

struct ScoredQuad {
    double qx[4], qy[4];
    double conf;
};

static inline double sigmoid(double v) {
    return 1.0 / (1.0 + std::exp(-v));
}

static inline double aabb_iou(double tlx1, double tly1, double brx1, double bry1,
                              double tlx2, double tly2, double brx2, double bry2) {
    const double iw = std::max(0.0, std::min(brx1, brx2) - std::max(tlx1, tlx2));
    const double ih = std::max(0.0, std::min(bry1, bry2) - std::max(tly1, tly2));
    const double inter = iw * ih;
    const double u = (brx1 - tlx1) * (bry1 - tly1) + (brx2 - tlx2) * (bry2 - tly2) - inter;
    return u > 1e-9 ? inter / u : 0.0;
}

// --- grid-decoder internals ------------------------------------------------
static const double kBase[3][4] = {
    {-0.5, 0.5, 0.5, -0.5}, {-0.5, -0.5, 0.5, 0.5}, {1.0, 1.0, 1.0, 1.0}};

// Decode one grid into scored quads (input-pixel space). Grid dims set the
// MN normalization, so any stride-16 input size shares this code path.
static void collectGrid(std::vector<ScoredQuad>& cands, const float* pred,
                        int Gh, int Gw, bool is_nchw, int C,
                        int in_w, int in_h, double side, double min_probability,
                        bool from_logits) {
    int conf_c, aff_c0;
    if (is_nchw && C == 7) {
        conf_c = 0;
        aff_c0 = 1;
    } else if (!is_nchw && C == 8) {
        conf_c = 0;
        aff_c0 = 2;  // skip bg channel (legacy NHWC)
    } else {
        return;  // unknown layout: refuse (fail-safe, no crash)
    }
    auto at = [&](int c, int y, int x) -> double {
        return is_nchw ? pred[(c * Gh + y) * Gw + x] : pred[(y * Gw + x) * C + c];
    };
    const double MNx = (double)Gw, MNy = (double)Gh;
    const double scale_gate = (double)std::max(in_w, in_h) / 400.0;  // old 30x10 @400px
    for (int y = 0; y < Gh; ++y) {
        for (int x = 0; x < Gw; ++x) {
            double raw = at(conf_c, y, x);
            const double conf = from_logits ? sigmoid(raw) : raw;
            if (conf < min_probability) continue;
            double a[6];
            for (int k = 0; k < 6; ++k) a[k] = at(aff_c0 + k, y, x);
            const double A[2][3] = {
                {std::max(a[0], 0.0), a[1], a[2]}, {a[3], std::max(a[4], 0.0), a[5]}};
            ScoredQuad q;
            bool neg = false;
            for (int j = 0; j < 4; ++j) {
                const double rx = (A[0][0] * kBase[0][j] + A[0][1] * kBase[1][j] +
                                   A[0][2] * kBase[2][j]) *
                                          side +
                                  (x + 0.5);
                const double ry = (A[1][0] * kBase[0][j] + A[1][1] * kBase[1][j] +
                                   A[1][2] * kBase[2][j]) *
                                          side +
                                  (y + 0.5);
                q.qx[j] = rx / MNx * in_w;
                q.qy[j] = ry / MNy * in_h;
                if (q.qx[j] < 0 || q.qy[j] < 0) {
                    neg = true;
                    break;
                }
            }
            if (neg) continue;
            double tlx = q.qx[0], tly = q.qy[0], brx = q.qx[0], bry = q.qy[0];
            for (int j = 1; j < 4; ++j) {
                tlx = std::min(tlx, q.qx[j]);
                tly = std::min(tly, q.qy[j]);
                brx = std::max(brx, q.qx[j]);
                bry = std::max(bry, q.qy[j]);
            }
            if ((brx - tlx) < 30 * scale_gate || (bry - tly) < 10 * scale_gate) continue;
            q.conf = conf;
            cands.push_back(q);
        }
    }
}

// AABB-NMS + warp of the winner. Returns 0.0 when nothing survives.
static float selectAndWarp(cv::Mat* output, float* tensor_output, const cv::Mat& image,
                           std::vector<ScoredQuad>& cands,
                           int out_w, int out_h, int topk, double nms_iou) {
    if (cands.empty()) return 0.0f;
    std::sort(cands.begin(), cands.end(),
              [](const ScoredQuad& p, const ScoredQuad& q) { return p.conf > q.conf; });

    // quad-NMS over enclosing AABBs (topk=1 reproduces old max-only behavior)
    std::vector<ScoredQuad> kept;
    for (const auto& c : cands) {
        double tlx = c.qx[0], tly = c.qy[0], brx = c.qx[0], bry = c.qy[0];
        for (int j = 1; j < 4; ++j) {
            tlx = std::min(tlx, c.qx[j]);
            tly = std::min(tly, c.qy[j]);
            brx = std::max(brx, c.qx[j]);
            bry = std::max(bry, c.qy[j]);
        }
        bool overlap = false;
        for (const auto& k : kept) {
            double kx0 = k.qx[0], ky0 = k.qy[0], kx1 = k.qx[0], ky1 = k.qy[0];
            for (int j = 1; j < 4; ++j) {
                kx0 = std::min(kx0, k.qx[j]);
                ky0 = std::min(ky0, k.qy[j]);
                kx1 = std::max(kx1, k.qx[j]);
                ky1 = std::max(ky1, k.qy[j]);
            }
            if (aabb_iou(tlx, tly, brx, bry, kx0, ky0, kx1, ky1) > nms_iou) {
                overlap = true;
                break;
            }
        }
        if (!overlap) kept.push_back(c);
        if ((int)kept.size() >= (topk > 0 ? topk : 1)) break;
    }
    if (kept.empty()) return 0.0f;
    const ScoredQuad& best = kept[0];

    std::vector<cv::Point2f> roi(4), dst(4);
    for (int j = 0; j < 4; ++j) {
        roi[j] = cv::Point2f((float)best.qx[j], (float)best.qy[j]);
        tensor_output[j] = (float)best.qx[j];
        tensor_output[j + 4] = (float)best.qy[j];
    }
    dst[0] = {0.f, 0.f};
    dst[1] = {(float)out_w, 0.f};
    dst[2] = {(float)out_w, (float)out_h};
    dst[3] = {0.f, (float)out_h};
    const cv::Mat H = cv::getPerspectiveTransform(roi, dst);
    cv::warpPerspective(image, *output, H, cv::Size(out_w, out_h), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, 0);
    return (float)best.conf;
}

// pred: raw output pointer (host). Layout selected by is_nchw:
//   NCHW: [C,Gh,Gw], C==7 (logit + affine6). from_logits=1.
//   NHWC legacy: [Gh,Gw,C], C==8 (prob + bg + affine6). from_logits=0.
// in_w/in_h: SGIE input (infer-dims) in pixels. out_w/out_h: plate size (256x96).
// Returns detection confidence (0.0 = none). tensor_output[0..3]=xs, [4..7]=ys
// of the winning quad in vehicle-crop pixels (same contract as before).
float reconstructIwpod(cv::Mat* output, float* tensor_output, const cv::Mat& image,
                       const float* pred, int C, int Gh, int Gw, bool is_nchw,
                       int in_w, int in_h, int out_w, int out_h, double net_stride,
                       double side, double min_probability, bool from_logits,
                       int topk, double nms_iou) {
    (void)net_stride;  // grid dims already encode the stride (Gh=H/stride)
    std::vector<ScoredQuad> cands;
    collectGrid(cands, pred, Gh, Gw, is_nchw, C,
                in_w, in_h, side, min_probability, from_logits);
    return selectAndWarp(output, tensor_output, image, cands, out_w, out_h, topk, nms_iou);
}

}  // namespace iwpod
