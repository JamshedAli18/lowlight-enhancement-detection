# enhance.py - enhance a low-light image with any of the benchmarked methods (CPU only)
import argparse

import cv2
import numpy as np


def gamma(img):
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean() / 255
    g = np.log(0.5) / np.log(np.clip(y, 1e-3, 0.99))
    lut = (np.linspace(0, 1, 256) ** g * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def he(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _big_blur(img, sigma):
    if sigma < 30:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 4), (w, h), interpolation=cv2.INTER_LINEAR)


def msrcr(img, sigmas=(15, 80, 250), alpha=125.0, beta=46.0, G=5.0, b=25.0):
    x = img.astype(np.float32) + 1.0
    retinex = sum(np.log10(x) - np.log10(_big_blur(x, s)) for s in sigmas) / len(sigmas)
    crf = beta * (np.log10(alpha * x) - np.log10(x.sum(axis=2, keepdims=True)))
    out = G * (retinex * crf + b)
    for c in range(3):
        lo, hi = np.percentile(out[..., c], (1, 99))
        out[..., c] = (out[..., c] - lo) / max(hi - lo, 1e-6) * 255
    return out.clip(0, 255).astype(np.uint8)


def gamma_nlm(img):
    return cv2.fastNlMeansDenoisingColored(gamma(img), None, h=7, hColor=7,
                                           templateWindowSize=7, searchWindowSize=21)


def zero_dce(img, model="models/zero_dce.onnx", quarter=True):
    import onnxruntime as ort
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    H, W = img.shape[:2]
    src = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA) if quarter else img
    x = src.transpose(2, 0, 1)[None].astype(np.float32) / 255
    A = sess.run(["curves"], {"image": x})[0][0]
    if quarter:
        A = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in A])
    y = img.transpose(2, 0, 1).astype(np.float32) / 255
    for k in range(8):
        r = A[3 * k:3 * k + 3]
        y = y + r * (y * y - y)
    return (y.transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)


METHODS = {"gamma": gamma, "he": he, "clahe": clahe, "msrcr": msrcr, "gamma_nlm": gamma_nlm,
           "zero_dce": lambda im: zero_dce(im, quarter=False), "zero_dce_quarter": zero_dce}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--method", default="zero_dce_quarter", choices=list(METHODS))
    p.add_argument("--out", default="enhanced.png")
    args = p.parse_args()
    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    out = METHODS[args.method](rgb)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"{args.method}: saved {args.out}")