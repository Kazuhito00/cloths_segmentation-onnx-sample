#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import cv2
import numpy as np
import onnxruntime  # type: ignore


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/cloths_segmentation.onnx',
    )
    parser.add_argument("--input_resize_rate", type=float, default=1.0)

    args: argparse.Namespace = parser.parse_args()

    return args


def get_padding(h: int, w: int, unit_size: int) -> tuple[int, int, int, int]:
    new_w: int = (w // unit_size + 1) * unit_size if w % unit_size != 0 else w
    new_h: int = (h // unit_size + 1) * unit_size if h % unit_size != 0 else h
    if h >= new_h:
        top: int = 0
        bottom: int = 0
    else:
        dh: int = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left: int = 0
        right: int = 0
    else:
        dw: int = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w

    return (left, top, right, bottom)


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
    input_resize_rate: float,
    unit_size: int = 32,
) -> np.ndarray:
    original_image_width: int = image.shape[1]
    original_image_height: int = image.shape[0]
    input_image: np.ndarray = cv2.resize(
        image,
        dsize=None,
        fx=input_resize_rate,
        fy=input_resize_rate,
    )

    # Pre-process: Padding
    image_width: int = input_image.shape[1]
    image_height: int = input_image.shape[0]
    padding: tuple[int, int, int, int] = get_padding(image_height,
                                                     image_width,
                                                     unit_size=unit_size)

    left, top, right, bottom = padding[0], padding[1], padding[2], padding[3]
    pad_value: int = 0
    input_image = cv2.copyMakeBorder(
        input_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )  # type: ignore
    # Pre-process: BGR to RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Pre-process: Normalize, Convert to BCHW format
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    input_image = ((input_image.astype(np.float32) / 255) - mean) / std
    input_image = np.array(input_image, dtype=np.float32)
    input_image = input_image.transpose(2, 0, 1)
    input_image = input_image.reshape(
        -1,
        3,
        input_image.shape[1],
        input_image.shape[2],
    )

    # Inference
    input_name: str = onnx_session.get_inputs()[0].name
    onnx_result: list[np.ndarray] = onnx_session.run(
        None,
        {input_name: input_image},
    )

    # Post-process: Remove padding, Resize
    mask: np.ndarray = np.array(onnx_result[0][0][0])
    mask = mask[top:top + image_height, left:left + image_width]
    mask = (mask > 0).astype('uint8')
    mask = np.clip(mask * 255, 0, 255)

    mask = cv2.resize(
        mask,
        dsize=(original_image_width, original_image_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask


def main() -> None:
    args: argparse.Namespace = get_args()
    cap_device: int | str = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    if args.movie is not None:
        cap_device = args.movie
    image_path: str | None = args.image

    model_path: str = args.model
    input_resize_rate: float = args.input_resize_rate

    # Initialize video capture
    cap: cv2.VideoCapture | None = None
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)
        if cap is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model
    onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    print('Providers:', onnx_session.get_providers())

    if image_path is not None:
        image: np.ndarray = cv2.imread(image_path)
        debug_image: np.ndarray = copy.deepcopy(image)

        # Warm up
        _ = run_inference(
            onnx_session,
            image,
            input_resize_rate,
        )

        start_time: float = time.time()

        # Inference execution
        mask: np.ndarray = run_inference(
            onnx_session,
            image,
            input_resize_rate,
        )

        elapsed_time: float = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            mask,
        )

        cv2.imshow('Cloths Segmentation Demo Input', image)
        cv2.imshow('Cloths Segmentation Demo Result', debug_image)
        cv2.waitKey(0)
    else:
        while cap is not None:
            start_time = time.time()

            # Capture read
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)

            # Inference execution
            mask = run_inference(
                onnx_session,
                frame,
                input_resize_rate,
            )

            elapsed_time = time.time() - start_time

            # Draw
            debug_image = draw_debug(
                debug_image,
                elapsed_time,
                mask,
            )

            key: int = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            cv2.imshow('Cloths Segmentation Demo Input', frame)
            cv2.imshow('Cloths Segmentation Demo Result', debug_image)

        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image: np.ndarray,
    elapsed_time: float,
    mask: np.ndarray,
) -> np.ndarray:
    # Create Mask Image
    temp_mask: np.ndarray = np.stack((mask, ) * 3, axis=-1).astype('float32')
    temp_mask = np.where(temp_mask != 0, 0, 1)

    # Combine mask image and original image
    debug_image: np.ndarray = copy.deepcopy(image)
    bg_image: np.ndarray = np.zeros(debug_image.shape, dtype=np.uint8)
    bg_image[:] = [0, 255, 0]
    mask_image: np.ndarray = np.where(temp_mask, debug_image, bg_image)

    # Combine as a semi-transparent image
    debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    # Inference elapsed time
    cv2.putText(
        debug_image,
        "Elapsed Time: " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()
