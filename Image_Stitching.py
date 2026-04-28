
import cv2
import numpy as np


def detect_and_match(img1, img2, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Not enough features were found.")

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        raise RuntimeError("Not enough matches for homography.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        raise RuntimeError("Homography calculation failed.")

    return H, good, kp1, kp2


def warp_two_images(base, img):
    H, _, _, _ = detect_and_match(img, base)

    h1, w1 = base.shape[:2]
    h2, w2 = img.shape[:2]

    corners_base = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)

    corners_img = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    warped_corners_img = cv2.perspectiveTransform(corners_img, H)

    all_corners = np.concatenate(
        (corners_base, warped_corners_img),
        axis=0
    )

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    warped_img = cv2.warpPerspective(
        img,
        translation @ H,
        (canvas_w, canvas_h)
    )

    result = warped_img.copy()

    x_offset = -x_min
    y_offset = -y_min

    base_area = result[
        y_offset:y_offset + h1,
        x_offset:x_offset + w1
    ]

    mask_base = np.any(base > 0, axis=2)
    mask_warped = np.any(base_area > 0, axis=2)

    overlap = mask_base & mask_warped
    only_base = mask_base & ~mask_warped

    base_area[only_base] = base[only_base]

    base_area[overlap] = (
        0.5 * base_area[overlap] +
        0.5 * base[overlap]
    ).astype(np.uint8)

    result[
        y_offset:y_offset + h1,
        x_offset:x_offset + w1
    ] = base_area

    return result


def spherical_warp(img, focal_length):
    h, w = img.shape[:2]

    cx = w / 2
    cy = h / 2

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            theta = (x - cx) / focal_length
            phi = (y - cy) / focal_length

            X = np.sin(theta) * np.cos(phi)
            Y = np.sin(phi)
            Z = np.cos(theta) * np.cos(phi)

            src_x = focal_length * X / Z + cx
            src_y = focal_length * Y / Z + cy

            map_x[y, x] = src_x
            map_y[y, x] = src_y

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return warped


def crop_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y + h, x:x + w]


def stitch_three_images(image_paths, output_path="stitched_result.jpg"):
    if len(image_paths) != 3:
        raise ValueError("Exactly 3 image paths are required.")

    images = []

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        images.append(img)

    focal_length = images[0].shape[1]

    spherical_images = []

    for img in images:
        warped = spherical_warp(img, focal_length)
        spherical_images.append(warped)

    left = spherical_images[0]
    center = spherical_images[1]
    right = spherical_images[2]

    panorama = center

    panorama = warp_two_images(panorama, left)
    panorama = warp_two_images(panorama, right)

    panorama = crop_black_border(panorama)

    cv2.imwrite(output_path, panorama)
    print(f"Saved: {output_path}")

    return panorama


if __name__ == "__main__":
    image_paths = [
        "img1.jpeg",
        "img2.jpeg",
        "img3.jpeg"
    ]

    result = stitch_three_images(
        image_paths,
        output_path="stitched_result.jpg"
    )

    cv2.imshow("Stitched Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()