import cv2
import numpy as np
import matplotlib.pyplot as plt


def cover_plate(results, car_image, overlay, scale_factor):
    # Load the main image
    main_image = cv2.imread(car_image)
    if main_image is None:
        raise ValueError(f"Could not load image {car_image}")
    main_image_rgb = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

    # Load the image to overlay
    overlay_image = cv2.imread(overlay)
    if overlay_image is None:
        raise ValueError(f"Could not load image {overlay}")
    overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    overlay_image_rgb = cv2.rotate(overlay_image_rgb, cv2.ROTATE_90_CLOCKWISE)
    overlay_image_rgb = cv2.flip(overlay_image_rgb, 0)

    if len(results) == 0:
        raise ValueError("No objects detected in the image")

    final_image = main_image_rgb.copy()

    # Iterate over each detected object
    for result in results:
        obb = result.obb
        if obb is None:
            continue

        # Extract the bounding box coordinates from obb.xyxyxyxy
        xyxyxyxy = obb.xyxyxyxy.squeeze().numpy()

        # Ensure the coordinates have the correct shape
        if xyxyxyxy.shape == (4, 2):
            # Apply the overlay to this detected object
            final_image = apply_overlay(
                final_image, overlay_image_rgb, xyxyxyxy, scale_factor
            )
        elif len(xyxyxyxy.shape) == 3 and xyxyxyxy.shape[0] > 1:
            # Handle multiple detections within a single result
            for pts in xyxyxyxy:
                final_image = apply_overlay(
                    final_image, overlay_image_rgb, pts, scale_factor
                )
        else:
            print(f"Skipping a detection due to unexpected shape: {xyxyxyxy.shape}")
            continue

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image)
    plt.axis("off")
    plt.show()

    return final_image


def apply_overlay(main_image_rgb, overlay_image_rgb, pts_dst, scale_factor):
    # Calculate the center of the slanted rectangle
    center = np.mean(pts_dst, axis=0)

    # Calculate the new destination points by moving them closer to the center
    scaled_pts_dst = center + scale_factor * (pts_dst - center)

    # Ensure the destination points have the correct shape
    if scaled_pts_dst.shape != (4, 2):
        print(
            f"Skipping a detection due to unexpected shape for pts_dst: {scaled_pts_dst.shape}"
        )
        return main_image_rgb

    # Source points from the overlay image (rectangular)
    pts_src = np.array(
        [
            [0, 0],
            [overlay_image_rgb.shape[1] - 1, 0],
            [overlay_image_rgb.shape[1] - 1, overlay_image_rgb.shape[0] - 1],
            [0, overlay_image_rgb.shape[0] - 1],
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix
    try:
        M = cv2.getPerspectiveTransform(pts_src, scaled_pts_dst)
    except cv2.error as e:
        print(f"Error in computing perspective transform for a detection: {e}")
        return main_image_rgb

    # Apply the perspective transformation to the overlay image
    warped_overlay = cv2.warpPerspective(
        overlay_image_rgb, M, (main_image_rgb.shape[1], main_image_rgb.shape[0])
    )

    # Create a mask of the warped overlay
    mask = np.zeros(main_image_rgb.shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        mask, scaled_pts_dst.astype(int), (255, 255, 255), lineType=cv2.LINE_AA
    )

    # Mask the area of the overlay in the main image
    masked_main_image = cv2.bitwise_and(main_image_rgb, cv2.bitwise_not(mask))

    # Combine the masked main image with the warped overlay
    final_image = cv2.add(masked_main_image, warped_overlay)

    return final_image


# Example usage:
# cover_plate("car_image.jpg", "overlay_image.jpg", scale_factor=0.9)
