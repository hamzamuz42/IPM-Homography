import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import perspective, Plane, load_camera_params, bilinear_sampler, warped

image = cv2.cvtColor(cv2.imread('eastbound.jpg'), cv2.COLOR_BGR2RGB)
interpolation_fn = bilinear_sampler  # or warped
TARGET_H, TARGET_W = 500, 500


def ipm_from_parameters(image, xyz, K, RT, interpolation_fn):
    # Flip y points positive upwards
    xyz[1] = -xyz[1]

    P = K @ RT
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    image2 = interpolation_fn(image, pixel_coords)
    return image2.astype(np.uint8)


def ipm_from_opencv(image, source_points, target_points):
    # Compute projection matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped


if __name__ == '__main__':
    ################
    # Derived method
    ################
    # Define the plane on the region of interest (road)
    plane = Plane(0, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.1)
    # Retrieve camera parameters
    extrinsic, intrinsic = load_camera_params('camera.json')
    # Apply perspective transformation
    warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic, interpolation_fn)

    ################
    # OpenCV
    ################
    # Vertices coordinates in the source image
    s = np.array([[724, 276],
                  [706, 170],
                  [1061, 272],
                  [944, 162]], dtype=np.float32)

    # Vertices coordinates in the destination image
    t = np.array([[177, 231],
                  [213, 231],
                  [178, 264],
                  [216, 264]], dtype=np.float32)

    # Warp the image
    warped2 = ipm_from_opencv(image, s, t)

    # Draw results
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image)
    ax[0].set_title('Front View')
    cv2.imwrite("Test.jpg", warped1)
    ax[1].imshow(warped1)
    ax[1].set_title('IPM')
    # ax[2].imshow(warped2)
    # ax[2].set_title('From OpenCv')
    plt.tight_layout()
    plt.show()
