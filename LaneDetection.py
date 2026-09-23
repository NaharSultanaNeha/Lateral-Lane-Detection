import os
import cv2
import numpy as np

# Camera calibration constants for the Udacity project camera
MTX = np.array([
    [1.15777818e+03, 0.0, 6.67113204e+02],
    [0.0, 1.15215980e+03, 3.86128937e+02],
    [0.0, 0.0, 1.0]
])
DIST = np.array([[-0.24688507, -0.02373154, -0.00109831, 0.00035107, -0.00259869]])

# Perspective transform coordinates (1280x720)
SRC_PTS = np.float32([
    [580, 460],
    [700, 460],
    [1096, 720],
    [200, 720]
])

DST_PTS = np.float32([
    [300, 0],
    [980, 0],
    [980, 720],
    [300, 720]
])

M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
MINV = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)

YM_PER_PIX = 30.0 / 720
XM_PER_PIX = 3.7 / 680


def get_combined_binary(img):
    undist = cv2.undistort(img, MTX, DIST, None, MTX)

    # Color space conversions
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    lab = cv2.cvtColor(undist, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]

    # Left Yellow Line (LAB B-channel cleanly isolates yellow from asphalt and gray barriers)
    yellow_binary = np.zeros_like(b_channel)
    yellow_binary[(b_channel >= 155) & (b_channel <= 200)] = 1

    # Right White Line (RGB high brightness + HLS Lightness)
    r_channel = undist[:, :, 2]
    g_channel = undist[:, :, 1]
    white_binary = np.zeros_like(l_channel)
    white_binary[(l_channel >= 195) & (r_channel >= 200) & (g_channel >= 200)] = 1

    # Sobel X gradient
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / (np.max(abs_sobel) + 1e-6))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= 25) & (scaled_sobel <= 120)] = 1

    # Combine masks
    combined = np.zeros_like(l_channel)
    combined[(yellow_binary == 1) | (white_binary == 1) | (sobel_binary == 1)] = 1

    return undist, combined


class LaneDetector:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.smoothing = 0.85

    def sliding_window(self, binary_warped):
        h, w = binary_warped.shape[:2]
        midpoint = int(w // 2)

        # Histogram across bottom 60% of warped image
        bottom_region = binary_warped[int(h * 0.4):, :]
        histogram = np.sum(bottom_region, axis=0)

        leftx_base = np.argmax(histogram[220:midpoint - 40]) + 220
        rightx_base = np.argmax(histogram[midpoint + 40:w - 60]) + (midpoint + 40)

        nwindows = 9
        window_height = int(h // nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 90
        minpix = 35

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = max(leftx_current - margin, 180)
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = min(rightx_current + margin, w - 40)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))

            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        if len(rightx) < 100:
            return None, None

        # 1. Fit the working right line freely
        right_fit = np.polyfit(righty, rightx, 2)

        # 2. FORCE LEFT TO BEND EXACTLY LIKE THE RIGHT:
        # highway lanes are parallel: curvature A and slope B must match the right line
        if len(leftx) >= 60:
            # Anchor left base position (C) to the actual yellow pixels detected at the bottom
            bottom_left_mask = lefty > int(h * 0.7)
            if np.sum(bottom_left_mask) > 20:
                base_x = np.mean(leftx[bottom_left_mask])
                base_y = np.mean(lefty[bottom_left_mask])
            else:
                base_x = np.mean(leftx)
                base_y = np.mean(lefty)

            # Left equation: x = A_right * y^2 + B_right * y + C_left
            left_fit = np.zeros(3)
            left_fit[0] = right_fit[0]
            left_fit[1] = right_fit[1]
            left_fit[2] = base_x - (right_fit[0] * base_y**2 + right_fit[1] * base_y)
        else:
            # Fallback if yellow line is occluded: clean parallel offset (-660px)
            left_fit = np.copy(right_fit)
            left_fit[2] -= 660

        return left_fit, right_fit

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        undist, binary = get_combined_binary(frame)
        warped = cv2.warpPerspective(binary, M, (w, h), flags=cv2.INTER_NEAREST)

        l_fit, r_fit = self.sliding_window(warped)

        if l_fit is not None and r_fit is not None:
            if self.left_fit is None:
                self.left_fit = l_fit
                self.right_fit = r_fit
            else:
                self.left_fit = self.smoothing * self.left_fit + (1.0 - self.smoothing) * l_fit
                self.right_fit = self.smoothing * self.right_fit + (1.0 - self.smoothing) * r_fit

        if self.left_fit is None or self.right_fit is None:
            return undist

        ploty = np.linspace(0, h - 1, h)
        left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Curvature radius
        y_eval = np.max(ploty)
        right_fit_cr = np.polyfit(ploty * YM_PER_PIX, right_fitx * XM_PER_PIX, 2)
        curverad = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1])**2)**1.5) / max(abs(2 * right_fit_cr[0]), 1e-6)

        # Center offset
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
        offset_m = (w / 2.0 - lane_center) * XM_PER_PIX

        # Unwarp overlay
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), thickness=20)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), thickness=20)

        newwarp = cv2.warpPerspective(color_warp, MINV, (w, h))
        result = cv2.addWeighted(undist, 1.0, newwarp, 0.4, 0)

        # HUD
        cv2.putText(result, f"Radius of Curvature: {int(min(curverad, 9999))} m", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        direction = "right" if offset_m < 0 else "left"
        cv2.putText(result, f"Vehicle Offset: {abs(offset_m):.2f}m {direction} of center", (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return result


if __name__ == "__main__":
    video_files = ["lanetest.mp4", "test1.mp4", "project_video.mp4"]
    input_path = next((f for f in video_files if os.path.exists(f)), None)

    if input_path is None:
        print("Error: Video file not found.")
        exit()

    cap = cv2.VideoCapture(input_path)
    tracker = LaneDetector()

    print(f"Running lane detection on: {input_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = tracker.process_frame(frame)
        cv2.imshow("Lane Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()