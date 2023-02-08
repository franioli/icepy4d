import numpy as np
import cv2 as cv

ix, iy, sx, sy = -1, -1, -1, -1

# mouse callback function
def draw_lines(event, x, y, flags, param):
    global ix, iy, sx, sy
    # if the left mouse button was clicked, record the starting
    if event == cv.EVENT_LBUTTONDOWN:

        # draw circle of 2px
        cv.circle(img, (x, y), 3, (0, 0, 127), -1)

        if ix != -1:  # if ix and iy are not first points, then draw a line
            cv.line(img, (ix, iy), (x, y), (0, 0, 127), 2, cv.LINE_AA)
        else:  # if ix and iy are first points, store as starting points
            sx, sy = x, y
        ix, iy = x, y

    elif event == cv.EVENT_LBUTTONDBLCLK:
        ix, iy = -1, -1  # reset ix and iy
        if (
            flags == 33
        ):  # if alt key is pressed, create line between start and end points to create polygon
            cv.line(img, (x, y), (sx, sy), (0, 0, 127), 2, cv.LINE_AA)


# read image from path and add callback
# img = cv.resize(cv.imread("themefoxx (214).jpg"), (1280, 720))
img = cv.cvtColor(images[cam].read_image(ep).value, cv.COLOR_RGB2BGR)

cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
cv.setMouseCallback("image", draw_lines)

while True:
    cv.imshow("image", img)
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()
