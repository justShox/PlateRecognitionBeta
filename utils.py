import os
import cv2
import pyautogui


def get_screen_size():
    """Get the screen size using pyautogui."""
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


def resize_frame_to_screen(frame, screen_width, screen_height):
    """Resize the frame to match the screen size."""
    return cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_AREA)


def get_rectangle_coordinates_from_stream(camera_index=0):
    global drawing, ix, iy, ex, ey, img_copy

    def draw_rectangle(event, x, y, flags, param):
        global drawing, ix, iy, ex, ey, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            img_copy = img.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = img.copy()
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Frame', img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            ex, ey = x, y
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (ex, ey), (0, 255, 0), 2)
            cv2.imshow('Frame', img_copy)
            print(f'Rectangle coordinates: Top-left: ({ix}, {iy}), Bottom-right: ({ex}, {ey})')

    drawing = False
    ix, iy = -1, -1
    ex, ey = -1, -1
    img_copy = None

    monitor_width, monitor_height = get_screen_size()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Cannot open camera")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None

    frame_height, frame_width = frame.shape[:2]
    if frame_width != monitor_width or frame_height != monitor_height:
        print(
            f"Frame size ({frame_width}, {frame_height}) does not match monitor size ({monitor_width}, {monitor_height}). Cropping...")
        frame = resize_frame_to_screen(frame, monitor_width, monitor_height)

    img = frame.copy()
    cv2.imshow('Frame', img)
    cv2.setMouseCallback('Frame', draw_rectangle)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            ix, iy, ex, ey = -1, -1, -1, -1
            break
        elif not drawing and ix != -1 and iy != -1 and ex != -1 and ey != -1:
            break

    cap.release()
    cv2.destroyAllWindows()

    return (ix, iy, ex, ey)


def is_rectangle_inside_another(inner_rect, outer_rect):
    """
    Check if the inner rectangle is completely inside the outer rectangle.

    Parameters:
    - inner_rect: A tuple of the inner rectangle's coordinates (x1, y1, x2, y2).
    - outer_rect: A tuple of the outer rectangle's coordinates (x1, y1, x2, y2).

    Returns:
    - True if the inner rectangle is inside the outer rectangle, otherwise False.

    """

    x1_inner, y1_inner, x2_inner, y2_inner = inner_rect
    x1_outer, y1_outer, x2_outer, y2_outer = outer_rect

    return (
            x1_outer <= x1_inner <= x2_outer and
            y1_outer <= y1_inner <= y2_outer and
            x1_outer <= x2_inner <= x2_outer and
            y1_outer <= y2_inner <= y2_outer
    )


def initialize_directories():
    """Create directories to save images."""
    output_dirs = {
        'result': './result',
    }
    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)
    return output_dirs

# outer_rectangle = (10, 20, 30, 40)  # Outer rectangle: top-left (10, 20), bottom-right (30, 40)
# inner_rectangle = (15, 25, 25, 35)  # Inner rectangle: top-left (15, 25), bottom-right (25, 35)

# if is_rectangle_inside_another(inner_rectangle, outer_rectangle):
#     print(f"The inner rectangle {inner_rectangle} is inside the outer rectangle {outer_rectangle}.")
# else:
#     print(f"The inner rectangle {inner_rectangle} is not inside the outer rectangle {outer_rectangle}.")
