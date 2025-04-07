import cv2

def resize_frame_to_screen(frame, screen_width, screen_height):
    """Resize the frame to fit the screen."""
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    if frame_width > screen_width or frame_height > screen_height:
        if screen_width / aspect_ratio <= screen_height:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))
    return frame