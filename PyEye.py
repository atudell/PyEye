import numpy as np
import cv2

def projectOnEyes(image_src, window_name = "projectOnEyes"):
    
    # Import the pre-trained models for face and eye detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Declare a cache for eye locations for when the program can't find the eyes
    eye_cache = None
    
    cap = cv2.VideoCapture(0)

    while True:

        # Read the frame
        ret, frame = cap.read()

        # Check to make sure camera loaded the image correctly
        if not ret:
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:

            # Draw a rectangle around the face
            roi_gray = gray_frame[y:y+h, x:x+w]
            #roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within the grayscale region of interest (ie, the face)
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Only detect 2 eyes
            if len(eyes) == 2:
                # Store the position of the eyes in cache
                eye_cache = eyes
            # If 2 eyes aren't detected, use the eye cache
            elif eye_cache is not None:
                eyes = eye_cache

            # define the destination matrix based on eye detected order. Order of points must be top-left, top-right, bottom-left,
            # and bottom-right
            if eyes[0][0] < eyes[1][0]:
                dst_mat = np.array([
                    [x + eyes[0][0], y + eyes[0][1]], # Top right
                    [x + eyes[1][0] + eyes[1][2], y + eyes[1][2]], # Top left
                    [x + eyes[1][0] + eyes[1][2], y + eyes[1][1] + eyes[1][3]], # bottom-left
                    [x + eyes[0][0], y + eyes[0][1] + eyes[0][3]] # bottom-right

                ])
            else:
                dst_mat = np.array([
                    [x + eyes[1][0], y + eyes[1][1]], # Top right
                    [x + eyes[0][0] + eyes[0][2], y + eyes[0][2]], # Top left
                    [x + eyes[0][0] + eyes[0][2], y + eyes[0][1] + eyes[1][3]], # bottom-left 
                    [x + eyes[1][0], y + eyes[1][1] + eyes[1][3]] # bottom-right

                ])
        # Get the dimensions of the face ROI
        face_h = frame.shape[0]
        face_w = frame.shape[1]

        # read the image and get its dimensions
        img = cv2.imread(image_src, -1)
        img_h = img.shape[0]
        img_w = img.shape[1]

        # Create source matrix
        src_mat = np.array([[0,0], [img_w, 0],  [img_w, img_h], [0, img_h]])

        # Find the Homorgraphy matrix
        hom = cv2.findHomography(src_mat, dst_mat)[0]

        # Warp the image to fit the homegraphy matrix
        warped = cv2.warpPerspective(img, hom, (face_w, face_h))

        # Grab the alpha channel of the warped image and create a mask
        mask = warped[:,:,3]
        mask_scale = mask.copy() / 255.0
        mask_scale = np.dstack([mask_scale] * 3)

        # Remove the alpha channel from the warped image
        warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)

        warped_multiplied = cv2.multiply(mask_scale, warped.astype("float"))
        image_multiplied = cv2.multiply(frame.astype(float), 1.0 - mask_scale)
        output = cv2.add(warped_multiplied, image_multiplied)
        output = output.astype("uint8")

        cv2.imshow(window_name, output)

        if cv2.waitKey(60) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
