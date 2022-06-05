import os
import cv2
import numpy as np
from dual_capture import open_dual_cam, close_dual_cam, dual_capture

def save_images(l_images, r_images):
    os.makedirs('./images/finetune/left', exist_ok=True)
    os.makedirs('./images/finetune/right', exist_ok=True)
    for i, (left, right) in enumerate(zip(l_images, r_images)):
        cv2.imwrite(f'./images/finetune/left/{str(i).zfill(2)}_left.png', left)
        cv2.imwrite(f'./images/finetune/right/{str(i).zfill(2)}_right.png', right)

if __name__ == '__main__':
    l_images = []
    r_images = []
    try:
        l_cam, r_cam = open_dual_cam()
        while True:
            l_image, r_image = dual_capture(l_cam, r_cam)
            l_r_image = np.hstack((l_image, r_image))
            #cv2.imshow('l_r_image', l_r_image)
            #key = cv2.waitKey(10) #& 0xFF
            key = input("press.")
            if key == 'c':
                l_images.append(l_image)
                r_images.append(r_image)
                print('capture images: {}'.format(len(l_images)))
            if key == 's':
                save_images(l_images, r_images)
                print('save images: {}'.format(len(l_images)))
                break
            if key == 'q':
                break
    except Exception as e:
        print(str(e))
    finally:
        close_dual_cam(l_cam, r_cam)
