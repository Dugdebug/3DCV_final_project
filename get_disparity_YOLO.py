import os
import cv2
import time
import math
import numpy as np
import torch
        
'''
    Dual camera.
'''
class Dual:
    def __init__(self):
        self.cam_width, self.cam_height = 640, 480
        self.ratio = 1.0
        self.img_width, self.img_height= int(self.cam_width*self.ratio), int(self.cam_height*self.ratio)

    def load_npz(self, npz_file):
        try:
            npzfile = np.load(npz_file)
        except Exception as e:
            raise ExceptionCommon(message=str(e))
        leftMapX = npzfile['leftMapX']
        leftMapY = npzfile['leftMapY']
        rightMapX = npzfile['rightMapX']
        rightMapY = npzfile['rightMapY']
        Q = npzfile['dispartityToDepthMap']
        return leftMapX, leftMapY, rightMapX, rightMapY, Q

    def get_rectified_dual_images(self, left_frame, right_frame, leftMapX, leftMapY, rightMapX, rightMapY):
        iml_rectified = cv2.remap(left_frame, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        imr_rectified = cv2.remap(right_frame, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return iml_rectified, imr_rectified
    
    def get_disparity_image(self, iml_rectified, imr_rectified):
        num = 5                 
        blockSize = 3          
        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity= 0,     
            numDisparities=16 * num, 
            blockSize=blockSize,    
            P1=8 * 3 * blockSize * blockSize, 
            P2=32 * 3 * blockSize * blockSize,
            disp12MaxDiff=1,        
            preFilterCap=15,       
            uniquenessRatio=10,    
            speckleWindowSize=100, 
            speckleRange=2,       
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        stereo_sgbm_right_matcher = cv2.ximgproc.createRightMatcher(stereo_sgbm)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_sgbm)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.2)

        gray_L = cv2.cvtColor(iml_rectified, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(imr_rectified, cv2.COLOR_BGR2GRAY)
        egray_L = cv2.equalizeHist(gray_L)
        egray_R = cv2.equalizeHist(gray_R)

        disparity_left = stereo_sgbm.compute(egray_L, egray_R)
        disparity_right = stereo_sgbm_right_matcher.compute(egray_R,egray_L)
        disparity_left = np.int16(disparity_left)
        disparity_right = np.int16(disparity_right)
        return disparity_left, disparity_right, wls_filter, gray_L, gray_R

    def distance(self, threeD, x, y):
        distance = threeD[y,x,2]
        return distance
    
    def get_threeD_img(self, l_image, r_image, npz_file): 
        leftMapX, leftMapY, rightMapX, rightMapY, Q = self.load_npz(npz_file)
        iml_rectified, imr_rectified = self.get_rectified_dual_images(l_image, r_image, leftMapX, leftMapY, rightMapX, rightMapY)
        
        disparity_left, disparity_right, wls_filter, gray_L, gray_R = self.get_disparity_image(iml_rectified, imr_rectified)
        os.makedirs('./images/disp/left', exist_ok=True)
        os.makedirs('./images/disp/right', exist_ok=True)
        fname = l_fname.split('/')[-1]
        print(fname)
        print(disparity_left.shape)
        cv2.imwrite(f'./images/disp/left/{fname}', disparity_left)
        cv2.imwrite(f'./images/disp/right/{fname}', disparity_right)

        disparity_filtered = wls_filter.filter(disparity_left, gray_L, None, disparity_right)
        imd = disparity_filtered
        color_image = iml_rectified

        threeD = cv2.reprojectImageTo3D(imd, Q, handleMissingValues=True)
        threeD = threeD * 16
        threeD = threeD.astype(np.float16) 

        disp = imd.astype(np.float32) / 16.0
        disp8U = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp8U = cv2.medianBlur(disp8U, 9)
        depth_heatmap = cv2.applyColorMap(disp8U, cv2.COLORMAP_JET)
        os.makedirs('./images/disp/heatmap', exist_ok=True)
        cv2.imwrite(f'./images/disp/heatmap/{fname}', depth_heatmap)
        
        return color_image, threeD, depth_heatmap

def detect_plate_position(l_fname, model):
    results = model(l_fname)
    bbox = results.pandas().xyxy[0]
    if bbox.shape[0] == 0:
        print(f'no plate be found in {l_fname} thus return x, y as (False, False)')
        return False, False
    else:
        bbox_most_conf = bbox.iloc[bbox.confidence.idxmax()]
        x = (bbox_most_conf.xmin + bbox_most_conf.xmax) / 2
        y = (bbox_most_conf.ymin + bbox_most_conf.ymax) / 2
        return int(x), int(y)

def put_text(image, x, y, depth):
    image = cv2.circle(image, (x,y), 3, (0,0,255), -1)
    image = cv2.putText(image, str(int(depth))+'cm', (x,y+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    return image 

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
    model.conf = 0.6
    dataset_path_left = './images/motor/left'
    dataset_path_right = './images/motor/right'
    npz_file = './calibrate_settings/fishstereocalibrate.npz'
    left_images = sorted([os.path.join(dataset_path_left, img) for img in os.listdir(dataset_path_left)])[:2]
    right_images = sorted([os.path.join(dataset_path_right, img) for img in os.listdir(dataset_path_right)])[:2]
    dual = Dual()
    last_depth = 0
    for i, (l_fname, r_fname) in enumerate(zip(left_images, right_images)):
        print(i)
        l_image, r_image = cv2.imread(l_fname), cv2.imread(r_fname)
        start_time = time.time()
        _, threeD, _ = dual.get_threeD_img(l_image, r_image, npz_file)
        print('time = %.2f' %(time.time() - start_time))
        x,y = detect_plate_position(l_fname, model)
        if not x:
            continue
        depth = threeD[:,:,2]
        point_depth = depth[y, x]/10
        if np.isinf(np.sum(point_depth)) or np.isnan(np.sum(point_depth)): 
            point_depth = last_depth
        last_depth = point_depth 
        draw_image = put_text(l_image, x, y, point_depth) 
        os.makedirs('./images/motor/result', exist_ok=True)
        fname = l_fname.split('/')[-1]
        cv2.imwrite(f'./images/motor/result/{fname}', draw_image)
