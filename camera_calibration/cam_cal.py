import os
import cv2
import numpy as np

# Global variables preset
total_photos = 15

# Camera resolution
photo_width = 640
photo_height = 480

# Image resolution for processing
img_width = 640
img_height = 480
image_size = (img_width,img_height)

# Chessboard parameters, the unit of squre size is mm.
#CHECKERBOARD = (7,12)
Nx_cor = 6 
Ny_cor = 9 
square_size = 24 

# Visualization options
drawCorners = False
showSingleCamUndistortionResults = True
showStereoRectificationResults = True
writeUdistortedImages = True

class Calibration:
    def __init__(self, img_width, img_height, Nx_cor, Ny_cor):
        self.img_width = img_width
        self.img_height = img_height
        self.Nx_cor = Nx_cor
        self.Ny_cor = Ny_cor
        print(self.img_width, self.img_height, self.Nx_cor, self.Ny_cor)
        
    def checkboard_points(self, total_photos, square_size):
        img_width = self.img_width
        img_height = self.img_height
        Nx_cor = self.Nx_cor
        Ny_cor = self.Ny_cor
        # Visualization options
        drawCorners = False
    
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30 , 1e-4)
    
        photo_width = img_width
        photo_height = img_height
        CHECKERBOARD = (Ny_cor, Nx_cor)
        objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
        objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= square_size
    
        _img_shape = None
        objpointsLeft = [] # 3d point in real world space
        imgpointsLeft = [] # 2d points in image plane.

        objpointsRight = [] # 3d point in real world space
        imgpointsRight = [] # 2d points in image plane.
    
        photo_counter = 0
        print ('Main cycle start.')
        print('Reading pairs image...')

        while photo_counter != total_photos:
            print ('Import pair No ' + str(photo_counter))
            leftName = './images/pairs/left/'+str(photo_counter).zfill(2)+'_left.png'
            rightName = './images/pairs/right/'+str(photo_counter).zfill(2)+'_right.png'
            leftExists = os.path.isfile(leftName)
            rightExists = os.path.isfile(rightName)
            print('exist: ', rightExists, leftExists)
            photo_counter = photo_counter + 1
        
            # If pair has no left or right image - exit
            if ((leftExists == False) or (rightExists == False)) and (leftExists != rightExists):
                print ("Pair No ", photo_counter, "has only one image! Left:", leftExists, " Right:", rightExists )
                continue

            # If stereopair is complete - go to processing
            if (leftExists and rightExists):
                imgL = cv2.imread(leftName,1)
                loadedY, loadedX, clrs  =  imgL.shape
                grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
                gray_small_left = cv2.resize(grayL, (img_width,img_height), interpolation = cv2.INTER_AREA)
                imgR = cv2.imread(rightName,1)
                grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
                gray_small_right = cv2.resize(grayR, (img_width,img_height), interpolation = cv2.INTER_AREA)

                # Find the chessboard corners
                retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

                # Draw images with corners found
                if (drawCorners):
                    cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
                    cv2.imshow('Corners LEFT', imgL)
                    cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
                    cv2.imshow('Corners RIGHT', imgR)
                    key = cv2.waitKey(0)
                    if key == ord("q"):
                        exit(0)
                    
                # Here is our scaling trick! Hi res for calibration, low res for real work!
                # Scale corners X and Y to our working resolution
                if ((retL == True) and (retR == True)) and (img_height <= photo_height):
                    scale_ratio = img_height/photo_height
                    cornersL = cornersL*scale_ratio 
                    cornersR = cornersR*scale_ratio 
                elif (img_height > photo_height):
                    print ("Image resolution is higher than photo resolution, upscale needed. Please check your photo and image parameters!")
                    exit (0)

                # Refine corners and add to array for processing
                if ((retL == True) and (retR == True)):
                    objpointsLeft.append(objp)
                    cv2.cornerSubPix(gray_small_left,cornersL,(3,3),(-1,-1),subpix_criteria)
                    imgpointsLeft.append(cornersL)
                    objpointsRight.append(objp)
                    cv2.cornerSubPix(gray_small_right,cornersR,(3,3),(-1,-1),subpix_criteria)
                    imgpointsRight.append(cornersR)
                else:
                    print ("Pair No", photo_counter, "ignored, as no chessboard found" )
                    continue
        print ('End cycle')
        return objpointsLeft, imgpointsLeft, objpointsRight, imgpointsRight, gray_small_left, gray_small_right

    # This function calibrates (undistort) a single camera
    def calibrate_one_camera (self, objpoints, imgpoints, right_or_left):
        img_width = self.img_width
        img_height = self.img_height
        #def calibrate_one_camera (objpoints, imgpoints, right_or_left):
        #print(img_width, img_height)
        # calibration_flags
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW  # 14
        #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    
        # Opencv sample code uses the var 'grey' from the last opened picture
        N_OK = len(objpoints)
        DIM= (img_width, img_height)

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        # Single camera calibration (undistortion)
        rms, camera_matrix, distortion_coeff, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                #grayL.shape[::-1],
                (img_width,img_height),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        #keep_this_function_in_mind
        #new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, None, np.eye(3), balance=0.0, fov_scale=1.0 )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        # Let's rectify our results
        #map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

        # Now we'll write our results to the file for the future use
        if (os.path.isdir('./calibration_data/{}p'.format(img_height))==False):
            os.makedirs('./calibration_data/{}p'.format(img_height))
        np.savez('./calibration_data/{}p/camera_calibration_{}.npz'.format(img_height, right_or_left),
            map1=map1, map2=map2, objpoints=objpoints, imgpoints=imgpoints,
            camera_matrix=camera_matrix, distortion_coeff=distortion_coeff)
        return (True)

    # Stereoscopic calibration
    def calibrate_stereo_cameras(self):
        res_x = self.img_width
        res_y = self.img_height
        #print(res_x, res_y)
        # We need a lot of variables to calibrate the stereo camera
        """
        Based on code from:
        https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013
        """
        processing_time01 = cv2.getTickCount()
        objectPoints = None

        rightImagePoints = None
        rightCameraMatrix = None
        rightDistortionCoefficients = None

        leftImagePoints = None
        leftCameraMatrix = None
        leftDistortionCoefficients = None

        rotationMatrix = None
        translationVector = None

        imageSize= (res_x, res_y)

        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        OPTIMIZE_ALPHA = 0.25

        try:
            npz_file = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(res_y))
        except:
            pass
        print('Loading left/right camera calibration data...')
        for cam_num in [0, 1]:
            right_or_left = ["_right" if cam_num==1 else "_left"][0]

            try:
                print ('./calibration_data/{}p/camera_calibration{}.npz'.format(res_y, right_or_left))
                npz_file = np.load('./calibration_data/{}p/camera_calibration{}.npz'.format(res_y, right_or_left))

                list_of_vars = ['map1', 'map2', 'objpoints', 'imgpoints', 'camera_matrix', 'distortion_coeff']

                if sorted(list_of_vars) == sorted(npz_file.files):
                    print("Camera calibration data has been found in cache.")
                    map1 = npz_file['map1']
                    map2 = npz_file['map2']
                    objectPoints = npz_file['objpoints']
                    if right_or_left == "_right":
                        rightImagePoints = npz_file['imgpoints']
                        rightCameraMatrix = npz_file['camera_matrix']
                        rightDistortionCoefficients = npz_file['distortion_coeff']
                    if right_or_left == "_left":
                        leftImagePoints = npz_file['imgpoints']
                        leftCameraMatrix = npz_file['camera_matrix']
                        leftDistortionCoefficients = npz_file['distortion_coeff']
                else:
                    print("Camera data file found but data corrupted.")
            except:
                #If the file doesn't exist
                print("Camera calibration data not found in cache.")
                return False


        print("Calibrating cameras together...")

        leftImagePoints = np.asarray(leftImagePoints, dtype=np.float64)
        rightImagePoints = np.asarray(rightImagePoints, dtype=np.float64)

        # Stereo calibration
        (RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
                objectPoints, leftImagePoints, rightImagePoints,
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                imageSize, None, None,
                cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
        # Print RMS result (for calibration quality estimation)
        print ("<><><><><><><><><><><><><><><><><><><><>")
        print ("<><>   RMS is ", RMS, " <><>")
        print ("<><><><><><><><><><><><><><><><><><><><>")
        print("Rectifying cameras...")
        R1 = np.zeros([3,3])
        R2 = np.zeros([3,3])
        P1 = np.zeros([3,4])
        P2 = np.zeros([3,4])
        Q = np.zeros([4,4])

        # Rectify calibration results
        '''
        (leftRectification, rightRectification, leftProjection, rightProjection,
                dispartityToDepthMap) = cv2.fisheye.stereoRectify(
                        leftCameraMatrix, leftDistortionCoefficients,
                        rightCameraMatrix, rightDistortionCoefficients,
                        imageSize, rotationMatrix, translationVector,
                        cv2.CALIB_ZERO_DISPARITY, (0,0) , balance=0.0, fov_scale=1.0)
        '''
        (leftRectification, rightRectification, leftProjection, rightProjection,
                dispartityToDepthMap) = cv2.fisheye.stereoRectify(
                        leftCameraMatrix, leftDistortionCoefficients,
                        rightCameraMatrix, rightDistortionCoefficients,
                        imageSize, rotationMatrix, translationVector,
                        cv2.CALIB_ZERO_DISPARITY, (0,0), 0, 0)
    
        # Saving calibration results for the future use
        print("Saving calibration...")
        leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
                leftCameraMatrix, leftDistortionCoefficients, leftRectification,
                leftProjection, imageSize, cv2.CV_16SC2)
        rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
                rightCameraMatrix, rightDistortionCoefficients, rightRectification,
                rightProjection, imageSize, cv2.CV_16SC2)

        np.savez_compressed('./calibration_data/{}p/stereo_camera_calibration.npz'.format(res_y), imageSize=imageSize,
                leftMapX=leftMapX, leftMapY=leftMapY,
                rightMapX=rightMapX, rightMapY=rightMapY, dispartityToDepthMap = dispartityToDepthMap)
        if (os.path.isdir('./calibrate_settings/')==False):
            os.makedirs('./calibrate_settings/')
        np.savez_compressed('./calibrate_settings/fishstereocalibrate.npz', imageSize=imageSize,
                leftMapX=leftMapX, leftMapY=leftMapY,
                rightMapX=rightMapX, rightMapY=rightMapY, dispartityToDepthMap = dispartityToDepthMap)
        return True

    def SingleCamUndistortionResults(self, gray_small_left, gray_small_right):
        img_width = self.img_width
        img_height = self.img_height
        try:
            npz_file = np.load('./calibration_data/{}p/camera_calibration{}.npz'.format(img_height, '_left'))
            if 'map1' and 'map2' in npz_file.files:
                map1 = npz_file['map1']
                map2 = npz_file['map2']
            else:
                print("Camera data file found but data corrupted.")
                exit(0)
        except:
            print("Camera calibration data not found in cache, file " & './calibration_data/{}p/camera_calibration{}.npz'.format(h, left))
            exit(0)

        # We didn't load a new image from file, but use last image loaded while calibration
        undistorted_left = cv2.remap(gray_small_left, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        try:
            npz_file = np.load('./calibration_data/{}p/camera_calibration{}.npz'.format(img_height, '_right'))
            if 'map1' and 'map2' in npz_file.files:
                #print("Camera calibration data has been found in cache.")
                map1 = npz_file['map1']
                map2 = npz_file['map2']
            else:
                print("Camera data file found but data corrupted.")
                exit(0)
        except:
            print("Camera calibration RIGHT data not found in cache.")
            exit(0)

        undistorted_right = cv2.remap(gray_small_right, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        result = np.concatenate((undistorted_left,undistorted_right),axis=1)
        result[::20,:] = 0
        cv2.imwrite("./undistorted.png", result)
        cv2.imshow('Left/Right UNDISTORTED', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def StereoRectificationResults(self, imgLTest, imgRTest):
        img_width = self.img_width
        img_height = self.img_height
        try:
            npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
        except:
            print("Camera calibration data not found in cache, file " & './calibration_data/{}p/stereo_camera_calibration.npz'.format(720))
            exit(0)

        leftMapX = npzfile['leftMapX']
        leftMapY = npzfile['leftMapY']
        rightMapX = npzfile['rightMapX']
        rightMapY = npzfile['rightMapY']
   

        # If pair has been loaded and splitted correclty?
        height_left, width_left = imgLTest.shape[:2]
        height_right, width_right = imgRTest.shape[:2]
        #print(width_left, height_left, width_right, height_right)
        if 0 in [width_left, height_left, width_right, height_right]:
            print("Error: Can't remap image.")

        # Rectifying left and right images
        imgL = cv2.remap(imgLTest, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        imgR = cv2.remap(imgRTest, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        result = np.concatenate((imgL, imgR), axis=1)
        result[::20, :] = 0
        cv2.imwrite("./rec.png", result)
        cv2.imshow("STEREO CALIBRATED",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    calib = Calibration(img_width, img_height, Nx_cor, Ny_cor)
    
    objpointsLeft, imgpointsLeft, objpointsRight, imgpointsRight, gray_small_left, gray_small_right\
                   = calib.checkboard_points(total_photos,square_size)
    # Now we have all we need to do stereoscopic fisheye calibration
    # Let's calibrate each camera, and than calibrate them together
    print ("Left camera calibration...")
    result = calib.calibrate_one_camera(objpointsLeft, imgpointsLeft, 'left')
    print ("Right camera calibration...")
    result = calib.calibrate_one_camera(objpointsRight, imgpointsRight,'right')
    print("Done.")
    print ("Stereoscopic calibration...")
    result = calib.calibrate_stereo_cameras()
    print("Done.")
    print ("Calibration complete!")
    if showSingleCamUndistortionResults == True:
        calib.SingleCamUndistortionResults(gray_small_left, gray_small_right)
    if showStereoRectificationResults == True:
        calib.StereoRectificationResults(gray_small_left, gray_small_right)
