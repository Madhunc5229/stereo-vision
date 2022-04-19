import cv2
import numpy as np
import random
import statistics as st
from matplotlib import pyplot as plt


#funstion to find matches 
def findMatches(image1, image2):

    print("Finding features and matchpoints...\n")
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10000)

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)

    final_matches = matches[:30]

    img_with_keypoints = cv2.drawMatches(img1,kp1,img2,kp2,final_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("images_with_matching_keypoints.png", img_with_keypoints)


    list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in final_matches] 
    list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in final_matches]

    return list_kp1, list_kp2


def findFundMat(pts_1, pts_2):

    set_1 = pts_1
    set_2 = pts_2

    A = np.empty((8,9))

    for i in range(len(pts_1)):
        x1, y1 = pts_1[i][0], pts_1[i][1]
        x2, y2 = pts_2[i][0], pts_2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    
    u,s,v = np.linalg.svd(A)
    F = np.reshape(v[-1,:],(3,3))

    u_f, s_f, v_f = np.linalg.svd(F)
    
    s_f[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i][i] = s_f[i]

    F = u_f.dot(s.dot(v_f))

    #Normalization
    set_1 = np.array(set_1)
    set_2 = np.array(set_2)

    s1_xm = st.mean(set_1[:,0])
    s1_ym = st.mean(set_1[:,1])

    s1_xsd = st.stdev(set_1[:,0])
    s1_ysd = st.stdev(set_1[:,1])

    offset_1 = np.array([[1,0, -s1_xm],[ 0,1,-s1_ym],[0, 0, 1]])
    scale_1 = np.array([[1/s1_xsd,0,0],[0,1/s1_ysd,0],[0,0,1]])

    set_2 = np.array(set_2)
    s2_xm = st.mean(set_2[:,0])
    s2_ym = st.mean(set_2[:,1])

    s2_xsd = st.stdev(set_2[:,0])
    s2_ysd = st.stdev(set_2[:,1])

    offset_2 = np.array([[1,0, -s2_xm],[ 0,1, -s2_ym],[ 0, 0, 1]])
    scale_2 = np.array([[1/s2_xsd,0,0],[0,1/s2_ysd,0],[0,0,1]])

    T_1 = scale_1.dot(offset_1)
    T_2 = scale_2.dot(offset_2)

    F_norm = np.transpose(T_2).dot(F.dot(T_1))

    return F_norm

#Estiamte the best F matrix using RANSAC
def bestFusingRansac(pts_1_list, pts_2_list):

    points = list(zip(pts_1_list,pts_2_list))


    max_inliers = 20

    threshold = 0.05

    N=0
    print("Estimating F using RANSAC...")

    while N<1000:
        #Pick random 8 points
        points = random.sample(points,8)

        img1_8pt, img2_8pt = zip(*points) 

        F_m = findFundMat(img1_8pt,img2_8pt)

        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(pts_1_list)):
            img1_x = np.array([pts_1_list[i][0], pts_1_list[i][1], 1])
            img2_x = np.array([pts_2_list[i][0], pts_2_list[i][1], 1])
            
            #Compute error
            error = abs(np.dot(img2_x.T, np.dot(F_m,img1_x)))
            

            if error < threshold:
                tmp_inliers_img1.append(pts_1_list[i])
                tmp_inliers_img2.append(pts_1_list[i])

        num_of_inliers = len(tmp_inliers_img1)
        
        if num_of_inliers > max_inliers:

            max_inliers = num_of_inliers
            Best_F = F_m
            # inliers_img1 = tmp_inliers_img1
            # inliers_img2 = tmp_inliers_img2
        N +=1

    return Best_F

#find E 
def findEssenMat(fundMatrix, KMatrix):
    return np.transpose(KMatrix).dot(fundMatrix.dot(KMatrix))


def cameraPoseEst(essenMat):
    U, D, Vt = np.linalg.svd(essenMat)

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = U.dot(W.dot(Vt))
    R2 = U.dot(np.transpose(W).dot(Vt))
    C1 = U[:,2]
    C2 = -U[:,2]

    return [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]

#plot epipolar lines
def plotepilines(img_1, img_2, lines, rpts_1, rpts_2):

    r, c,ch = img_1.shape


    for r, pt1, pt2 in zip(lines, rpts_1, rpts_2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img_1 = cv2.line(img_1, (x0, y0), (x1, y1), color, 1)
        img_1 = cv2.circle(img_1, tuple(pt1), 5, color, -1)
        img_2 = cv2.circle(img_2, tuple(pt2), 5, color, -1)

    return img_1, img_2

# get best R and T based on Chirelity check
def bestCameraPose(camera_poses, kp1):

    max_len = 0
    # Calculating 3D points 
    for pose in camera_poses:

        front_points = []        
        for point in kp1:
            # Chirelity check
            X = np.array([point[0], point[1], 1])
            V = X - pose[1]
            
            condition = np.dot(pose[0][2], V)

            if condition > 0:
                front_points.append(point)    

        if len(front_points) > max_len:
            max_len = len(front_points)
            best_camera_pose =  pose
    
    return best_camera_pose

#SSD
def sum_of_squared_diff(pixel_vals_1, pixel_vals_2):

    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum((pixel_vals_1 - pixel_vals_2)**2)


def block_compare(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    

    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)
    
    first = True
    min_ssd = None
    min_index = None

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            block_right = right_array[y: y+block_size, x: x+block_size]
            ssd = sum_of_squared_diff(block_left, block_right)
            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)

    return min_index


def ssd_correspondence(rimg_1, rimg_2):

    img1 = cv2.resize(rimg_1, (int(rimg_1.shape[1]*0.3),int(rimg_1.shape[0]*0.3)),interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(rimg_2, (int(rimg_2.shape[1]*0.3),int(rimg_2.shape[0]*0.3)),interpolation = cv2.INTER_AREA)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    block_size = 15 
    x_search_block_size = 50
    y_search_block_size = 1
    h, w = img1.shape
    disparity_map = np.zeros((h, w))

    for y in range(block_size, h-block_size):
        for x in range(block_size, w-block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = block_compare(y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)
    
    disparity_map_unscaled = disparity_map.copy()


    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_map[i][j] = int((disparity_map[i][j]*255)/(max_pixel-min_pixel))
    
    disparity_map_scaled = disparity_map
    
    return disparity_map_unscaled, disparity_map_scaled

def disparity_to_depth(baseline, f, img):

    depth_map = np.zeros((img.shape[0], img.shape[1]))
    depth_array = np.zeros((img.shape[0], img.shape[1]))

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if img[i][j] !=0:
                depth_map[i][j] = 1/img[i][j]
                depth_array[i][j] = baseline*f/img[i][j]

    return depth_map, depth_array

if __name__=='__main__':

    img1 = cv2.imread('../media/curule/im0.png')
    img2 = cv2.imread('../media/curule/im1.png')

    print("CALIBRATION\n")
    kp1,kp2 = findMatches(img1,img2)

    # print(kp1[0][0])
    while True:
        try:
            fundMat = bestFusingRansac(kp1,kp2)

            pts1 = np.int32(kp1)
            pts2 = np.int32(kp2)

            h1, w1,ch = img1.shape
            h2, w2,ch = img2.shape
            _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, fundMat, imgSize=(w1, h1))

            break
        except:
            continue

    print("\nThe Fundamental Matrix F = \n", fundMat,"\n")

    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))

    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    
    K = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0,0,1]])

    E = findEssenMat(fundMat,K)

    print("Essential Matrix E = \n", E,"\n")

    camera_poses = cameraPoseEst(E)

    best_camera_pose = bestCameraPose(camera_poses,kp1)

    print("The Camera Pose, R = ", best_camera_pose[0], "\n", "T = ", best_camera_pose[1])
    print("RECTIFICATION")
    print("The Homography matrices H1 = \n", H1,"\n", "\nH2 = ", H2,"\n")
    r_pts_1 = np.zeros((pts1.shape), dtype=int)

    r_pts_2 = np.zeros((pts2.shape), dtype=int)

    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        r_pts_1[i] = new_point1

        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        r_pts_2[i] = new_point2

    lines1 = cv2.computeCorrespondEpilines(r_pts_2.reshape(-1, 1, 2), 2, fundMat)
    lines1 = lines1.reshape(-1, 3)

    img1EL,img21EL = plotepilines(img1_rectified,img2_rectified,lines1,r_pts_1,r_pts_2)

    lines2 = cv2.computeCorrespondEpilines(r_pts_1.reshape(-1, 1, 2), 2, fundMat)
    lines2 = lines2.reshape(-1, 3)

    img2EL, img12EL = plotepilines(img2_rectified, img1_rectified, lines2, r_pts_2, r_pts_1)

    cv2.imwrite("left_image.png", img1EL)
    cv2.imwrite("right_image.png", img2EL)

    print("CORRESPONDENCE\n")

    disparity_map_unscaled, disparity_map_scaled = ssd_correspondence(img1_rectified, img2_rectified)

    plt.figure(1)
    plt.title('Disparity-Map : Graysacle')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.figure(2)
    plt.title('Disparity-Map : Hot')
    plt.imshow(disparity_map_scaled, cmap='hot')

    baseline, f = 88.39, 1758.23

    print("DEPTH ESTIMATION\n")

    depth_map, depth_array = disparity_to_depth(baseline, f, disparity_map_unscaled)

    plt.figure(3)
    plt.title('Depth-Map : Graysacle')
    plt.imshow(depth_map, cmap='gray')
    plt.figure(4)
    plt.title('Depth-Map : Hot')
    plt.imshow(depth_map, cmap='hot')
    plt.show()
