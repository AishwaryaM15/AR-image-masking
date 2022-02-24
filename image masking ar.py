import cv2
import numpy as np

cap=cv2.VideoCapture(0)
imgTarget1=cv2.imread(r'C:\Users\hp\Desktop\targetimg.jpeg')
myVid=cv2.VideoCapture(r'C:\Users\hp\Desktop\dispvid.mp4')

detection=False
frameCounter=0




scale_factor=0.2
width=int(imgTarget1.shape[1]*scale_factor)
height=int(imgTarget1.shape[0]*scale_factor)
dim=(width,height)
imgTarget=cv2.resize(imgTarget1,dim, interpolation=cv2.INTER_AREA)

success, imgVideo=myVid.read()
h1,w1,c1=imgTarget.shape
imgVideo=cv2.resize(imgVideo,(w1,h1))

orb=cv2.ORB_create(nfeatures=1000)
kp1,des1=orb.detectAndCompute(imgTarget,None)
#imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)



while True:
    sucsess, imgWebcam=cap.read()
    imgAug=imgWebcam.copy()
    kp2,des2=orb.detectAndCompute(imgWebcam,None)
    #imgWebcam=cv2.drawKeypoints(imgWebcam,kp2,None)
    
    if detection==False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
    else:
        if frameCounter==myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0
        success,imgVideo=myVid.read()
        imgVideo=cv2.resize(imgVideo,(w1,h1))
    
    bf=cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)
    good=[]
    for m,n in matches:
        if m.distance<2*n.distance:
            good.append(m)
    print(len(good))
    imgFeatures=cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)
    
    if len(good)>100:
        detection=True
        srcPts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        matrix,mask=cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)
        
        pts=np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        dst=cv2.perspectiveTransform(pts,matrix)
        img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,255),3)
        
        imgWarp=cv2.warpPerspective(imgVideo,matrix,(img2.shape[1],img2.shape[0]))
        
        maskNew=np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv=cv2.bitwise_not(maskNew)
        imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
        imgAug=cv2.bitwise_or(imgWarp,imgAug)
    
    cv2.imshow('maskNew', imgAug)
    cv2.imshow('imgWarp', imgWarp)
    #cv2.imshow('img2', img2)
    #cv2.imshow('imgFeatures', imgFeatures)
    #cv2.imshow('imgTarget', imgTarget)
    #cv2.imshow('myVid', imgVideo)
    #cv2.imshow('Webacam', imgWebcam)
    cv2.waitKey(1)
    frameCounter+=1