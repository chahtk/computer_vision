# organize imports
import cv2
import imutils
import numpy as np
import socket


####### global ########
bl=[]
tr=[]
region=[]
BS=[]
chk = False
ix=0
iy=0
xx=0
yy=0
#######################

class Back_Sub:

    def __init__(self,b,l,t,r):
        '''

        :param b: x1
        :param l: y1
        :param t: x2
        :param r: y2
        '''
        # global variables
        self.bg = None
        self.top = t
        self.bottom = b
        self.right = r
        self.left = l
        self.gray = None
        self.aWeight=0.5
        self.pb=0
        # self.run_avg()
        # self.segment()

    #-------------------------------------------------------------------------------
    # Function - To find the running average over the background
    #-------------------------------------------------------------------------------
    def run_avg(self, image, aWeight):

        self.bg

        # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, aWeight)

    #-------------------------------------------------------------------------------
    # Function - To segment the region of hand in the image
    #---
    # ----------------------------------------------------------------------------
    def segment(self, image, threshold=25):

        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff,
                                    threshold,
                                    255,
                                    cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def make_roi(self, frame,clone,num_frames,n):
        aWeight=0.5
        # thresholded=None

        # get the ROI
        '''
        top,bottom,right,left = mouse call back
        '''
        roi = frame[self.bottom:self.top, self.left:self.right]

        # convert the roi to grayscale and blur it
        self.gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 10:
            self.run_avg(self.gray, aWeight)
        else:
            # segment the hand region
            hand = self.segment(self.gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand
                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (self. left, self.bottom)], -1, (0, 0, 255))
                cv2.imshow("%d"%n, thresholded)

                #find how many non zero
                shape = thresholded.shape
                size = thresholded.size
                img = np.zeros(shape, np.uint8)
                res = cv2.bitwise_or(thresholded, img)
                nz=np.count_nonzero(res)
                self.pb=nz/size*100

        # draw the segmented hand
        cv2.rectangle(clone, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)
        return

def mouse_select(event, x, y, flags, param):

    global bl,tr,f,ix,iy,xx,yy, chk

    if event == cv2.EVENT_LBUTTONDOWN:
        chk=True
        bl=(x,y)
        xx,yy=x,y

    if event == cv2.EVENT_MOUSEMOVE:
        ix,iy=x,y

    if event == cv2.EVENT_LBUTTONUP:
        chk=False
        tr=(x,y)
        #save bltr
        BS.append(Back_Sub(bl[1],bl[0],tr[1],tr[0]))

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
def backsub():

    ###################
    #####park name#####
    park_name = 'park1'
    ###################

    global ix,iy,xx,yy,chk

    # initialize weight for running average
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    '''
    coord=mouse callback
    top, right, bottom, left = coord
    '''
    # initialize num of frames
    num_frames = 0
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        # s.connect(('127.0.0.1',4000))

        while True:
            _,f = camera.read()
            f = cv2.flip(f,1)
            cv2.setMouseCallback('select', mouse_select)
            if chk==True and ix != 0 and iy !=0:
                cv2.rectangle(f,(ix,iy),(xx,yy),(0,255,0),2)
            cv2.imshow('select',f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        # make roi and background sub

        # keep looping, until interrupted
        while True:
            # get the current frame
            (_, frame) = camera.read()

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            data=''
            n = 0
            for B in BS:
                B.make_roi(frame, clone, num_frames, n)
                # print(B.pb)
                if B.pb > 50 :
                    data+='1'
                else :
                    data+='0'
                n += 1
                num_frames += 1
            data=park_name+'/@1#/'+str(len(BS))+'/@!#/'+data
            # print(data)
            try:
                s.send(bytes(data,encoding='utf-8'))
            except:
                if chk == False:
                    print('socket error')
                    chk = True

            cv2.imshow("Video Feed", clone)

            if cv2.waitKey(1) & 0xFF == ord('w'):
                for B in BS:
                    B.run_avg(B.gray, B.aWeight)

            # if the user pressed "q", then stop looping
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # free up memory
    s.close()
    print('socket closed')
    camera.release()
    cv2.destroyAllWindows()
    return

backsub()
