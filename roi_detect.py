# organize imports
import cv2
import imutils
import numpy as np

chk = True
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
        global chk
        aWeight=0.5

        # thresholded=None

        # get the ROI
        '''
        top,bottom,right,left = mouse call back
        '''
        # print(self.top,self.bottom, self.right,self.left)
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
                cv2.drawContours(clone, [segmented + (self.left, self.bottom)], -1, (0, 0, 255))
                # print("Thesholded%d"%n)
                cv2.imshow("%d"%n, thresholded)


        # draw the segmented hand
        cv2.rectangle(clone, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)
        if chk == True:
            print(bl,tr)
            print(self.bottom,self.left,self.top,self.right)
            chk = False

        return

bl=[]
tr=[]
region=[]
BS=[]

def mouse_select(event, x, y, flags, param):

    global bl,tr

    if event == cv2.EVENT_LBUTTONDOWN:
        bl=(x,y)

    if event == cv2.EVENT_LBUTTONUP:
        tr=(x,y)
        # region.append((bl,tr))
        print(bl[0],tr[0],bl[1],tr[1])

        #save bltr
        BS.append(Back_Sub(bl[1],bl[0],tr[1],tr[0]))
        # print(BS[0].bottom,BS[0].left,BS[0].top,BS[0].right)
#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
def backsub():
    # initialize weight for running average
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    '''
    coord=mouse callback
    top, right, bottom, left = coord
    '''
    # initialize num of frames
    num_frames = 0
    while True:
        _,f = camera.read()
        f = cv2.flip(f,1)
        cv2.imshow('select',f)
        cv2.setMouseCallback('select', mouse_select)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    # make roi and background sub
    # BS1 = Back_Sub(0,0,200,300)
    # BS2 = Back_Sub(200,300,300,400)

    # keep looping, until interrupted
    while True:
        # get the current frame
        (_, frame) = camera.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # BS1.make_roi(frame, clone, num_frames,n)
        # BS2.make_roi(frame, clone, num_frames,n+1)
        # num_frames+=1
        n = 0
        for B in BS:
            B.make_roi(frame, clone, num_frames, n)
            n += 1
            num_frames += 1

        cv2.imshow("Video Feed", clone)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            # BS1.run_avg(BS1.gray, BS1.aWeight)
            # BS2.run_avg(BS2.gray, BS2.aWeight)
            for B in BS:
                B.run_avg(B.gray, B.aWeight)
            print("[press W iwn] : " + str(n))

        #TODO 잘 안꺼짐
        # if the user pressed "q", then stop looping
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()
    return

backsub()
