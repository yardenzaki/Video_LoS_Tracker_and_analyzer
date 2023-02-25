import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import pandas as pd
from mss import mss
import time
from scipy.signal import sosfiltfilt, butter
from matplotlib.widgets import RectangleSelector
import tkinter as tk
import datetime
from scipy import stats, signal
from tkinter import filedialog
import ctypes
import imutils
from imutils import paths
import sys
# from PyQt5 import QtTest
import seaborn as sns
from scipy.stats import norm
from matplotlib.widgets import RectangleSelector, Slider, Button, RadioButtons


## create exe ----
# pip install pyinstaller
# cd to the script directory
# pyinstaller Boresight_GUI.py

## Videos to test ----
# no drift -- https://youtu.be/9F4uhlGqnAg
# with drift -- https://youtu.be/E-itDPK4n9A


def apply_brightness_contrast(bgr_ff, input_img, brightness=255, contrast=127, thresh_limit=127, Gauss_filter=1,
                              min_contour=5000, max_contour=50000,
                              ilowH=0, ilowS=0, ilowV=0, ihighH=179, ihighS=255, ihighV=255):
    global param_dict

    BGR_buf = bgr_ff

    param_dict = {}
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    # GRAY TRESHOLDING ------------------------
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    if thresh_limit != 0:
        f = float(thresh_limit)
        buf = cv.threshold(buf, f, 255, cv.THRESH_BINARY)[1]
    if Gauss_filter != 0:
        f = int(Gauss_filter)
        if f % 2 == 0:
            f = f + 1
        buf = cv.GaussianBlur(buf, (f, f), 0)
    if min_contour != 0:
        min_contour = int(min_contour)
    if max_contour != 0:
        max_contour = int(max_contour)

    cnts = cv.findContours(buf, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    n = 0
    for c in cnts:
        # print("Found Contours:", "with Area:", cv.contourArea(c))
        if cv.contourArea(c) < max_contour and cv.contourArea(c) > min_contour:
            print("Contour Area", cv.contourArea(c))
            cv.drawContours(buf, [c], 0, (150, 150, 150), 2)
            n = n + 1

    # HSV TRESHOLDING -----------------------------------------------
    hsv = cv.cvtColor(BGR_buf, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv.inRange(hsv, lower_hsv, higher_hsv)
    buf2 = cv.bitwise_and(BGR_buf, BGR_buf, mask=mask)
    if Gauss_filter != 0:
        f = int(Gauss_filter)
        if f % 2 == 0:
            f = f + 1
        buf2 = cv.GaussianBlur(buf2, (f, f), 0)
    if min_contour != 0:
        min_contour = int(min_contour)
    if max_contour != 0:
        max_contour = int(max_contour)
    if thresh_limit != 0:
        f = float(thresh_limit)
        buf2 = cv.cvtColor(buf2, cv.COLOR_BGR2GRAY)
        buf2 = cv.threshold(buf2, f, 255, cv.THRESH_BINARY)[1]

        # find CNTS only if TRESH !=0 otherwisw the pic is BGR and no CNTS
        cnts = cv.findContours(buf2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        n_cont_buf2 = 0
        for c in cnts:
            # print("Found Contours:", "with Area:", cv.contourArea(c))
            if cv.contourArea(c) < max_contour and cv.contourArea(c) > min_contour:
                try:
                    print("Contour Area buf2", cv.contourArea(c))
                    cv.drawContours(buf2, [c], 0, (220, 220, 220), 2)
                    n_cont_buf2 = n_cont_buf2 + 1
                except:
                    print("Error drawing contours")
                    pass
    else:
        n_cont_buf2 = 0

    # Adding On Figures ------------------------------------------------------------
    cv.putText(buf2,
               'H:{}-{},S:{}-{},V:{}-{},T:{},Cntrs:{}'.format(ilowH, ihighH, ilowS, ihighS, ilowV, ihighV, thresh_limit,
                                                              n_cont_buf2),
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(buf, 'B:{},C:{},T:{},GB:{},Cntrs:{}'.format(brightness, contrast, thresh_limit, Gauss_filter, n),
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    param_dict["brightness"] = brightness
    param_dict["contrast"] = contrast
    param_dict["thresh_limit"] = thresh_limit
    param_dict["Gauss_filter"] = Gauss_filter
    param_dict["min_contour"] = min_contour
    param_dict["max_contour"] = max_contour
    param_dict["N_contours"] = n
    param_dict["ilowH"] = ilowH
    param_dict["ihighH"] = ihighH
    param_dict["ilowS"] = ilowS
    param_dict["ihighS"] = ihighS
    param_dict["ilowV"] = ilowV
    param_dict["ihighV"] = ihighV
    param_dict["N_contours_HSV"] = n_cont_buf2

    return buf, buf2


def map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def signal_stats(a, b, num):
    """
    a is the time column.
    b is the amplitude column.
    num is the number of coordinates
    Return
          sr - sample rate
          dt - time step
        mean - average
          sd - standard deviation
         rms - root mean square
        skew - skewness
    kurtosis - peakedness
         dur - duration
    """
    bmax = max(b)
    bmin = min(b)

    ave = np.mean(b)

    dur = a[num - 1] - a[0];

    dt = dur / float(num - 1)
    sr = 1 / dt

    rms = np.sqrt(np.var(b))
    sd = np.std(b)

    skewness = stats.skew(b)
    kurtosis = stats.kurtosis(b, fisher=False)

    to_prnt = "\n max = %8.4g  min=%8.4g \n" % (bmax, bmin)
    self.lines.append(to_prnt + '\n')

    to_prnt = "     mean = %8.4g " % ave
    self.lines.append(to_prnt + '\n')
    to_prnt = "  std dev = %8.4g " % sd
    self.lines.append(to_prnt + '\n')
    to_prnt = "      rms = %8.4g " % rms
    self.lines.append(to_prnt + '\n')
    to_prnt = " skewness = %8.4g " % skewness
    self.lines.append(to_prnt + '\n')
    to_prnt = " kurtosis = %8.4g " % kurtosis
    self.lines.append(to_prnt + '\n')

    to_prnt = "\n  start = %8.4g sec  end = %8.4g sec" % (a[0], a[num - 1])
    self.lines.append(to_prnt + '\n')
    to_prnt = "    dur = %8.4g sec \n" % dur
    self.lines.append(to_prnt + '\n')
    to_prnt = "    sample rate = %8.4g Hz \n" % sr
    self.lines.append(to_prnt + '\n')
    return sr, dt, ave, sd, rms, skewness, kurtosis, dur


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # self.lines.append(f"Clicked (x,y) {(x, ' ', y)} \n")

        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(calibration_fig, str(x) + ',' +
                   str(y), (x, y), font,
                   1, (0, 0, 255), 2)
        calibration_clicks.append((x, y))
        cv.imshow('Pick points to calibrate', calibration_fig)
        if len(calibration_clicks) == 2:
            cv.imshow('Pick points to calibrate', calibration_fig)
            cv.waitKey(1000)
            cv.destroyAllWindows()

    # checking for right mouse clicks
    if event == cv.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        cv.imsave('calibration.jpg', calibration_fig)


def screen_record_efficient(mon) -> int:
    title = "[MSS] FPS benchmark"
    fps = 0
    sct = mss()
    last_time = time.time()

    while time.time() - last_time < 1:
        img = np.asarray(sct.grab(mon))
        fps += 1

        # cv.imshow(title, img)
        if cv.waitKey(25) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break

    return fps


def select_ROI(frame):
    """
    Select a ROI and then press SPACE or ENTER button!
    Cancel the selection process by pressing c button!
    Finish the selection process by pressing ESC button!

    """
    fromCenter = False
    ROIs = cv.selectROIs('Select ROIs', frame, fromCenter)

    return ROIs


def line_select_callback_dx(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    from_frame = round(max(x1, np.array(time_lst).min())) - 1
    to_frame = round(min(x2, np.array(time_lst).max())) + 1

    # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    print(f"DX {from_frame:3.0f} --> {to_frame:3.0f}")
    self.lines.append(f"DX {from_frame:3.0f} --> {to_frame:3.0f} \n")
    time_cropped = data_df.loc[(data_df['time'] > from_frame) & (data_df['time'] < to_frame)]["time"]
    dx_cropped = data_df.loc[(data_df['time'] > from_frame) & (data_df['time'] < to_frame)]["dx_pix_filtered"]
    a, b = np.polyfit(time_cropped, dx_cropped, 1)
    # print(dy_cropped)

    # DETREND_Center_X_cropped = signal.detrend(Center_X_cropped)
    # RMS = round(DETREND_Center_X_cropped.std(), 2)

    stats_str = f" Selection linear fitting (y=a*x+b):   a={a} , b={b}"
    self.lines.append(stats_str + "\n")
    ax2.set_title(stats_str)


def line_select_callback_dy(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    from_frame = round(max(x1, np.array(time_lst).min())) - 1
    to_frame = round(min(x2, np.array(time_lst).max())) + 1

    # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    print(f"{from_frame:3.0f} --> {to_frame:3.0f}")
    self.lines.append(f"DY {from_frame:3.0f} --> {to_frame:3.0f} \n")
    time_cropped = data_df.loc[(data_df['time'] > from_frame) & (data_df['time'] < to_frame)]["time"]
    dy_cropped = data_df.loc[(data_df['time'] > from_frame) & (data_df['time'] < to_frame)]["dy_pix_filtered"]
    a, b = np.polyfit(time_cropped, dy_cropped, 1)
    # print(dy_cropped)

    # DETREND_Center_X_cropped = signal.detrend(Center_X_cropped)
    # RMS = round(DETREND_Center_X_cropped.std(), 2)

    stats_str = f" Selection linear fitting (y=a*x+b):   a={a} , b={b}"
    self.lines.append(stats_str + "\n")
    ax3.set_title(stats_str)

    # fig_title = center_title + "_Selected_ranges.png"
    # plt.savefig(fig_title)


class App:
    global detection_algorithm, thresh_limit

    # plt.style.use('ggplot')

    def __init__(self, root):
        self.EndFrame = "inf"
        self.error_occured = False
        self.CALIBRATED_FLAG = False
        self.global_wd = os.getcwd()

        # initialize live plotter line
        self.line1 = []

        # intitialize logfile
        self.lines = []
        self.root = root
        self.root.title("Video Extensiometer and LoS Measurement | by Yarden Zaki")
        self.root.geometry("650x700")  # set window size to 500x500

        # create menu label for radio buttons
        self.menu_label = tk.Label(root, text="*** Input Source ***", font=(14), anchor="w")
        self.menu_label.pack(pady=5)

        # create entry label and widget
        self.screen_sample_entry_label = tk.Label(root, text="Sample monitor every: [seconds]", font=('Arial', 9),
                                                  anchor="w")
        self.screen_sample_entry = tk.Entry(root, width=5)
        self.screen_sample_entry_label.place(x=160, y=45)
        self.screen_sample_entry.place(x=360, y=45)
        self.default_val_sampling = 0
        self.screen_sample_entry.insert(0, self.default_val_sampling)  # set default value for entry widget

        # create radio buttons
        self.Input_source = tk.IntVar()
        self.monitor_rb = tk.Radiobutton(root, text="Monitor Sampling", font=(14), variable=self.Input_source, value=0,
                                         command=self.show_entry)
        self.file_rb = tk.Radiobutton(root, text="File", font=(14), variable=self.Input_source, value=1,
                                      command=self.load_file)

        self.monitor_rb.pack(pady=5, anchor="w")
        self.file_rb.pack(pady=5, anchor="w")
        ##############---------------------------------------------------
        # create menu label for radio buttons
        self.menu_label = tk.Label(root, text="*** Task ***", font=(14), anchor="w")
        self.menu_label.pack(pady=5)

        # create radio buttons
        self.task = tk.IntVar()
        self.extensiometer_rb = tk.Radiobutton(root, text="Digital Extensiometer", font=(14), variable=self.task,
                                               value=0, command=self.change_rb_state)
        self.los_rb = tk.Radiobutton(root, text="LoS Stability", font=(14), variable=self.task, value=1,
                                     command=self.change_rb_state)
        self.extensiometer_rb.pack(pady=5, anchor="w")
        self.los_rb.pack(pady=5, anchor="w")
        #############---------------------------------------------

        # create menu label for radio buttons
        self.menu_label = tk.Label(root, text="*** Detection Algorithm ***", font=(14), anchor="w")
        self.menu_label.pack(pady=5)

        # create radio buttons
        self.DetectAlgo = tk.IntVar()
        self.radio_button1 = tk.Radiobutton(root, text="Brightest spot", font=(14), variable=self.DetectAlgo, value=0,
                                            command=self.hide_entry)
        self.radio_button2 = tk.Radiobutton(root, text="Average Intensity", font=(14), variable=self.DetectAlgo,
                                            value=1, command=self.show_entry)
        self.radio_button3 = tk.Radiobutton(root, text="Template Matching", font=(14), variable=self.DetectAlgo,
                                            value=2, command=self.show_entry)
        self.radio_button4 = tk.Radiobutton(root, text="Contour Center", font=(14), variable=self.DetectAlgo, value=3,
                                            command=self.show_entry)
        self.radio_button1.pack(pady=5, anchor="w")
        self.radio_button2.pack(pady=5, anchor="w")
        self.radio_button3.pack(pady=5, anchor="w")
        self.radio_button4.pack(pady=5, anchor="w")

        # DISABLE FOR NOW ----
        self.radio_button1.config(state="disabled")
        self.radio_button2.config(state="disabled")
        self.radio_button4.select()
        # ------------------------

        # create menu label for radio buttons
        self.options_label = tk.Label(root, text="*** Options ***", font=(16), anchor="w")
        self.options_label.pack(pady=10)

        # Show Tips:
        self.checkvar_TIP = tk.IntVar()
        self.check_button_TIP = tk.Checkbutton(root, text="Show Instructions Pop-Ups", font=(11),
                                               variable=self.checkvar_TIP)
        self.check_button_TIP.pack(pady=5, anchor="w")
        # create inverting label:
        self.checkvar_SaveVideo = tk.IntVar()
        self.check_button = tk.Checkbutton(root, text="Save Analyzed Video", font=(11),
                                           variable=self.checkvar_SaveVideo)
        self.check_button.pack(pady=5, anchor="w")
        # create error handling check var:
        self.checkvar_error_occured = tk.IntVar()
        self.check_button_error_occured = tk.Checkbutton(root, text="Pause script on detection error", font=(7),
                                                         variable=self.checkvar_error_occured)
        self.check_button_error_occured.pack(pady=5, anchor="w")

        # create buttons
        self.load_button = tk.Button(root, text="Load\n Input", font=('Arial', 14), command=self.open_media_first_frame)
        self.calibration_button = tk.Button(root, text="Calibrate\n", font=('Arial', 14), command=self.calibrate)
        self.set_params_button = tk.Button(root, text="Set\n Parameters", font=('Arial', 14), command=self.set_params)
        self.execute_button = tk.Button(root, text="Execute\n", font=('Arial', 14), command=self.execute)
        self.exit_button = tk.Button(root, text="Exit\n", font=('Arial', 14), command=self.exit)
        self.work_button = tk.Button(root, text="Work\n Dir", font=('Arial', 14), command=self.open_cwd)
        self.post_button = tk.Button(root, text="Post\n Processing", font=('Arial', 14), command=self.POST_PROC)

        self.load_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.calibration_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.set_params_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.execute_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.work_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.post_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.exit_button.pack(side="left", padx=5, pady=20, anchor="w")

    def live_plotter(self, x, y):
        # initialization:
        if self.line1 == []:
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            self.line1, = ax.plot(x, y, '-o', alpha=0.8)
            plt.ylabel("LoS")
            plt.title(f"Live Plotting -- {self.task_legend[self.task]}")
            plt.show()
        self.line1.set_ydata(y)
        if np.min(y) <= self.line1.axes.get_ylim()[0] or np.max(y) >= self.line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y) - np.std(y), np.max(y) + np.std(y)])
        plt.pause(0.01)

    def create_folder(self):

        source = self.source_legend[self.Input_source]
        task = self.task_legend[self.task]

        image_folder_name = '\Result'
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_folder_name = image_folder_name + source + task + time_stamp
        path = os.getcwd()
        Newdir_path = path + image_folder_name

        try:
            os.mkdir(Newdir_path)
            if self.CALIBRATED_FLAG:
                os.rename(path + "/Calibration.jpg", Newdir_path + "/Calibration.jpg")
            if self.GRAY_TRESH:
                os.rename(path + "/GRAY_TRESH_image_params.jpg", Newdir_path + "/GRAY_TRESH_image_params.jpg")
                print("moved GRAY_TRESH_image_params.jpg")
            if self.HSV_TRESH:
                os.rename(path + "/HSV_TRESH_image_params.jpg", Newdir_path + "/HSV_TRESH_image_params.jpg")
                print("moved HSV_TRESH_image_params.jpg")


        except OSError:
            print("Creation of the directory %s failed" % Newdir_path)

        os.chdir(Newdir_path)
        print("current dir.", os.getcwd())

    def funcBrightContrast(self, bright=0):

        method = cv.getTrackbarPos('Method', 'EditBrightContrast')

        bright = cv.getTrackbarPos('Bright', 'EditBrightContrast')
        contrast = cv.getTrackbarPos('Contrast', 'EditBrightContrast')
        thresh_limit = cv.getTrackbarPos('Threshold', 'EditBrightContrast')
        Gauss_filter = cv.getTrackbarPos('GaussBlur', 'EditBrightContrast')
        min_contour = cv.getTrackbarPos('MinArea', 'EditBrightContrast')
        max_contour = cv.getTrackbarPos('MaxArea', 'EditBrightContrast')

        # get trackbar positions
        ilowH = cv.getTrackbarPos('lowH', 'EditBrightContrast')
        ihighH = cv.getTrackbarPos('highH', 'EditBrightContrast')
        ilowS = cv.getTrackbarPos('lowS', 'EditBrightContrast')
        ihighS = cv.getTrackbarPos('highS', 'EditBrightContrast')
        ilowV = cv.getTrackbarPos('lowV', 'EditBrightContrast')
        ihighV = cv.getTrackbarPos('highV', 'EditBrightContrast')

        # get Frame number to show on screen
        if self.Input_source.get() == 1:  # File
            selected_frame = cv.getTrackbarPos('Frame#', 'EditBrightContrast')
            cap = cv.VideoCapture(self.file_path)
            cap.set(cv.CAP_PROP_POS_FRAMES, selected_frame)
            _, self.bgr_ff = cap.read()
            ROIs = self.bounding_box
            self.bgr_ff = self.bgr_ff[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
            self.sct_img_temp = cv.cvtColor(np.array(self.bgr_ff), cv.COLOR_BGR2GRAY)
        else:
            pass

        Result, Result2 = apply_brightness_contrast(self.bgr_ff, self.sct_img_temp, bright, contrast, thresh_limit,
                                                    Gauss_filter,
                                                    min_contour,
                                                    max_contour, ilowH, ilowS, ilowV, ihighH, ihighS, ihighV)
        if method == 0:
            cv.imshow('Result', Result)
            cv.imwrite('GRAY_TRESH_image_params.jpg', Result)
            self.GRAY_TRESH = True
            self.HSV_TRESH = False
            try:
                cv.destroyWindow("HSV")
            except:
                pass

        if method == 1:
            cv.imshow('HSV', Result2)
            cv.imwrite('HSV_TRESH_image_params.jpg', Result2)
            self.HSV_TRESH = True
            self.GRAY_TRESH = False
            try:
                cv.destroyWindow("Result")
            except:
                pass

    def open_cwd(self):
        cwd = os.getcwd()
        print("path", cwd)
        os.startfile(cwd)

    def onChangeLmts(self, trackbarValue):
        cap = cv.VideoCapture(self.file_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, trackbarValue)
        err, img = cap.read()
        cv.imshow("SelectMediaLimits", img)
        pass

    def select_media_limits(self):

        if self.checkvar_TIP.get() == 1:
            ctypes.windll.user32.MessageBoxW(0,
                                             "Select media frame range to analyze.\n\nFor 'Digital Extensiometer' task, make sure that the first frame\n is the desired initial state\n\nFor 'LoS Stability' task, make sure to choose frame range with representative jitter",
                                             "Info", 0)

        cap = cv.VideoCapture(self.file_path)
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        cv.namedWindow("SelectMediaLimits", cv.WINDOW_NORMAL)
        cv.createTrackbar('start', 'SelectMediaLimits', 0, length, self.onChangeLmts)
        cv.createTrackbar('end', 'SelectMediaLimits', 10, length, self.onChangeLmts)
        # cv.resizeWindow("SelectMediaLimits", 1080, 720)

        self.onChangeLmts(0)
        cv.waitKey()

        start = cv.getTrackbarPos('start', 'SelectMediaLimits')
        end = cv.getTrackbarPos('end', 'SelectMediaLimits')

        print("start,end", start, end)
        self.lines.append(f"Selected Frames: {start} - {end} \n")
        if start >= end:
            raise Exception("start must be less than end")

        cap.set(cv.CAP_PROP_POS_FRAMES, start)
        while cap.isOpened():
            err, img = cap.read()
            if cap.get(cv.CAP_PROP_POS_FRAMES) >= end:
                break
            cv.imshow("SelectMediaLimits", img)
            k = cv.waitKey(10) & 0xff
            if k == 27:
                break
            if (cv.waitKey(10) & 0xFF) == ord('q'):
                break

        self.StartFrame = int(start)
        self.EndFrame = int(end)

    def load_file(self):
        # Hiding screen_sample_entry for screen sampling
        self.screen_sample_entry.place_forget()  # hide entry widget initially
        self.screen_sample_entry_label.place_forget()  # hide entry label initially

        self.lines.append("Loading file --------------------------\n")

        print("LOAD FILE")
        self.file_path = filedialog.askopenfilename()
        self.lines.append(f"File Path: {self.file_path} \n ")
        print("file_path", self.file_path)
        input_source_flag = self.Input_source.get()
        if input_source_flag == 1:
            self.select_media_limits()
            ctypes.windll.user32.MessageBoxW(0,
                                             f"Range Selected {self.StartFrame} --> {self.EndFrame}.\n\n Continue to 'Load Input'",
                                             "Info", 0)

            # # create starting frame window
            # self.start_frame_window = tk.Toplevel(self.root)
            # self.start_frame_window.geometry("350x100")  # set window size
            # self.start_frame_window.title("Enter Desired Media Range")
            #
            # tk.Label(self.start_frame_window, text="Start From (frame):").pack()
            # self.InitialFrameNum_entry = tk.Entry(self.start_frame_window)
            # self.InitialFrameNum_entry.pack()
            # self.InitialFrameNum_entry.insert(0, 1)  # set default value for entry widget
            #
            # tk.Label(self.start_frame_window, text="Ends at (frame):").pack()
            # self.LastFrameNum_entry = tk.Entry(self.start_frame_window)
            # self.LastFrameNum_entry.pack()
            # self.LastFrameNum_entry.insert(0, 10000)  # set default value for entry widget
            #
            # tk.Button(self.start_frame_window, text="OK", command=self.set_media_lmts).pack()

        self.lines.append(f"Start Frame: {self.StartFrame} ----- End Frame:{self.EndFrame}\n")
        self.lines.append("END Loading file --------------------------\n")

    # def set_media_lmts(self):
    #     self.StartFrame = int(self.InitialFrameNum_entry.get())
    #     self.EndFrame = int(self.LastFrameNum_entry.get())
    #
    #     self.lines.append(f"Selected range: from frame {self.StartFrame} to frame {self.EndFrame}\n")
    #     print(f"Selected range: from frame {self.StartFrame} to frame {self.EndFrame}\n")
    #     self.start_frame_window.destroy()

    def hide_entry(self):
        # hide entry widget and label if radio button 1 is selected
        # self.entry.pack_forget()
        # self.entry_label.pack_forget()
        return

    def change_rb_state(self):
        task = self.task.get()
        if task == 0:  # digital extensiometer
            self.radio_button1.config(state="disabled")
            self.radio_button2.config(state="disabled")
            self.radio_button3.config(state="normal")
            self.radio_button4.config(state="normal")
            self.radio_button4.select()

        if task == 1:  # LoS
            self.radio_button1.config(state="disabled")
            self.radio_button2.config(state="disabled")
            self.radio_button3.config(state="normal")
            self.radio_button4.config(state="normal")
            self.radio_button4.select()

    def show_entry(self):
        self.default_val_sampling = self.screen_sample_entry.get()
        detection_algorithm = self.DetectAlgo.get()
        # if detection_algorithm == 1:
        #     # show entry widget and label if radio button 2 is selected
        #     self.entry_label.pack(side="left", pady=5, padx=5, anchor="w")
        #     self.entry.pack(side="left", pady=5, padx=5, anchor="w")
        # if detection_algorithm == 2:
        #     # self.entry.pack_forget()
        #     # self.entry_label.pack_forget()
        if self.Input_source.get() == 0:
            # self.entry_label.pack(pady=5)
            # self.entry.pack(pady=5)
            self.screen_sample_entry_label.place(x=160, y=45)
            self.screen_sample_entry.place(x=360, y=45)
            self.screen_sample_entry.delete(0, 'end')
            self.screen_sample_entry.insert(0, self.default_val_sampling)  # set default value for entry widget
            print("self.screen_sample_entry", self.screen_sample_entry.get())

    def set_params(self):
        if self.checkvar_TIP.get() == 1:
            ctypes.windll.user32.MessageBoxW(0, "Drag scrollbars to apply target segmentation", "Info", 0)
        self.bgr_ff = self.firstframe
        self.sct_img_temp = cv.cvtColor(np.array(self.firstframe), cv.COLOR_BGR2GRAY)
        cv.namedWindow('EditBrightContrast', cv.WINDOW_NORMAL)
        cv.resizeWindow('EditBrightContrast', 300, 200)
        # cv.setWindowProperty("EditBrightContrast",cv.WND_PROP_AUTOSIZE,cv.WND_PROP_AUTOSIZE)

        bright = 255
        contrast = 127
        thresh_limit = 0
        Gauss_filter = 10
        min_contour = 5000
        max_countour = 50000

        # HSV
        ilowH = 0
        ihighH = 179
        ilowS = 0
        ihighS = 255
        ilowV = 0
        ihighV = 255

        # Brightness value range -255 to 255
        # Contrast value range -127 to 127
        # threshold value range -127 to 127
        cv.createTrackbar('Bright', 'EditBrightContrast', bright, 2 * 255, self.funcBrightContrast)
        cv.createTrackbar('Contrast', 'EditBrightContrast', contrast, 2 * 127, self.funcBrightContrast)
        cv.createTrackbar('Threshold', 'EditBrightContrast', thresh_limit, 2 * 127, self.funcBrightContrast)
        cv.createTrackbar('GaussBlur', 'EditBrightContrast', Gauss_filter, 2 * 10, self.funcBrightContrast)
        cv.createTrackbar('MinArea', 'EditBrightContrast', min_contour, 2 * 5000, self.funcBrightContrast)
        cv.createTrackbar('MaxArea', 'EditBrightContrast', max_countour, 2 * 50000, self.funcBrightContrast)

        # create trackbars for color change
        cv.createTrackbar('lowH', 'EditBrightContrast', ilowH, 179, self.funcBrightContrast)
        cv.createTrackbar('highH', 'EditBrightContrast', ihighH, 179, self.funcBrightContrast)

        cv.createTrackbar('lowS', 'EditBrightContrast', ilowS, 255, self.funcBrightContrast)
        cv.createTrackbar('highS', 'EditBrightContrast', ihighS, 255, self.funcBrightContrast)

        cv.createTrackbar('lowV', 'EditBrightContrast', ilowV, 255, self.funcBrightContrast)
        cv.createTrackbar('highV', 'EditBrightContrast', ihighV, 255, self.funcBrightContrast)

        print("self.Input_source=", self.Input_source)
        if self.Input_source.get() == 1:  # File
            self.Selectedlength = self.EndFrame - self.StartFrame
            print("self.Selectedlength = self.EndFrame - self.StartFrame", self.Selectedlength,
                  self.EndFrame, self.StartFrame)
            cv.createTrackbar('Frame#', 'EditBrightContrast', self.StartFrame, self.Selectedlength,
                              self.funcBrightContrast)
        cv.createTrackbar('Method', 'EditBrightContrast', 0, 1, self.funcBrightContrast)

        self.funcBrightContrast(0)
        help_img = cv.imread(os.path.join(self.global_wd, "instructions.jpg"))
        # help_img2 = cv.resize(help_img, (500, 300))
        cv.imshow('EditBrightContrast', help_img)
        cv.waitKey(0)
        self.ParamDct = param_dict
        param_dict_str = ""
        for k, v in self.ParamDct.items():
            param_dict_str = param_dict_str + str(k) + "=" + str(v) + "\n"
        print("param_dict_str", param_dict_str)
        self.lines.append(f"User Parameters -------------------- \n")
        self.lines.append(f"param_dict: \n {param_dict_str} \n")
        if self.GRAY_TRESH:
            self.lines.append(f"Tresholding Method: GRAY TRESH\n")
        if self.HSV_TRESH:
            self.lines.append(f"Tresholding Method: HSV TRESH\n")
        self.lines.append(f"END User Parameters -------------------- \n")

    def apply_resulted_brightness_contrast(self, input_img):

        brightness = self.ParamDct["brightness"]
        contrast = self.ParamDct["contrast"]
        thresh_limit = self.ParamDct["thresh_limit"]
        Gauss_filter = self.ParamDct["Gauss_filter"]
        min_contour = self.ParamDct["min_contour"]
        max_contour = self.ParamDct["max_contour"]
        n = self.ParamDct["N_contours"]
        ilowH = self.ParamDct["ilowH"]
        ihighH = self.ParamDct["ihighH"]
        ilowS = self.ParamDct["ilowS"]
        ihighS = self.ParamDct["ihighS"]
        ilowV = self.ParamDct["ilowV"]
        ihighV = self.ParamDct["ihighV"]
        n_cont_buf2 = self.ParamDct["N_contours_HSV"]

        if self.GRAY_TRESH:
            # print("GRAY TRESHOLDING!!!!!!!!!!!")
            if brightness != 0:
                if brightness > 0:
                    shadow = brightness
                    highlight = 255
                else:
                    shadow = 0
                    highlight = 255 + brightness
                alpha_b = (highlight - shadow) / 255
                gamma_b = shadow
                buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
            else:
                buf = input_img.copy()
            if contrast != 0:
                f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
                alpha_c = f
                gamma_c = 127 * (1 - f)
                buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)
            if thresh_limit != 0:
                f = float(thresh_limit)
                buf = cv.threshold(buf, f, 255, cv.THRESH_BINARY)[1]
            if Gauss_filter != 0:
                f = int(Gauss_filter)
                if f % 2 == 0:
                    f = f + 1
                buf = cv.GaussianBlur(buf, (f, f), 0)

            grabbed_edited = buf

        if self.HSV_TRESH:
            # print("HSV TRESHOLDING!!!!!!!!!!!")
            # HSV TRESHOLDING -----------------------------------------------
            hsv = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)
            lower_hsv = np.array([ilowH, ilowS, ilowV])
            higher_hsv = np.array([ihighH, ihighS, ihighV])
            mask = cv.inRange(hsv, lower_hsv, higher_hsv)
            grabbed_edited = cv.bitwise_and(input_img, input_img, mask=mask)
            if Gauss_filter != 0:
                f = int(Gauss_filter)
                if f % 2 == 0:
                    f = f + 1
                grabbed_edited = cv.GaussianBlur(grabbed_edited, (f, f), 0)
            if thresh_limit != 0:
                f = float(thresh_limit)
                grabbed_edited = cv.cvtColor(grabbed_edited, cv.COLOR_BGR2GRAY)
                grabbed_edited = cv.threshold(grabbed_edited, f, 255, cv.THRESH_BINARY)[1]

        return grabbed_edited

    def open_media_first_frame(self):
        if self.checkvar_TIP.get() == 1:
            ctypes.windll.user32.MessageBoxW(0, "Drag the mouse to crop Region of Interest for analysis", "Info", 0)

        input_source_flag = self.Input_source.get()
        global sct_img

        if input_source_flag == 0:  # MONITOR
            self.sct = mss()
            sct_img = self.sct.grab(self.sct.monitors[1])
            sct_img = np.asarray(sct_img)
            sct_img = cv.cvtColor(sct_img, cv.COLOR_BGRA2BGR)

        if input_source_flag == 1:  # File Input
            self.cap = cv.VideoCapture(self.file_path)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.StartFrame)  # starts from frame
            ret, sct_img = self.cap.read()

        ROIs = select_ROI(sct_img)
        print("Bounding Box FF", ROIs, type(ROIs), ROIs[0][0])
        sct_img = sct_img[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
        cv.destroyAllWindows()  # close the main select roi window
        self.bounding_box = ROIs

        # bounding_box = {'top': int(ROIs[0][1]), 'left': int(ROIs[0][0]), 'width': int(ROIs[0][2]),'height': int(ROIs[0][3])}
        self.firstframe = sct_img

        # cv.imshow("ff", self.firstframe)
        # cv.waitKey(0)

    def calibrate(self):
        self.lines.append("Start Calibration  --------------------------\n")
        global calibration_clicks
        calibration_clicks = []
        # implement your calibration code here
        global sct_img

        # print(type(sct_img))

        # ("np.mean(sct_img)",np.mean(sct_img))
        try:
            self.lines.append(f"mean of image: {np.mean(self.firstframe)} \n")
        except AttributeError:
            ctypes.windll.user32.MessageBoxW(0, "ERROR: YOU MUST PRESS ON 'LOAD INPUT'", "Info", 0)
            print("ERROR: YOU MUST PRESS ON 'LOAD INPUT'")
        # if np.mean(self.firstframe) > 200:
        #     color = 0
        # else:
        #     color = 255
        #
        # # print("color",color)
        # self.lines.append(f"color: {color} \n")

        # displaying the image
        global calibration_fig
        calibration_fig = np.array(self.firstframe)
        if self.checkvar_TIP.get() == 1:
            ctypes.windll.user32.MessageBoxW(0, "Pick 2 points to calibrate pixel/mm ratio", "Info", 0)
        cv.imshow('Pick points to calibrate', calibration_fig)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv.setMouseCallback('Pick points to calibrate', click_event)

        # create calibration window
        self.calibration_window = tk.Toplevel(self.root)
        self.calibration_window.geometry("350x100")  # set window size
        self.calibration_window.title("Enter Calibration Value")
        tk.Label(self.calibration_window, text="Enter calibration value:").pack()
        self.calibration_entry = tk.Entry(self.calibration_window)
        self.calibration_entry.pack()
        tk.Button(self.calibration_window, text="OK", command=self.calibrate_pressed_ok).pack()
        cv.waitKey(0)
        # self.sct_img = sct_img

        pass

    def calibrate_pressed_ok(self):
        # unit = input("Enter how many units (mm/rad) this distance is equal to (EXAMPLE: 102.3)")
        # unit = round(float(unit), 2)
        unit = self.calibration_entry.get()
        unit = float(unit)

        point1 = calibration_clicks[0]
        point2 = calibration_clicks[1]
        dist = np.linalg.norm(np.array(point1) - np.array(point2))
        midpoint = (int((point1[0] + point2[0]) / 2), int(((point1[1] + point2[1]) / 2)) - 30)
        txt = str(round(dist, 2)) + " pix. = " + str(unit) + " units"
        cv.putText(calibration_fig, txt, midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.line(calibration_fig, point1, point2,
                (0, 0, 255), 2)
        self.CALIBRATED_FLAG = True
        self.pixel_to_unit = unit / dist
        print("pixel_to_unit", self.pixel_to_unit)
        self.lines.append(f"unit {unit} \n")
        self.lines.append(f"dist {dist} \n")
        self.lines.append(f"Pixel to unit (unit/dist) {self.pixel_to_unit} \n")
        cv.imwrite("Calibration.jpg", calibration_fig)
        cv.imshow('Pick points to calibrate', calibration_fig)
        cv.waitKey(0)
        self.calibration_window.destroy()
        self.lines.append("Calibration END --------------------------\n")
        pass

    def execute(self):
        self.lines.append("Analysis Started --------------------------\n")
        # unpack data:
        # --- Input Source:
        self.Input_source = self.Input_source.get()
        self.source_legend = {0: "Monitor Sampling ", 1: "From File "}
        self.lines.append(f"Source: {self.source_legend[self.Input_source]} \n")

        # --- Task:
        self.task = self.task.get()
        self.task_legend = {0: "Digital Extensiometer ", 1: "LoS Stability "}
        self.lines.append(f"Task: {self.task_legend[self.task]} \n")

        # --- Detection Algo:
        self.DetectAlgo = self.DetectAlgo.get()
        Detection_Algo_legend = {0: "Brightest spot ", 1: "Average Intensity ", 2: "Template Matching ",
                                 3: "Contour Center "}
        self.lines.append(f"Detection Algorithm: {Detection_Algo_legend[self.DetectAlgo]} \n")

        # --- Image Params:
        self.brightness = param_dict["brightness"]
        self.contrast = param_dict["contrast"]
        self.thresh_limit = param_dict["thresh_limit"]
        self.Gauss_filter = param_dict["Gauss_filter"]
        self.min_contour = param_dict["min_contour"]
        self.max_contour = param_dict["max_contour"]

        print("Input_source", self.Input_source, self.source_legend[self.Input_source])
        print("task", self.task, self.task_legend[self.task])
        print("detection_algorithm", self.DetectAlgo, Detection_Algo_legend[self.DetectAlgo])
        print("brightness", self.brightness)
        print("contrast", self.contrast)
        print("thresh_limit", self.thresh_limit)
        print("Gauss_filter", self.Gauss_filter)
        print("min_contour", self.min_contour)
        print("max_contour", self.max_contour)

        ###### END UNPACKING DATA -------------------------------

        # Initialize OutputData
        self.cX1 = float("NaN")
        self.cY1 = float("NaN")
        self.cX2 = float("NaN")
        self.cY2 = float("NaN")
        self.distance = float("NaN")
        # ------------------------------------------------------
        self.frame_int_lst = []  # lst with frame numbers
        self.distance_lst = []  # lst with calculated distances (if extensiometer)
        self.cX1_lst = []  # center1 X coords
        self.cX2_lst = []  # center2 X coords (if extensiometer)
        self.cY1_lst = []  # center1 Y coords
        self.cY2_lst = []  # center2 Y coords (if extensiometer)
        ###### END Initializing OUT-DATA -------------------------------

        # create out folder
        self.create_folder()

        # SET VideoRecorder
        self.WriteVideo = False
        if self.checkvar_SaveVideo.get() == 1:
            self.WriteVideo = True
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            (h, w) = self.firstframe.shape[:2]
            output = 'Output.avi'
            self.fps = int(10)
            self.writer = cv.VideoWriter(output, fourcc, 5, (w * 1, h * 2), True)

        # Translating FF to Gray
        print("self.GRAY_TRESH:", self.GRAY_TRESH)
        print("self.HSV_TRESH:", self.HSV_TRESH)
        if self.GRAY_TRESH:
            self.firstframe = cv.cvtColor(np.array(self.firstframe), cv.COLOR_BGR2GRAY)
        if self.HSV_TRESH:
            pass

        # Picking possible Rois (where center could possibly lay) --- Task dependent
        if True:  # LoS -- single point
            if self.checkvar_TIP.get() == 1:
                ctypes.windll.user32.MessageBoxW(0,
                                                 "Drag the mouse to Pick Region of Interest where the FIRST point lays",
                                                 "Info", 0)
            self.firstframe_after_param = self.apply_resulted_brightness_contrast(self.firstframe)

            self.roi_point_1 = select_ROI(self.firstframe_after_param)
            ROIs = self.roi_point_1
            self.cropped_roi_1 = self.firstframe_after_param[ROIs[0][1]:ROIs[0][1] + ROIs[0][3],
                                 ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]

            if self.DetectAlgo == 2:  # Template Matching
                if self.checkvar_TIP.get() == 1:
                    ctypes.windll.user32.MessageBoxW(0,
                                                     "Drag the mouse to Pick Region of Interest with a recognizable template to match (registration)",
                                                     "Info", 0)
                self.template_roi_1 = select_ROI(self.cropped_roi_1)
                ROIs = self.template_roi_1
                self.template_to_match_1 = self.cropped_roi_1[ROIs[0][1]:ROIs[0][1] + ROIs[0][3],
                                           ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]

        if self.task == 0:  # Digital Extensiometer -- 2 points
            if self.checkvar_TIP.get() == 1:
                ctypes.windll.user32.MessageBoxW(0,
                                                 "Drag the mouse to Pick Region of Interest where the SECOND point lays",
                                                 "Info", 0)
            self.roi_point_2 = select_ROI(self.firstframe_after_param)
            ROIs = self.roi_point_2
            self.cropped_roi_2 = self.firstframe_after_param[ROIs[0][1]:ROIs[0][1] + ROIs[0][3],
                                 ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]

            if self.DetectAlgo == 2:  # Template Matching
                if self.checkvar_TIP.get() == 1:
                    ctypes.windll.user32.MessageBoxW(0,
                                                     "Drag the mouse to Pick Region of Interest with a recognizable template to match (registration)",
                                                     "Info", 0)
                self.template_roi_2 = select_ROI(self.cropped_roi_2)
                ROIs = self.template_roi_2
                self.template_to_match_2 = self.cropped_roi_2[ROIs[0][1]:ROIs[0][1] + ROIs[0][3],
                                           ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]

        try:
            cv.imshow("cropped_roi_1", self.cropped_roi_1)
            cv.imwrite('ROI1.jpg', self.cropped_roi_1)
            if self.DetectAlgo == 2:
                cv.imshow("Template_1", self.template_to_match_1)
                cv.imwrite('Template_1.jpg', self.template_to_match_1)
            cv.imshow("cropped_roi_2", self.cropped_roi_2)
            cv.imwrite('ROI2.jpg', self.cropped_roi_2)
            if self.DetectAlgo == 2:
                cv.imshow("Template_2", self.template_to_match_2)
                cv.imwrite('Template_2.jpg', self.template_to_match_2)
        except:
            print("skip")
        cv.waitKey(0)

        # Start Sampling -----------------------
        if self.checkvar_TIP.get() == 1:
            ctypes.windll.user32.MessageBoxW(0,
                                             "Long - press on keyboard 'q' at any time to finish sampling",
                                             "Info", 0)
        self.CONTINUE_SAMPLING = True
        if self.Input_source == 0:  # MonitorSampling
            self.WhileLoopMonitorSampling()
        if self.Input_source == 1:  # Input File
            self.WhileLoopOfInputFile()
        # writing to log
        self.lines.append(f"End Sampling------------- \n ")
        if self.CONTINUE_SAMPLING:  # if user pressed on 'q' reduce 1 from frame num
            self.lines.append(f"Total number of frames: {self.current_frame_num} \n ")
        else:
            self.lines.append(f"Total number of frames: {self.current_frame_num - 1} \n ")

        self.write_log()
        self.export_csv_data()

        # When everything done, release the capture
        cv.destroyAllWindows()
        if self.WriteVideo:
            self.writer.release()

    def record_vid(self, frameA, frameB):
        (h, w) = self.firstframe.shape[:2]
        # print("(h, w)", (h, w))
        # print("self.firstframe.shape", self.firstframe.shape)
        self.RecVidFrameOutput = np.zeros((h * 2, w * 1, 3), dtype="uint8")
        self.RecVidFrameOutput[0:h, 0:w] = cv.cvtColor(frameA, cv.COLOR_GRAY2BGR)
        # self.RecVidFrameOutput[0:h, w:w * 2] = frameB #is alreasy BGR!! cv.cvtColor(frameB, cv.COLOR_GRAY2BGR)
        # self.RecVidFrameOutput[h:h * 2, w:w * 2] =frameB #cv.cvtColor(frameC, cv.COLOR_GRAY2BGR)
        self.RecVidFrameOutput[h:h * 2, 0:w] = frameB  # cv.cvtColor(frameD, cv.COLOR_GRAY2BGR)

        self.writer.write(self.RecVidFrameOutput)
        cv.namedWindow("RecVidFrameOutput", cv.WINDOW_NORMAL)
        cv.imshow("RecVidFrameOutput", self.RecVidFrameOutput)
        # cv.resizeWindow("RecVidFrameOutput", 1080, 720)

    def TemplateMatching(self, frame, original):
        self.error_occured = False
        text = str(self.current_frame_num)
        method = eval('cv.TM_CCOEFF')  # Template matching

        cv.putText(original, "Frame #" + text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.putText(frame, "Frame #" + text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        t = 1
        try:
            # ROI OF POINT 1
            ROIs = self.roi_point_1
            cropped_frame = frame[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
            # SEARCH FOR Template1 in cropped frame
            res1 = cv.matchTemplate(self.template_to_match_1, cropped_frame, method)
            min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)
            # get template roi w,h
            _, _, w, h = [i for i in self.template_roi_1[0]]
            # Find Center of template after matching
            crpd_point = tuple(np.array((max_loc1[0] + w / 2, max_loc1[1] + h / 2), dtype=int))
            # print("crpd_point", crpd_point)
            # Add the coord of ROI POINT 1 in the bigger window
            point1 = tuple(np.array(crpd_point) + np.array((self.roi_point_1[0][0], self.roi_point_1[0][1])))
            # print("point",point)
            # Write to global CX ,CY
            self.cX1 = point1[0]
            self.cY1 = point1[1]

            # Design frames
            cv.circle(frame, point1, 4, (120, 120, 120), 2)
            cv.rectangle(frame, (int(self.cX1 - w / 2), int(self.cY1 - h / 2)),
                         (int(self.cX1 + w / 2), int(self.cY1 + h / 2)), (120, 120, 120), 1)
            circle_text = "center" + str(t) + ":" + str(self.cX1) + "," + str(self.cY1)
            cv.putText(frame, circle_text, (self.cX1 - 20, self.cY1 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (120, 120, 120), 1)

            cv.circle(original, point1, 4, (0, 0, 255), 2)
            cv.rectangle(original, (int(self.cX1 - w / 2), int(self.cY1 - h / 2)),
                         (int(self.cX1 + w / 2), int(self.cY1 + h / 2)), (0, 0, 255), 1)
            cv.putText(original, circle_text, (self.cX1 - 20, self.cY1 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                       1)

            if self.task == 0:  # Digital Extensiometer
                t += 1

                # ROI OF POINT 2
                ROIs = self.roi_point_2
                cropped_frame = frame[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
                # SEARCH FOR Template 2 in cropped frame
                res2 = cv.matchTemplate(self.template_to_match_2, cropped_frame, method)
                min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(res2)
                # get template roi w,h
                _, _, w, h = [i for i in self.template_roi_2[0]]
                # Find Center of template after matching
                crpd_point = tuple(np.array((max_loc2[0] + w / 2, max_loc2[1] + h / 2), dtype=int))
                # print("crpd_point", crpd_point)
                # Add the coord of ROI POINT 2 in the bigger window
                point2 = tuple(np.array(crpd_point) + np.array((self.roi_point_2[0][0], self.roi_point_2[0][1])))
                # print("point",point)

                # Calculate Distance and put cv.Line
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                midpoint = (int((point1[0] + point2[0]) / 2), int(((point1[1] + point2[1]) / 2)))
                linetxt = str(round(dist, 2)) + " pix."

                cv.putText(frame, linetxt, midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                cv.line(frame, point1, point2, (120, 120, 120), 2)

                cv.putText(original, linetxt, midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv.line(original, point1, point2, (0, 0, 255), 2)

                # Write to global CX ,CY
                self.cX2 = point2[0]
                self.cY2 = point2[1]
                self.distance = dist

                # Design frames
                cv.circle(frame, point2, 4, (120, 120, 120), 2)
                cv.rectangle(frame, (int(self.cX2 - w / 2), int(self.cY2 - h / 2)),
                             (int(self.cX2 + w / 2), int(self.cY2 + h / 2)), (120, 120, 120), 1)
                circle_text = "center" + str(t) + ":" + str(self.cX2) + "," + str(self.cY2)
                cv.putText(frame, circle_text, (self.cX2 - 20, self.cY2 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (120, 120, 120), 1)

                cv.circle(original, point2, 4, (0, 0, 255), 2)
                cv.rectangle(original, (int(self.cX2 - w / 2), int(self.cY2 - h / 2)),
                             (int(self.cX2 + w / 2), int(self.cY2 + h / 2)), (0, 0, 255), 1)
                cv.putText(original, circle_text, (self.cX2 - 20, self.cY2 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 255), 1)


        except Exception as ex:
            print(ex)
            print("Error occured - skipping detection")
            self.error_occured = True
            self.cX1 = float("NaN")
            self.cY1 = float("NaN")
            self.cX2 = float("NaN")
            self.cY2 = float("NaN")
            self.distance = float("NaN")

        return frame, original

    def FindContourCenter(self, frame, original):
        self.error_occured = False
        text = str(self.current_frame_num)
        try:
            cnts = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        except:
            ctypes.windll.user32.MessageBoxW(0,
                                             "No contours found!! make sure you chose parameters correctly",
                                             "Info", 0)

        cnts = imutils.grab_contours(cnts)
        centers = []
        locations = []
        t = 0  # initializing number of centers
        # initializing contour center to None
        cX = None
        cY = None

        # Set use defined ranges for point1 and point2 (if extensiometer)
        range_x1 = range(self.roi_point_1[0][0], self.roi_point_1[0][0] + self.roi_point_1[0][2])
        range_y1 = range(self.roi_point_1[0][1], self.roi_point_1[0][1] + self.roi_point_1[0][3])

        if self.task == 0:  # Digital Extensiometer
            range_x2 = range(self.roi_point_2[0][0], self.roi_point_2[0][0] + self.roi_point_2[0][2])
            range_y2 = range(self.roi_point_2[0][1], self.roi_point_2[0][1] + self.roi_point_2[0][3])
        else:
            range_x2 = range_x1
            range_y2 = range_y1
        # print(f"range x1:{range_x1}\n range x2:{range_x2}\n range y1:{range_y1}\n range y2:{range_y2}\n")
        # ---------------------- END SET UP
        try:
            for c in cnts:
                # What is an acceptable contour? :
                if cv.contourArea(c) < self.max_contour and cv.contourArea(c) > self.min_contour:
                    # print("Contour Area", cv.contourArea(c))
                    # compute the center of the contour
                    M = cv.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # draw the contour and center of the shape on the image
                    (x, y, w, h) = cv.boundingRect(c)
                    # print("x,y,w,h",x,y,w,h)
                    # cX=int(x+w/2)
                    # cY=int(y+h/2)
                    # print("cX", cX, "cY", cY)
                    cond_x_1 = cX in range_x1
                    cond_x_2 = cX in range_x2
                    cond_x = cond_x_1 or cond_x_2
                    cond_y_1 = cY in range_y1
                    cond_y_2 = cY in range_y2
                    cond_y = cond_y_1 or cond_y_2
                    if cond_x and cond_y:
                        t = t + 1  # add 1 to center counter, acceptable contour found
                        # Draws On original frame (feed)
                        cv.drawContours(original, [c], 0, (0, 0, 255), 1)
                        # cv.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv.circle(original, (cX, cY), 4, (0, 0, 255), -1)
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 0), 1)
                        cv.circle(frame, (cX, cY), 4, (120, 120, 120), -1)
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]
                        self.cX1 = cX
                        self.cY1 = cY
                        circle_text = "center" + str(t) + ":" + str(round(cX, 1)) + "," + str(round(cY, 1))
                        cv.putText(frame, circle_text, (x - 20, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120),
                                   1)
                        if self.task == 1:  # LoS
                            centers.append((cX, cY))
                        if self.task == 0:  # Digital Extensiometer
                            locations.append((cX, cY))

                        cv.putText(original, "Frame #" + text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv.putText(frame, "Frame #" + text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

                        if self.task == 0:  # Digital Extensiometer
                            # print("locations", locations)
                            for i in range(1, len(locations)):
                                point1 = (int(locations[i][0]), int(locations[i][1]))
                                point2 = (int(locations[i - 1][0]), int(locations[i - 1][1]))
                                # dx.append(point2[0] - point1[0])
                                # dy.append(point2[1] - point1[1])
                                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                                # distance_lst.append(dist)
                                print("dist", dist)
                                midpoint = (int((point1[0] + point2[0]) / 2), int(((point1[1] + point2[1]) / 2)))
                                txt = str(round(dist, 2)) + " pix."
                                cv.putText(frame, txt, midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                                cv.line(frame, point1, point2, (120, 120, 120), 2)
                                cv.putText(original, txt, midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                cv.line(original, point1, point2, (0, 0, 255), 2)

                                self.cX1 = point1[0]
                                self.cY1 = point1[1]
                                self.cX2 = point2[0]
                                self.cY2 = point2[1]
                                self.distance = dist

        except Exception as ex:
            print(ex)
            self.error_occured = True
            self.cX1 = float("NaN")
            self.cY1 = float("NaN")
            self.cX2 = float("NaN")
            self.cY2 = float("NaN")
            self.distance = float("NaN")
            pass

        # When is error occuring? ----------------------------
        # 1. no contours conditions met: t=0
        # 2. found too many contours on LoS: self.task == 1 AND t!=1
        # 3. found too many contours on Extensiometer self.task == 0 AND t!=2
        bool1 = t == 0
        bool2 = t != 1 and self.task == 1
        bool3 = t != 2 and self.task == 0
        if bool1 or bool2 or bool3:
            if bool2:
                msg = f"Error occured - skipping detection\n bool2= {bool2} means LoS task and num of contours found is not equal to 1\n"
            if bool3:
                msg = f"Error occured - skipping detection\n bool3 = {bool3} means Extensiometer task and num of contours found is not equal to 2\n"
            if bool1:
                msg = f"Error occured - skipping detection\n bool1 = {bool1} means no contours conditions met - num of contours found is equal to 0\n"
            msg += f"min area:{self.min_contour}\n max area:{self.max_contour}\n"
            msg += f"range x1:{range_x1}\n range x2:{range_x2}\n range y1:{range_y1}\n range y2:{range_y2}\n"
            msg += f"\n\n cX = {cX} cY = {cY} ContourArea ={cv.contourArea(c)}"
            if self.checkvar_error_occured.get() == 1:
                ctypes.windll.user32.MessageBoxW(0, msg, "Info", 0)
            print(msg)
            self.error_occured = True
            self.cX1 = float("NaN")
            self.cY1 = float("NaN")
            self.cX2 = float("NaN")
            self.cY2 = float("NaN")
            self.distance = float("NaN")

        return frame, original

    def WhileLoopMonitorSampling(self):
        self.current_frame_num = 1
        ROIs = self.bounding_box
        grab_region = {'top': int(ROIs[0][1]), 'left': int(ROIs[0][0]), 'width': int(ROIs[0][2]),
                       'height': int(ROIs[0][3])}

        # print("self.CONTINUE_SAMPLING1", self.CONTINUE_SAMPLING)
        while self.CONTINUE_SAMPLING:
            # print("self.CONTINUE_SAMPLING",self.CONTINUE_SAMPLING)
            if (cv.waitKey(20) & 0xFF) == ord('q'):
                self.CONTINUE_SAMPLING = False
                if self.WriteVideo:
                    self.writer.release()
                cv.destroyAllWindows()
                print('\nDone!\n')
                break

            cropped_frame = np.asarray(self.sct.grab(grab_region))
            cropped_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGRA2BGR)

            if self.GRAY_TRESH:
                cropped_gray = cv.cvtColor(np.array(cropped_frame), cv.COLOR_BGR2GRAY)
                cropped_gray_after_param = self.apply_resulted_brightness_contrast(cropped_gray)
            if self.HSV_TRESH:
                cropped_gray_after_param = self.apply_resulted_brightness_contrast(cropped_frame)

            # Find Point/s Based on Algo:
            if self.DetectAlgo == 2:  # Template Matching
                self.result_frame, original = self.TemplateMatching(cropped_gray_after_param, cropped_frame)

            if self.DetectAlgo == 3:  # Contour Center
                self.result_frame, original = self.FindContourCenter(cropped_gray_after_param, cropped_frame)

            if self.WriteVideo:
                self.record_vid(self.result_frame, original)

            cv.imshow('Screen', self.result_frame)
            cv.waitKey(100)

            if not self.error_occured:
                self.frame_int_lst.append(self.current_frame_num)
                self.cX1_lst.append(self.cX1)
                self.cY1_lst.append(self.cY1)
                self.cX2_lst.append(self.cX2)
                self.cY2_lst.append(self.cY2)
                self.distance_lst.append(self.distance)
                self.error_occured = False

                # live plotting:
                # self.live_plotter(self.cX1_lst,self.cY1_lst)

            # Add user prompt wanted grabbing DELAY:
            if float(self.screen_sample_entry.get()) > 0:
                for remaining in range(int(self.screen_sample_entry.get()), 0, -1):
                    sys.stdout.write("\r")
                    sys.stdout.write("{:2d} seconds remaining.".format(remaining))
                    sys.stdout.flush()
                    wk = cv.waitKey(1000)
                    if (wk & 0xFF) == ord('q'):
                        self.CONTINUE_SAMPLING = False
                        if self.WriteVideo:
                            self.writer.release()
                        cv.destroyAllWindows()
                        print('\nDone!\n')
                        break
                sys.stdout.write("\rGrabbing a new frame.....            \n")

                # print(f"Sleeping for: {self.screen_sample_entry.get()} seconds")
                # time.sleep(float(self.screen_sample_entry.get()))
            print(f"Frame: {self.current_frame_num} ----------------------")
            self.current_frame_num += 1

            # continue

    def WhileLoopOfInputFile(self):
        self.current_frame_num = self.StartFrame
        ret, frame = self.cap.read()
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.StartFrame)

        while (ret) and self.CONTINUE_SAMPLING:
            ret, frame = self.cap.read()
            # Stop conditions
            if (cv.waitKey(20) & 0xFF) == ord('q'):
                self.CONTINUE_SAMPLING = False
                if self.WriteVideo:
                    self.writer.release()
                cv.destroyAllWindows()
                print('\nDone!\n')
                break
            if not ret:
                self.cap.release()
                if self.WriteVideo:
                    self.writer.release()
                cv.destroyAllWindows()
                print('\nDone!\n')
                break
            if self.current_frame_num == self.EndFrame:
                if self.WriteVideo:
                    self.writer.release()
                cv.destroyAllWindows()
                print('\nDone!\n')
                break

            ROIs = self.bounding_box
            cropped_frame = frame[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
            if self.GRAY_TRESH:
                cropped_gray = cv.cvtColor(np.array(cropped_frame), cv.COLOR_BGR2GRAY)
                cropped_gray_after_param = self.apply_resulted_brightness_contrast(cropped_gray)
            if self.HSV_TRESH:
                cropped_gray_after_param = self.apply_resulted_brightness_contrast(cropped_frame)

            # Find Point/s Based on Algo:
            if self.DetectAlgo == 2:  # Template Matching
                self.result_frame, original = self.TemplateMatching(cropped_gray_after_param, cropped_frame)

            if self.DetectAlgo == 3:  # Contour Center
                self.result_frame, original = self.FindContourCenter(cropped_gray_after_param, cropped_frame)

            if self.WriteVideo:
                self.record_vid(self.result_frame, original)

            cv.imshow('Screen', self.result_frame)
            cv.waitKey(10)

            if not self.error_occured:
                self.frame_int_lst.append(self.current_frame_num)
                self.cX1_lst.append(self.cX1)
                self.cY1_lst.append(self.cY1)
                self.cX2_lst.append(self.cX2)
                self.cY2_lst.append(self.cY2)
                self.distance_lst.append(self.distance)
                self.error_occured = False
            print(f"Frame: {self.current_frame_num}/{self.EndFrame}")
            self.current_frame_num += 1

            # continue

    def exit(self):
        self.root.destroy()

    def POST_PROC(self):
        root_POST = tk.Tk()
        POST_app = POST_App(root_POST)
        root_POST.mainloop()

    def write_log(self):
        path = os.path.join(os.getcwd(), "log.txt")
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines))
        os.startfile(path)

    def post_proccesing_oldddd(self, dx, dy, time_lst, distance_lst):
        # moving to zero
        dx = [dx[i] - dx[0] for i in range(0, len(dx))]
        self.lines.append(" --------------- dx stats ------------------- \n ")
        sr_DX, dt, mean, sd, rms, skew, kurtosis, dur = signal_stats(time_lst, dx, len(time_lst))

        dy = [dy[i] - dy[0] for i in range(0, len(dy))]
        self.lines.append(" --------------- dy stats ------------------- \n ")
        sr_DX, dt, mean, sd, rms, skew, kurtosis, dur = signal_stats(time_lst, dy, len(time_lst))

        distance_lst = [distance_lst[i] - distance_lst[0] for i in range(0, len(distance_lst))]
        self.lines.append(" --------------- dist stats ------------------- \n ")
        sr_DX, dt, mean, sd, rms, skew, kurtosis, dur = signal_stats(time_lst, distance_lst, len(time_lst))

        # filtered Data
        sos = butter(2, 0.01, output='sos')
        filt_dx = sosfiltfilt(sos, dx)
        self.lines.append(" --------------- filtered dx stats ------------------- \n ")
        sr_DX, dt, mean, sd, rms, skew, kurtosis, dur = signal_stats(time_lst, filt_dx, len(time_lst))

        filt_dy = sosfiltfilt(sos, dy)
        self.lines.append(" --------------- filtered dy stats ------------------- \n ")
        sr_DX, dt, mean, sd, rms, skew, kurtosis, dur = signal_stats(time_lst, filt_dy, len(time_lst))

        filt_dist = sosfiltfilt(sos, distance_lst)
        self.lines.append(" --------------- filtered dist stats ------------------- \n ")
        sr_DX, dt, mean, sd, rms, skew, kurtosis, dur = signal_stats(time_lst, filt_dist, len(time_lst))

        data = {"time": time_lst, "dx_pix": dx, "dy_pix": dy, "dist_pix": distance_lst,
                "dx_pix_filtered": filt_dx, "dy_pix_filtered": filt_dy, "dist_pix_filtered": filt_dist,
                "pixel_to_unit": self.pixel_to_unit}
        # data["dx_units"] = data["dx_pix"] * self.pixel_to_unit
        # data["dy_units"] = data["dy_pix"] * self.pixel_to_unit
        # data["dist_units"] = data["dist_pix"] * self.pixel_to_unit
        global data_df
        data_df = pd.DataFrame(data)
        data_df.to_csv("data.csv")

        ################################################################################
        # initialize plot
        global ax1, ax2, ax3
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5))
        fig.suptitle(f'Digital Extensiometer')
        # fig.suptitle(f'Boresight retention ; 1 pixel = {round(self.pixel_to_unit, 3)} units ')

        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_ylabel('Distance', fontsize=18)
        ax1.grid(linestyle='--', linewidth=1, which='both')
        ax1.plot(time_lst, distance_lst, label="Boresight [pix]", linestyle="--", color="k",
                 marker=".")  # , linestyle="--",color="k"
        ax1.plot(time_lst, filt_dist, color="r", label='filtered signal')
        ax1.legend()

        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.set_ylabel('Distance', fontsize=18)
        ax2.grid(linestyle='--', linewidth=1, which='both')
        ax2.plot(time_lst, dx, label="dx [pix]", linestyle="--", color="k", marker=".")  # ,
        ax2.plot(time_lst, filt_dx, color="r", label='filtered signal - dx')
        ax2.legend()

        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.set_xlabel('Time (s)', fontsize=18)
        ax3.set_ylabel('Distance', fontsize=18)
        ax3.grid(linestyle='--', linewidth=1, which='both')
        ax3.plot(time_lst, dy, label="dy", linestyle="--", color="k", marker=".")  # , linestyle="--",
        ax3.plot(time_lst, filt_dy, color="r", label='filtered signal - dy')
        ax3.legend()

        fig.tight_layout()

        range_frames_dy = RectangleSelector(ax3, line_select_callback_dy,
                                            drawtype='box', useblit=False, button=[1],
                                            minspanx=5, minspany=5, spancoords='pixels',
                                            interactive=True)

        range_frames_dx = RectangleSelector(ax2, line_select_callback_dx,
                                            drawtype='box', useblit=False, button=[1],
                                            minspanx=5, minspany=5, spancoords='pixels',
                                            interactive=True)

        path = os.path.join(os.getcwd(), "results.txt")
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines))
        os.startfile(path)
        plt.show()

    def export_csv_data(self):
        df_title = os.path.join(os.getcwd(), "Results.csv")
        DataStructure = {"Frame": self.frame_int_lst,
                         "Center1_X": self.cX1_lst,
                         "Center1_Y": self.cY1_lst,
                         "Center2_X": self.cX2_lst,
                         "Center2_Y": self.cY2_lst,
                         "Distance": self.distance_lst
                         }

        Results_DF = pd.DataFrame(DataStructure)
        Results_DF.to_csv(df_title)


############################################################################################
#########################################################################################
class POST_App:

    def __init__(self, root):
        self.error_occured = False

        # intitialize logfile
        self.lines = []
        self.root = root
        self.root.title("Post Processing | by Yarden Zaki")
        self.root.geometry("400x400")  # set window size to 500x500

        # create menu label for radio buttons
        self.menu_label = tk.Label(root, text="*** Input Results.csv ***", font=(14), anchor="w")
        self.menu_label.pack(pady=5)

        # # create radio buttons
        # self.Input_source = tk.IntVar()
        # self.monitor_rb = tk.Radiobutton(root, text="Monitor Sampling", font=(14), variable=self.Input_source, value=0)
        # self.file_rb = tk.Radiobutton(root, text="File", font=(14), variable=self.Input_source, value=1,
        #                               command=self.load_file)
        # self.monitor_rb.pack(pady=5, anchor="w")
        # self.file_rb.pack(pady=5, anchor="w")
        # ##############---------------------------------------------------
        # # create menu label for radio buttons
        # self.menu_label = tk.Label(root, text="*** Task ***", font=(14), anchor="w")
        # self.menu_label.pack(pady=5)
        #
        # # create radio buttons
        # self.task = tk.IntVar()
        # self.extensiometer_rb = tk.Radiobutton(root, text="Digital Extensiometer", font=(14), variable=self.task,
        #                                        value=0, command=self.change_rb_state)
        # self.los_rb = tk.Radiobutton(root, text="LoS Stability", font=(14), variable=self.task, value=1,
        #                              command=self.change_rb_state)
        # self.extensiometer_rb.pack(pady=5, anchor="w")
        # self.los_rb.pack(pady=5, anchor="w")
        # #############---------------------------------------------
        #
        # # create menu label for radio buttons
        # self.menu_label = tk.Label(root, text="*** Detection Algorithm ***", font=(14), anchor="w")
        # self.menu_label.pack(pady=5)
        #
        # # create radio buttons
        # self.DetectAlgo = tk.IntVar()
        # self.radio_button1 = tk.Radiobutton(root, text="Brightest spot", font=(14), variable=self.DetectAlgo, value=0,
        #                                     command=self.hide_entry)
        # self.radio_button2 = tk.Radiobutton(root, text="Average Intensity", font=(14), variable=self.DetectAlgo,
        #                                     value=1, command=self.show_entry)
        # self.radio_button3 = tk.Radiobutton(root, text="Template Matching", font=(14), variable=self.DetectAlgo,
        #                                     value=2, command=self.show_entry)
        # self.radio_button4 = tk.Radiobutton(root, text="Contour Center", font=(14), variable=self.DetectAlgo, value=3,
        #                                     command=self.hide_entry)
        # self.radio_button1.pack(pady=5, anchor="w")
        # self.radio_button2.pack(pady=5, anchor="w")
        # self.radio_button3.pack(pady=5, anchor="w")
        # self.radio_button4.pack(pady=5, anchor="w")
        #
        # # DISABLE FOR NOW ----
        # self.radio_button1.config(state="disabled")
        # self.radio_button2.config(state="disabled")
        # self.radio_button4.select()
        # # ------------------------
        #
        # # create menu label for radio buttons
        # self.options_label = tk.Label(root, text="*** Options ***", font=(16), anchor="w")
        # self.options_label.pack(pady=10)
        #
        # # Show Tips:
        # self.checkvar_TIP = tk.IntVar()
        # self.check_button_TIP = tk.Checkbutton(root, text="Show Instructions Pop-Ups", font=(11),
        #                                        variable=self.checkvar_TIP)
        # self.check_button_TIP.pack(pady=5, anchor="w")
        # # create inverting label:
        # self.checkvar_SaveVideo = tk.IntVar()
        # self.check_button = tk.Checkbutton(root, text="Save Analyzed Video", font=(11),
        #                                    variable=self.checkvar_SaveVideo)
        # self.check_button.pack(pady=5, anchor="w")
        # # create Pixels Histogram:
        # self.checkvar_pix_hist = tk.IntVar()
        # self.check_button_pix_hist = tk.Checkbutton(root, text="TBD2", font=(7), variable=self.checkvar_pix_hist)
        # self.check_button_pix_hist.pack(pady=5, anchor="w")
        #
        # # create entry label and widget
        # self.entry_label = tk.Label(root, text="Avg. every pixel\n higher than:", font=('Arial', 10), anchor="w")
        # self.entry_label.pack(pady=5)
        # self.entry = tk.Entry(root, width=5)
        # self.entry.pack(pady=5)
        # self.entry.pack_forget()  # hide entry widget initially
        # self.entry_label.pack_forget()  # hide entry label initially
        # self.entry.insert(0, 120)  # set default value for entry widget
        #
        # create buttons
        self.load_button = tk.Button(root, text="Load\n Input", font=('Arial', 14), command=self.load_XLSX_file)
        # self.calibration_button = tk.Button(root, text="Calibrate\n", font=('Arial', 14), command=self.calibrate)
        # self.set_params_button = tk.Button(root, text="Set\n Parameters", font=('Arial', 14), command=self.set_params)
        self.analyze_button = tk.Button(root, text="Analyze\n", font=('Arial', 14), command=self.Analyze)
        #
        self.exit_button = tk.Button(root, text="Exit\n", font=('Arial', 14), command=self.exit)
        # self.work_button = tk.Button(root, text="Work\n Dir", font=('Arial', 14), command=self.open_cwd)
        self.load_button.pack(side="left", padx=5, pady=20, anchor="w")
        # self.calibration_button.pack(side="left", padx=5, pady=20, anchor="w")
        # self.set_params_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.analyze_button.pack(side="left", padx=5, pady=20, anchor="w")
        # self.work_button.pack(side="left", padx=5, pady=20, anchor="w")
        self.exit_button.pack(side="left", padx=5, pady=20, anchor="w")

    def exit(self):
        self.root.destroy()

    def create_folder(self):
        image_folder_name = '\Analyzed_' + str(self.POSTPROCTASK) + "_"
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_folder_name = image_folder_name + time_stamp
        Newdir_path = self.folder_path + image_folder_name

        try:
            os.mkdir(Newdir_path)
            # if self.CALIBRATED_FLAG:
            #     os.rename(path + "/Calibration.jpg", Newdir_path + "/Calibration.jpg")
            # if self.GRAY_TRESH:
            #     os.rename(path + "/GRAY_TRESH_image_params.jpg", Newdir_path + "/GRAY_TRESH_image_params.jpg")
            #     print("moved GRAY_TRESH_image_params.jpg")
            # if self.HSV_TRESH:
            #     os.rename(path + "/HSV_TRESH_image_params.jpg", Newdir_path + "/HSV_TRESH_image_params.jpg")
            #     print("moved HSV_TRESH_image_params.jpg")

        except OSError:
            print("Creation of the directory %s failed" % Newdir_path)

        self.analyzed_folder_path = Newdir_path
        os.chdir(Newdir_path)
        print("current dir.", os.getcwd())

    def load_XLSX_file(self):
        self.POSTPROCTASK = None
        print("LOAD FILE")
        self.file_path = filedialog.askopenfilename()
        self.folder_path = os.path.dirname(self.file_path)
        print("file_path", self.file_path)
        print("folder_path", self.folder_path)
        if "LoS" in self.folder_path.split("/")[-1]:
            self.POSTPROCTASK = "LoS"
        if "Extensiometer" in self.folder_path.split("/")[-1]:
            self.POSTPROCTASK = "Extensiometer"
        print(f"TASK: {self.POSTPROCTASK}")

        # Creating Output folder for analysis  results
        self.create_folder()

        # Loading CSV to DF
        self.centers_df = pd.read_csv(self.file_path)
        print(self.centers_df.head(20))

    def update_extensiometer_fig(self, val):
        cutoff_freq = self.slider_CUTOFF_FREQ.val

        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'low')
            # detrend_Center_0_X = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_X"])
            self.DX_detrened = signal.filtfilt(b, a, self.DX)
            self.DY_detrened = signal.filtfilt(b, a, self.DY)
            self.Distance_detrened = signal.filtfilt(b, a, self.Distance)

        else:
            self.DX_detrened = self.DX
            self.DY_detrened = self.DY
            self.Distance_detrened = self.Distance

        for i in self.axs[0].lines:
            i.remove()
        for i in self.axs[1].lines:
            i.remove()
        for i in self.axs[2].lines:
            i.remove()

        self.axs[0].plot(self.FRAME, self.DX_detrened, label="DX", color="k", linestyle="--", marker=".")
        self.axs[0].set_ylabel('DX [pix]', fontsize=18)
        self.axs[0].tick_params(axis='both', which='major', labelsize=10)
        self.axs[0].grid(linestyle='--', linewidth=1, which='both')

        self.axs[1].plot(self.FRAME, self.DY_detrened, label="DY", color="k", linestyle="--", marker=".")
        self.axs[1].set_ylabel('DY [pix]', fontsize=18)
        self.axs[1].tick_params(axis='both', which='major', labelsize=10)
        self.axs[1].grid(linestyle='--', linewidth=1, which='both')

        self.axs[2].plot(self.FRAME, self.Distance_detrened, label="DY", color="k", linestyle="--", marker=".")
        self.axs[2].set_xlabel('Frame #', fontsize=18)
        self.axs[2].set_ylabel('Distance [pix]', fontsize=18)
        self.axs[2].tick_params(axis='both', which='major', labelsize=10)
        self.axs[2].grid(linestyle='--', linewidth=1, which='both')

        # use the values to set  ylim
        # get min and max y
        ylim_min = self.DX_detrened.min()
        ylim_max = self.DX_detrened.max()
        self.axs[0].set_ylim(ylim_min, ylim_max)

        ylim_min = self.DY_detrened.min()
        ylim_max = self.DY_detrened.max()
        self.axs[1].set_ylim(ylim_min, ylim_max)

        ylim_min = self.Distance_detrened.min()
        ylim_max = self.Distance_detrened.max()
        self.axs[2].set_ylim(ylim_min, ylim_max)

        self.fig.canvas.draw()

    def extensiometer_rect_callback_x(self, eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        from_frame = round(max(x1, self.FRAME.min())) - 1
        to_frame = round(min(x2, self.FRAME.max())) + 1

        self.range_x[:] = [from_frame, to_frame]
        print(f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f}")

        self.Center1_X_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Center1_X"]

        self.Center2_X_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Center2_X"]

        self.DX_cropped = self.Center1_X_cropped - self.Center2_X_cropped
        # Calc Extension: DL-L0
        L0 = self.centers_df["Center1_X"][0] - self.centers_df["Center2_X"][0]
        print("L0", L0)
        self.DX_cropped = self.DX_cropped - L0

        cutoff_freq = self.slider_CUTOFF_FREQ.val
        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'low')
            # detrend_Center_0_X = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_X"])
            self.DX_cropped_detrened = signal.filtfilt(b, a, self.DX_cropped)
        else:
            self.DX_cropped_detrened = self.DX_cropped
        print("DX_cropped_detrened", self.DX_cropped_detrened)
        MIN = round(self.DX_cropped_detrened.min(), 2)
        MAX = round(self.DX_cropped_detrened.max(), 2)

        # if USE_ANGULAR:
        #     stats_str = f" X movements , RMS of selected range :  {RMS} pix. ; {round(RMS * IFOV, 2)} urad"
        # else:
        #     stats_str = f" X movements , RMS of selected range:  {RMS} pix."

        stats_str = f" DX ; Selected Range: {from_frame:3.0f} --> {to_frame:3.0f} ({len(self.DX_cropped_detrened)} points) ;  (Min,Max) of selected range:  ({MIN}, {MAX}) pix."
        self.axs[0].set_title(stats_str)

        # fig_title = center_title + "_Selected_ranges.png"
        # plt.savefig(fig_title)

    def extensiometer_rect_callback_y(self, eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        from_frame = round(max(x1, self.FRAME.min())) - 1
        to_frame = round(min(x2, self.FRAME.max())) + 1

        self.range_x[:] = [from_frame, to_frame]
        print(f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f}")

        self.Center1_Y_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Center1_Y"]

        self.Center2_Y_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Center2_Y"]

        self.DY_cropped = self.Center1_Y_cropped - self.Center2_Y_cropped
        # Calc Extension: DL-L0
        L0 = self.centers_df["Center1_Y"][0] - self.centers_df["Center2_Y"][0]
        print("L0", L0)
        self.DY_cropped = self.DY_cropped - L0

        cutoff_freq = self.slider_CUTOFF_FREQ.val
        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'low')
            # detrend_Center_0_Y = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_Y"])
            self.DY_cropped_detrened = signal.filtfilt(b, a, self.DY_cropped)
        else:
            self.DY_cropped_detrened = self.DY_cropped
        print("DY_cropped_detrened", self.DY_cropped_detrened)
        MIN = round(self.DY_cropped_detrened.min(), 2)
        MAX = round(self.DY_cropped_detrened.max(), 2)

        # if USE_ANGULAR:
        #     stats_str = f" X movements , RMS of selected range :  {RMS} pix. ; {round(RMS * IFOV, 2)} urad"
        # else:
        #     stats_str = f" X movements , RMS of selected range:  {RMS} pix."

        stats_str = f" DY ; Selected Range: {from_frame:3.0f} --> {to_frame:3.0f} ({len(self.DY_cropped_detrened)} points) ;  (Min,Max) of selected range:  ({MIN}, {MAX}) pix."
        self.axs[1].set_title(stats_str)

        # fig_title = center_title + "_Selected_ranges.png"
        # plt.savefig(fig_title)

    def extensiometer_rect_callback_z(self, eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        from_frame = round(max(x1, self.FRAME.min())) - 1
        to_frame = round(min(x2, self.FRAME.max())) + 1

        self.range_x[:] = [from_frame, to_frame]
        print(f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f}")

        self.Distance_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Distance"]
        # Calc Extension: DL-L0
        L0 = self.centers_df["Distance"][0]
        print("L0", L0)
        self.Distance_cropped = self.Distance_cropped - L0

        cutoff_freq = self.slider_CUTOFF_FREQ.val
        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'low')
            # detrend_Center_0_X = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_X"])
            self.Distance_cropped_detrened = signal.filtfilt(b, a, self.Distance_cropped)
        else:
            self.Distance_cropped_detrened = self.Distance_cropped
        print("Distance_cropped_detrened", self.Distance_cropped_detrened)
        MIN = round(self.Distance_cropped_detrened.min(), 2)
        MAX = round(self.Distance_cropped_detrened.max(), 2)

        # if USE_ANGULAR:
        #     stats_str = f" X movements , RMS of selected range :  {RMS} pix. ; {round(RMS * IFOV, 2)} urad"
        # else:
        #     stats_str = f" X movements , RMS of selected range:  {RMS} pix."

        stats_str = f" Distance ; Selected Range: {from_frame:3.0f} --> {to_frame:3.0f} ({len(self.Distance_cropped_detrened)} points) ;  (Min,Max) of selected range:  ({MIN}, {MAX}) pix."
        self.axs[2].set_title(stats_str)

        # fig_title = center_title + "_Selected_ranges.png"
        # plt.savefig(fig_title)

    def update_los_fig(self, val):
        cutoff_freq = self.slider_CUTOFF_FREQ.val

        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'high')
            # detrend_Center_0_X = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_X"])
            self.Center1_X_detrened = signal.filtfilt(b, a, self.Center1_X)
            self.Center1_Y_detrened = signal.filtfilt(b, a, self.Center1_Y)
        else:
            self.Center1_X_detrened = self.Center1_X
            self.Center1_Y_detrened = self.Center1_Y

        for i in self.axs[0].lines:
            i.remove()
        for i in self.axs[1].lines:
            i.remove()

        self.axs[0].plot(self.FRAME, self.Center1_X_detrened, label="Center - X", color="k", linestyle="--", marker=".")
        self.axs[1].plot(self.FRAME, self.Center1_Y_detrened, label="Center - Y", color="k", linestyle="--", marker=".")
        self.axs[0].set_ylabel('Center X coord.', fontsize=18)
        self.axs[0].grid(linestyle='--', linewidth=1, which='both')
        self.axs[1].set_xlabel('Frame #', fontsize=18)
        self.axs[1].set_ylabel('Center Y coord.', fontsize=18)
        self.axs[1].grid(linestyle='--', linewidth=1, which='both')
        # self.axs[0].legend()
        # self.axs[1].legend()

        # use the values to set  ylim
        # get min and max y
        ylim_min = self.Center1_X_detrened.min()
        ylim_max = self.Center1_X_detrened.max()
        self.axs[0].set_ylim(ylim_min, ylim_max)

        ylim_min = self.Center1_Y_detrened.min()
        ylim_max = self.Center1_Y_detrened.max()
        self.axs[1].set_ylim(ylim_min, ylim_max)

        self.fig.canvas.draw()

    def los_rect_callback_x(self, eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        from_frame = round(max(x1, self.FRAME.min())) - 1
        to_frame = round(min(x2, self.FRAME.max())) + 1

        self.range_x[:] = [from_frame, to_frame]
        print(f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f}")

        self.Center1_X_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Center1_X"]

        cutoff_freq = self.slider_CUTOFF_FREQ.val
        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'high')
            # detrend_Center_0_X = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_X"])
            try:
                self.Center1_X_cropped_detrened = signal.filtfilt(b, a, self.Center1_X_cropped)
            except:
                self.Center1_X_cropped_detrened = np.array([0])
                ctypes.windll.user32.MessageBoxW(0,
                                                 "Select a minimum of 12 points",
                                                 "Info", 0)
        else:
            self.Center1_X_cropped_detrened = self.Center1_X_cropped
        print("self.Center1_X_cropped_detrened", self.Center1_X_cropped_detrened)
        RMS = round(self.Center1_X_cropped_detrened.std(), 2)
        MEAN = round(self.Center1_X_cropped_detrened.mean(), 2)

        # if USE_ANGULAR:
        #     stats_str = f" X movements , RMS of selected range :  {RMS} pix. ; {round(RMS * IFOV, 2)} urad"
        # else:
        #     stats_str = f" X movements , RMS of selected range:  {RMS} pix."

        stats_str = f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f} ({len(self.Center1_X_cropped_detrened)} points) ;  (MEAN,RMS) of selected range:  ({MEAN}, {RMS}) pix."
        self.axs[0].set_title(stats_str)

        # fig_title = center_title + "_Selected_ranges.png"
        # plt.savefig(fig_title)

    def los_rect_callback_y(self, eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        from_frame = round(max(x1, self.FRAME.min())) - 1
        to_frame = round(min(x2, self.FRAME.max())) + 1

        self.range_y[:] = [from_frame, to_frame]
        print(f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f}")

        self.Center1_Y_cropped = \
            self.centers_df.loc[(self.centers_df['Frame'] > from_frame) & (self.centers_df['Frame'] < to_frame)][
                "Center1_Y"]

        cutoff_freq = self.slider_CUTOFF_FREQ.val
        if cutoff_freq > 0 and cutoff_freq < 1:
            b, a = signal.butter(3, cutoff_freq, 'high')
            # detrend_Center_0_X = signal.filtfilt(b, a, centers_df.loc[1:, "Center_0_X"])
            try:
                self.Center1_Y_cropped_detrened = signal.filtfilt(b, a, self.Center1_Y_cropped)

            except:
                self.Center1_Y_cropped_detrened = np.array([0])
                ctypes.windll.user32.MessageBoxW(0,
                                                 "Select a minimum of 12 points",
                                                 "Info", 0)

        else:
            self.Center1_Y_cropped_detrened = self.Center1_Y_cropped
        print("self.Center1_Y_cropped_detrened", self.Center1_Y_cropped_detrened)
        RMS = round(self.Center1_Y_cropped_detrened.std(), 2)
        MEAN = round(self.Center1_Y_cropped_detrened.mean(), 2)

        # if USE_ANGULAR:
        #     stats_str = f" Y movements , RMS of selected range :  {RMS} pix. ; {round(RMS * IFOV, 2)} urad"
        # else:
        #     stats_str = f" Y movements , RMS of selected range:  {RMS} pix."

        stats_str = f"Selected Range: {from_frame:3.0f} --> {to_frame:3.0f} ({len(self.Center1_Y_cropped_detrened)} points) ;  (MEAN,RMS) of selected range:  ({MEAN}, {RMS}) pix."
        self.axs[1].set_title(stats_str)

        # fig_title = center_title + "_Selected_ranges.png"
        # plt.savefig(fig_title)

    def initial_los_plot(self):
        # Plotting + Rangeselection
        self.fig, self.axs = plt.subplots(2, 1)
        title = " Center (X,Y) movements [pixels]"
        plt.suptitle(title)

        self.axs[0].plot(self.FRAME, self.Center1_X, label="Center - X", color="k", linestyle="--", marker=".")
        # self.axs[0].plot(self.Frame, detrend_Center_0_X, label="HPF - " + str(HPF_FREQ))
        self.axs[0].set_ylabel('Center X coord.', fontsize=18)
        self.axs[0].grid(linestyle='--', linewidth=1, which='both')
        # self.axs[0].legend()

        self.axs[1].plot(self.FRAME, self.Center1_Y, label="Center - Y", color="k", linestyle="--", marker=".")
        # self.axs[1].plot(Frame, detrend_Center_0_Y, label="HPF - " + str(HPF_FREQ))
        self.axs[1].set_xlabel('Frame #', fontsize=18)
        self.axs[1].set_ylabel('Center Y coord.', fontsize=18)
        self.axs[1].grid(linestyle='--', linewidth=1, which='both')
        # self.axs[1].legend()

        # use the values to set  ylim
        # get min and max y
        ylim_min = self.Center1_X.min()
        ylim_max = self.Center1_X.max()
        self.axs[0].set_ylim(ylim_min, ylim_max)

        ylim_min = self.Center1_Y.min()
        ylim_max = self.Center1_Y.max()
        self.axs[1].set_ylim(ylim_min, ylim_max)

        range_frames_x = RectangleSelector(self.axs[0], self.los_rect_callback_x,
                                           drawtype='box', useblit=False, button=[1],
                                           minspanx=5, minspany=5, spancoords='pixels',
                                           interactive=True)

        range_frames_y = RectangleSelector(self.axs[1], self.los_rect_callback_y,
                                           drawtype='box', useblit=False, button=[1],
                                           minspanx=5, minspany=5, spancoords='pixels',
                                           interactive=True)
        axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor=axcolor)
        self.slider_CUTOFF_FREQ = Slider(axfreq, 'HPF Cutoff Frequency', 0, 1, valinit=0, valstep=0.005)
        self.slider_CUTOFF_FREQ.on_changed(self.update_los_fig)
        self.slider_CUTOFF_FREQ.on_changed(self.update_los_fig)

        plt.show()
        title = "LoS_results"
        self.fig.savefig(title + '.png', dpi=300, bbox_inches='tight')

    def initial_extensiometer_plot(self):
        # Plotting + Rangeselection
        self.fig, self.axs = plt.subplots(3, 1, figsize=(5, 5))
        self.fig.suptitle(f'Digital Extensiometer')
        # fig.suptitle(f'Boresight retention ; 1 pixel = {round(self.pixel_to_unit, 3)} units ')

        self.axs[0].plot(self.FRAME, self.DX, label="DX", color="k", linestyle="--", marker=".")
        self.axs[0].set_ylabel('DX [pix]', fontsize=18)
        self.axs[0].tick_params(axis='both', which='major', labelsize=10)
        self.axs[0].grid(linestyle='--', linewidth=1, which='both')

        self.axs[1].plot(self.FRAME, self.DY, label="DY", color="k", linestyle="--", marker=".")
        self.axs[1].set_ylabel('DY [pix]', fontsize=18)
        self.axs[1].tick_params(axis='both', which='major', labelsize=10)
        self.axs[1].grid(linestyle='--', linewidth=1, which='both')

        self.axs[2].plot(self.FRAME, self.Distance, label="DY", color="k", linestyle="--", marker=".")
        self.axs[2].set_xlabel('Frame #', fontsize=18)
        self.axs[2].set_ylabel('Distance [pix]', fontsize=18)
        self.axs[2].tick_params(axis='both', which='major', labelsize=10)
        self.axs[2].grid(linestyle='--', linewidth=1, which='both')

        #
        # use the values to set  ylim
        # get min and max y
        ylim_min = self.DX.min()
        ylim_max = self.DX.max()
        self.axs[0].set_ylim(ylim_min, ylim_max)

        ylim_min = self.DY.min()
        ylim_max = self.DY.max()
        self.axs[1].set_ylim(ylim_min, ylim_max)

        ylim_min = self.Distance.min()
        ylim_max = self.Distance.max()
        self.axs[2].set_ylim(ylim_min, ylim_max)

        range_frames_x = RectangleSelector(self.axs[0], self.extensiometer_rect_callback_x,
                                           drawtype='box', useblit=False, button=[1],
                                           minspanx=5, minspany=5, spancoords='pixels',
                                           interactive=True)

        range_frames_y = RectangleSelector(self.axs[1], self.extensiometer_rect_callback_y,
                                           drawtype='box', useblit=False, button=[1],
                                           minspanx=5, minspany=5, spancoords='pixels',
                                           interactive=True)
        range_frames_z = RectangleSelector(self.axs[2], self.extensiometer_rect_callback_z,
                                           drawtype='box', useblit=False, button=[1],
                                           minspanx=5, minspany=5, spancoords='pixels',
                                           interactive=True)

        axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor=axcolor)
        self.slider_CUTOFF_FREQ = Slider(axfreq, 'LPF Cutoff Frequency', 0, 1, valinit=1, valstep=0.005)
        self.slider_CUTOFF_FREQ.on_changed(self.update_extensiometer_fig)
        self.slider_CUTOFF_FREQ.on_changed(self.update_extensiometer_fig)

        plt.show()

        title = "Extensiometer_results"
        self.fig.savefig(title + '.png', dpi=300, bbox_inches='tight')

    def Analyze(self):
        # Initializing PARAMETERS
        self.range_x = []
        self.range_y = []
        self.Center1_X_detrened = None
        self.Center1_Y_detrened = None
        self.DX_detrened = None
        self.DY_detrened = None
        self.Distance_detrened = None

        # READING COLUMNS
        try:
            self.FRAME = self.centers_df.loc[0:, "Frame"].astype(int)
            self.Center1_X = self.centers_df.loc[0:, "Center1_X"].astype(float)
            self.Center1_Y = self.centers_df.loc[0:, "Center1_Y"].astype(float)
            self.Center2_X = self.centers_df.loc[0:, "Center2_X"].astype(float)
            self.Center2_Y = self.centers_df.loc[0:, "Center2_Y"].astype(float)
            self.Distance = self.centers_df.loc[0:, "Distance"].astype(float)

            print("self.FRAME", self.FRAME)
            print("self.Center1_X", self.Center1_X)
            print("self.Center1_Y", self.Center1_Y)
            print("self.Center2_X", self.Center2_X)
            print("self.Center2_Y", self.Center2_Y)
            print("self.Distance", self.Distance)
        except:
            print("Could not read. Check Results.csv")
            ctypes.windll.user32.MessageBoxW(0, "Could not read. Check Results.csv", "Info", 0)

        ###############################################################################
        if self.POSTPROCTASK == "LoS":
            # plot
            self.initial_los_plot()
            # Getting Detrened Signals
            if self.slider_CUTOFF_FREQ.val > 0 and self.slider_CUTOFF_FREQ.val < 1:
                b, a = signal.butter(3, self.slider_CUTOFF_FREQ.val, 'high')
                self.Center1_X_detrened = signal.filtfilt(b, a, self.Center1_X)
                self.Center1_Y_detrened = signal.filtfilt(b, a, self.Center1_Y)
            else:
                self.Center1_X_detrened = self.Center1_X
                self.Center1_Y_detrened = self.Center1_Y

            wanted_cols = {"Frame": self.FRAME, "DX": self.Center1_X_detrened, "DY": self.Center1_Y_detrened}

        ###############################################################################
        if self.POSTPROCTASK == "Extensiometer":
            # adding Distances calculations
            self.DX = self.Center1_X - self.Center2_X
            self.DY = self.Center1_Y - self.Center2_Y

            # Calc Extensions --> DL - L0
            self.DX = self.DX - self.DX[0]
            self.DY = self.DY - self.DY[0]
            self.Distance = self.Distance - self.Distance[0]

            # plot
            self.initial_extensiometer_plot()

            # Getting Detrened Signals
            if self.slider_CUTOFF_FREQ.val > 0 and self.slider_CUTOFF_FREQ.val < 1:
                b, a = signal.butter(3, self.slider_CUTOFF_FREQ.val, 'low')
                self.DX_detrened = signal.filtfilt(b, a, self.DX)
                self.DY_detrened = signal.filtfilt(b, a, self.DY)
                self.Distance_detrened = signal.filtfilt(b, a, self.Distance)
            else:
                self.DX_detrened = self.DX
                self.DY_detrened = self.DY
                self.Distance_detrened = self.Distance
            wanted_cols = {"Frame": self.FRAME, "DX": self.DX_detrened, "DY": self.DY_detrened,
                           "Distance": self.Distance_detrened}

        self.Plots_DF = pd.DataFrame(wanted_cols)

        # Gaussians:
        self.PLOT_HISTOGRAMS = True
        if self.PLOT_HISTOGRAMS:
            self.plot_los_histograms()
        # Correlations:
        self.PLOT_CORR = True
        if self.PLOT_CORR:
            self.plot_los_correlation()
        # Scatter Plot:
        self.PLOT_SCATTER_XY = True
        if self.PLOT_SCATTER_XY:
            self.plot_los_scatter()
        # Scatter Plot:
        self.PLOT_PAIRPLOT = True
        if self.PLOT_PAIRPLOT:
            self.Pariplot()

        plt.show()

    def Pariplot(self):
        sns.set()
        g = sns.pairplot(self.Plots_DF, height=2.5, corner=True)
        title = " PairPlot - "
        plt.suptitle(title)
        g.map_lower(sns.kdeplot, levels=4, color=".2")
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')

    def plot_los_histograms(self):
        ################################## Histograms:
        '''
        Histogram - Kurtosis and skewness.
        Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.
        '''
        fig = plt.figure()
        mean = round(self.Center1_X_detrened.mean(), 2)
        std = round(self.Center1_X_detrened.std(), 2)
        var = round(self.Center1_X_detrened.var(), 2)
        # skew = round(self.Center1_X_detrened .skew(), 2)
        # kurt = round(self.Center1_X_detrened .kurt(), 2)
        skew = 0
        kurt = 0
        new_line = '\n'
        stats_str = f" Statistics: {new_line} mean: {mean} {new_line} std.: {std} {new_line} var.: {var} {new_line} skewness: {skew} {new_line} kurtosis: {kurt}  "
        ax_hist_x = sns.distplot(self.Center1_X_detrened, fit=norm)
        title = "DX (pixels) Histogram"

        plt.title(title)

        x_lim = ax_hist_x.get_xlim()
        y_lim = ax_hist_x.get_ylim()
        ax_hist_x.text(0.85 * x_lim[0], 0.7 * y_lim[1], stats_str, bbox=dict(facecolor='blue', alpha=0.1))
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
        ###
        # fig = plt.figure()
        # res = stats.probplot(delta_disp_DF_Simulation["DX"], plot=plt)
        # title = "Q-Q plot DX - " + center_title
        # plt.savefig(title + '.png', dpi=300, bbox_inches='tight')

        ###
        fig = plt.figure()
        mean = round(self.Center1_Y_detrened.mean(), 2)
        std = round(self.Center1_Y_detrened.std(), 2)
        var = round(self.Center1_Y_detrened.var(), 2)
        # skew = round(self.Center1_Y_detrened.skew(), 2)
        # kurt = round(self.Center1_Y_detrened.kurt(), 2)
        skew = 0
        kurt = 0

        new_line = '\n'
        stats_str = f" Statistics: {new_line} mean: {mean} {new_line} std.: {std} {new_line} var.: {var} {new_line} skewness: {skew} {new_line} kurtosis: {kurt}  "
        ax_hist_y = sns.distplot(self.Center1_Y_detrened, fit=norm)

        title = "DY (pixels) Histogram"
        plt.title(title)
        x_lim = ax_hist_y.get_xlim()
        y_lim = ax_hist_y.get_ylim()
        ax_hist_y.text(0.85 * x_lim[0], 0.7 * y_lim[1], stats_str, bbox=dict(facecolor='blue', alpha=0.1))
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
        ###
        # fig = plt.figure()
        # res = stats.probplot(self.Center1_Y_detrened, plot=plt)
        # title = "Q-Q plot DY - " + center_title
        # plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
        ###

    def plot_los_correlation(self):
        # corr = np.corrcoef(list(df['Center1_X']), list(df['Center1_Y']))
        # print("CORR",corr)

        plt.figure(figsize=(6, 6))
        heatmap = sns.heatmap(self.Plots_DF.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
        title = 'Correlation Heatmap - '
        heatmap.set_title(title, fontdict={'fontsize': 18}, pad=12)
        # save heatmap as .png file
        # dpi - sets the resolution of the saved image in dots/inches
        # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')

    def plot_los_scatter(self):
        ##########################################################
        # SCATTER PLOTS:
        plt.figure(figsize=(12, 8))
        dx = self.Center1_X_detrened
        dy = self.Center1_Y_detrened

        corr = np.corrcoef(list(dx), list(dy))
        # corr_test = np.corrcoef(list(delta_disp_DF_Test['DX']), list(delta_disp_DF_Test['DY']))
        print("CORR", corr)
        # print("CORR", corr_test)

        label = "Correlation = " + str(round(corr[0, 1], 3))
        # label_test = "Test ; Correlation = " + str(round(corr_test[0, 1], 1))

        sns.kdeplot(x=dx, y=dy, fill=True)
        sns.scatterplot(x=dx, y=dy, label=label, color="r", alpha=0.95, s=7)

        title = "DX vs. DY (pixels) Corr. Comparison"
        plt.xlabel('DX', fontsize=18)
        plt.ylabel('DY', fontsize=18)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')


root = tk.Tk()
app = App(root)
root.mainloop()
