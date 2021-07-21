import os
import sys
import glob

import dlib

if len(sys.argv) != 2:
    print('err')
    exit()
airplane_folder = sys.argv[1]


options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False

options.C = 5

options.num_threads = 2
options.be_verbose = True


training_xml_path = os.path.join(airplane_folder, "mydataset.xml")
testing_xml_path = os.path.join(airplane_folder, "test/testdataset.xml")

dlib.train_simple_object_detector(training_xml_path, "airplane_detector.svm", options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "airplane_detector.svm")))

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "airplane_detector.svm")))

detector = dlib.simple_object_detector("airplane_detector.svm")

win_det = dlib.image_window()
win_det.set_image(detector)

print("Showing detections on the images in the airplanes/unknwon folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(os.path.join(airplane_folder, 'unknown'), "*.jpeg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()