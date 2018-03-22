import numpy as np
import os
from sklearn.svm import LinearSVC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from imageio import imread
from utils import *
import time
from PIL import Image

'''
RUN_DETECTOR Given an image, runs the SVM detector and outputs bounding
boxes and scores

Arguments:
    im - the image matrix

    clf - the sklearn SVM object. You will probably use the
        decision_function() method to determine whether the object is
        a face or not.
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    window_size - an array which contains the height and width of the sliding
    	window

    cell_size - each cell will be of size (cell_size, cell_size) pixels

    block_size - each block will be of size (block_size, block_size) cells

    nbins - number of histogram bins

Returns:
    bboxes - D x 4 bounding boxes that tell [xmin ymin width height] per bounding
    	box

    scores - the SVM scores associated with each bounding box in bboxes

You can compute the HoG features using the compute_hog_features() method
that you implemented in PS3. We have provided an implementation in utils.py,
but feel free to use your own implementation. You will use the HoG features
in a sliding window based detection approach.

Recall that using a sliding window is to take a certain section (called the
window) of the image and compute a score for it. This window then "slides"
across the image, shifting by either n pixels up or down (where n is called
the window's stride).

Using a sliding window approach (with stride of block_size * cell_size / 2),
compute the SVM score for that window. If it's greater than 1 (the SVM decision
boundary), add it to the bounding box list. At the very end, after implementing
nonmaximal suppression, you will filter the nonmaximal bounding boxes out.
'''
def run_detector(im, clf, window_size, cell_size, block_size, nbins, thresh=1):
    bboxes = []
    scores = []
    stride = block_size * cell_size / 2
    n_windows_vert = int((im.shape[0] - window_size[0]) / stride + 1)
    n_windows_horiz = int((im.shape[1] - window_size[1]) / stride + 1)
    for i in range(n_windows_vert):
        for j in range(n_windows_horiz):
            y_start = i * stride
            y_end = y_start + window_size[0]
            x_start = j * stride
            x_end = x_start + window_size[1]
            window = im[y_start:y_end, x_start:x_end]
            hog_features = compute_hog_features(window, cell_size, block_size, nbins)
            score = clf.decision_function(hog_features.reshape(1, -1))
            if score > thresh:
                bbox = np.asarray([x_start, y_start, window_size[1], window_size[0]])
                bboxes.append(bbox)
                scores.append(np.squeeze(score))
    return np.asarray(bboxes), np.asarray(scores)

'''
NON_MAX_SUPPRESSION Given a list of bounding boxes, returns a subset that
uses high confidence detections to suppresses other overlapping
detections. Detections can partially overlap, but the
center of one detection can not be within another detection.

Arguments:
    bboxes - ndarray of size (N,4) where N is the number of detections,
        and each row is [x_min, y_min, width, height]

    confidences - ndarray of size (N, 1) of the SVM confidence of each bounding
    	box.


Returns:
    nms_bboxes -  ndarray of size (N, 4) where N is the number of non-overlapping
        detections, and each row is [x_min, y_min, width, height]. Each bounding box
        should not be overlapping significantly with any other bounding box.

In order to get the list of maximal bounding boxes, first sort bboxes by
confidences. Then go through each of the bboxes in order, adding them to
the list if they do not significantly overlap with any already in the list.
A significant overlap is if the center of one bbox is in the other bbox.
'''
def non_max_suppression(bboxes, confidences):
    sorted_idxs = np.flip(np.argsort(confidences), axis=0)
    nms_bboxes = []
    for i in sorted_idxs:
        idx = sorted_idxs[i]
        suppress = False
        x_min, y_min, width, height = np.squeeze(bboxes[idx]).tolist()
        bbox_center = [x_min + width/2, y_min + height/2]
        for bbox in nms_bboxes:
            bbox = np.squeeze(bbox)
            if bbox_center[0] > bbox[0] and bbox_center[0] < bbox[0] + bbox[2] and bbox_center[1] > bbox[1] and bbox_center[1] < bbox[1] + bbox[3]:
                suppress = True
        if not suppress:
            nms_bboxes.append(bboxes[idx])
    return np.asarray(nms_bboxes).reshape(-1, 4)


if __name__ == '__main__':
    block_size = 2
    cell_size = 6
    nbins = 9
    window_size = np.array([36, 36])

    # compute or load features for training
    if not os.path.exists('features_pos.npy'):
        tic = time.time()
        features_pos = get_positive_features('train_dice_scenes', cell_size, window_size, block_size, nbins)
        np.save('features_pos.npy', features_pos)
        print "Took %s seconds" % (time.time() - tic)
    else:
        features_pos = np.load('features_pos.npy')
    if not os.path.exists('features_neg.npy'):
        tic = time.time()
        num_negative_examples = 10000
        features_neg = get_random_negative_features('train_non_dice_scenes', cell_size, window_size, block_size, nbins, num_negative_examples)
        np.save('features_neg.npy', features_neg)
        print "Took %s seconds" % (time.time() - tic)
    else:
        features_neg = np.load('features_neg.npy')

    X = np.vstack((features_pos, features_neg))
    Y = np.hstack((np.ones(len(features_pos)), np.zeros(len(features_neg))))

    # Train the SVM
    tic = time.time()
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(X, Y)
    score = clf.score(X, Y)
    print "Trained in %f seconds" % (time.time() - tic)

    # Part A: Sliding window detector
    #im = imread('../examples/examples/dice_002.jpg').astype(np.uint8)
    testfile = open('train.txt', 'r')
    crop_image_files = [f.strip() for f in testfile]
    image_files = []
    for crop in crop_image_files:
        image_file = 'test_dice_pics/' + crop[:len('dice_xxx')] + '.jpg'
        if image_file not in image_files:
            image_files.append(image_file)
    testfile.close()
    predfile = open('train_predictions.txt', 'w+')
    total_time = 0.0
    for image_file in image_files:
        print "Detecting dice in %s" % image_file
        tic = time.time()
        im = Image.open(image_file)
        im.load()
        im = im.convert('L')
        im = np.asarray(im)
        bboxes, scores = run_detector(im, clf, window_size, cell_size, block_size, nbins, 0.5)
        bboxes = non_max_suppression(bboxes, scores)
        print "Detected in %s seconds" % (time.time() - tic)
        total_time += time.time() - tic
        for box in bboxes:
            xmin, ymin, width, height = np.squeeze(box).tolist()
            predfile.write('%s %f %f %f %f %f\n' % (image_file[:-len('.jpg')], 1.0, xmin, ymin, xmin+width, ymin+height))
        # plot_img_with_bbox(im, bboxes, 'With nonmaximal suppresion')
        # plt.show()
    predfile.close()
    avg_time = total_time / float(len(image_files))
    print "Average detection time: %f seconds" % avg_time
