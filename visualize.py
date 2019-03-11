import cv2
import os



def visualize(pred, gt, paths, path_to_images):

    pred_count = 0

    for i in range(len(gt)):
        #print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0])
        for person in range(gt[i].shape[0]):
            img = cv2.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(gt[i][person][0][1])) + ".png")
            height_, width_, _ = img.shape
            for frame in range(gt[i].shape[1]):

                #display ground truth
                cv2.rectangle(img, (int(gt[i][person][frame][2] * width_), int(gt[i][person][frame][3] * height_)), (
                (int(gt[i][person][frame][2] * width_) + int(gt[i][person][frame][4] * width_)),
                (int(gt[i][person][frame][3] * height_) + int(gt[i][person][frame][5] * height_))), (255, 0, 0), 1)

                #display prediction
                cv2.rectangle(img, (int(pred[pred_count][frame][0] * width_), int(pred[pred_count][frame][1] * height_)), (
                (int(pred[pred_count][frame][0] * width_) + int(pred[pred_count][frame][2] * width_)),
                (int(pred[pred_count][frame][1] * height_) + int(pred[pred_count][frame][3] * height_))), (0, 255, 0), 1)

            pred_count += 1
            cv2.imshow('ImageWindow', img)
            cv2.waitKey()




