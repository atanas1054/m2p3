import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

def kmeans_cluster(samples):

    kmeans = KMeans(n_clusters=3).fit(samples)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels


def visualize_result(pred, probs, obs_gt, gt, paths, path_to_images):

    pred_count = 0
    save_dir = 'results/'
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    locations = [[0, 100], [0, 200], [0, 300]]

    #probabilistic prediction
    if len(pred.shape)>3:
        for i in range(len(gt)):
            #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #out = cv2.VideoWriter(save_dir + str(i) + '.avi', fourcc, 30, (1280, 720))
            # print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0])
            for person in range(gt[i].shape[0]):

                img = cv2.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                    int(gt[i][person][0][1])) + ".png")
                height_, width_, _ = img.shape

                # selected pedestrian
                cv2.rectangle(img, (int(gt[i][person][0][2] * width_), int(gt[i][person][0][3] * height_)),
                              (
                                  (int(gt[i][person][0][2] * width_) + int(gt[i][person][0][4] * width_)),
                                  (int(gt[i][person][0][3] * height_) + int(
                                      gt[i][person][0][5] * height_))), (255, 255, 255), 2)

                for s in range(pred.shape[1]):
                    #draw probabilities
                    cv2.putText(img, str(int(probs[pred_count][s]*100)) + '%', tuple(locations[s]), 0, 5e-3 * 200,
                                tuple(colors[s]), 1)
                    for frame in range(pred.shape[2]):
                        cv2.circle(img, (int(pred[pred_count][s][frame][0] * width_ + pred[pred_count][s][frame][2] * width_),
                                        int(pred[pred_count][s][frame][1] * height_ + pred[pred_count][s][frame][3] * height_)), 2, tuple(colors[s]), -1)

                pred_count += 1

                cv2.imwrite(save_dir + str(i) + str(person)+ ".png", img)
                #cv2.imshow('ImageWindow', img)
                #cv2.waitKey()

    #single point prediction
    else:
        for i in range(len(gt)):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(save_dir+str(i)+'.avi', fourcc, 30, (1280, 720))
            #print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0])
            for person in range(gt[i].shape[0]):
                gt_obs = obs_gt[i][person]
                gt_preds = gt[i][person]
                current_preds = pred[pred_count]


                #Past observation
                for p in range(obs_gt[i].shape[1]):
                    img = cv2.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                        int(obs_gt[i][person][p][1])) + ".png")
                    img = cv2.resize(img, (1280, 720))
                    height_, width_, _ = img.shape
                    for point in range(p):
                        x_start = (int(gt_obs[point][2] * width_) + int(int(gt_obs[point][4] * width_) / 2))
                        y_start = (int(gt_obs[point][3] * height_) + int(int(gt_obs[point][5] * height_) / 2))
                        x_end = (int(gt_obs[point+1][2] * width_) + int(int(gt_obs[point+1][4] * width_) / 2))
                        y_end = (int(gt_obs[point+1][3] * height_) + int(int(gt_obs[point+1][5] * height_) / 2))

                        cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 255, 255), thickness=3, lineType=8)
                        #cv2.circle(img, (x_end,y_end), 5, (255, 255, 255), 1)


                    #cv2.imshow('ImageWindow', img)
                    #cv2.waitKey()
                    cv2.rectangle(img, (int(obs_gt[i][person][p][2] * width_), int(obs_gt[i][person][p][3] * height_)),
                                  (
                                      (int(obs_gt[i][person][p][2] * width_) + int(obs_gt[i][person][p][4] * width_)),
                                      (int(obs_gt[i][person][p][3] * height_) + int(
                                          obs_gt[i][person][p][5] * height_))), (255, 255, 255), 1)
                    #cv2.imwrite(save_dir + str(i) + '_' + str(person) + '_'+ str(int(obs_gt[i][person][p][1])) + '.jpg', img)
                    out.write(img)

                #Prediction
                # for p in range(gt[i].shape[1]):
                #
                #     height_, width_, _ = img.shape
                #     for point in range(p):
                #         x_start = (int(current_preds[point][0] * width_) + int(int(current_preds[point][2] * width_) / 2))
                #         y_start = (int(current_preds[point][1] * height_) + int(int(current_preds[point][3] * height_) / 2))
                #         x_end = (int(current_preds[point+1][0] * width_) + int(int(current_preds[point+1][2] * width_) / 2))
                #         y_end = (int(current_preds[point+1][1] * height_) + int(int(current_preds[point+1][3] * height_) / 2))
                #
                #         cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), thickness=3, lineType=8)
                #
                #     cv2.imshow('ImageWindow', img)
                #     cv2.waitKey()

                #Ground truth future + prediction
                for p in range(gt[i].shape[1]):
                    #img = cv2.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                     #int(gt[i][person][p][1])) + ".png")
                    for point in range(p):
                        x_start = (int(gt_preds[point][2] * width_) + int(int(gt_preds[point][4] * width_) / 2))
                        y_start = (int(gt_preds[point][3] * height_) + int(int(gt_preds[point][5] * height_) / 2))
                        x_end = (int(gt_preds[point+1][2] * width_) + int(int(gt_preds[point+1][4] * width_) / 2))
                        y_end = (int(gt_preds[point+1][3] * height_) + int(int(gt_preds[point+1][5] * height_) / 2))

                        cv2.line(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), thickness=3, lineType=8)

                        x_start = (int(current_preds[point][0] * width_) + int(
                            int(current_preds[point][2] * width_) / 2))
                        y_start = (int(current_preds[point][1] * height_) + int(
                            int(current_preds[point][3] * height_) / 2))
                        x_end = (int(current_preds[point + 1][0] * width_) + int(
                            int(current_preds[point + 1][2] * width_) / 2))
                        y_end = (int(current_preds[point + 1][1] * height_) + int(
                            int(current_preds[point + 1][3] * height_) / 2))

                        cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), thickness=3, lineType=8)


                    #cv2.imshow('ImageWindow', img)
                    #cv2.waitKey()
                    #cv2.imwrite(save_dir + str(i) + '_' + str(person) + '_' + str(int(gt[i][person][p][1])) + '.jpg', img)
                    out.write(img)

                for frame in range(gt[i].shape[1]) :

                    #display ground truth
                    cv2.rectangle(img, (int(gt[i][person][frame][2] * width_), int(gt[i][person][frame][3] * height_)), (
                    (int(gt[i][person][frame][2] * width_) + int(gt[i][person][frame][4] * width_)),
                    (int(gt[i][person][frame][3] * height_) + int(gt[i][person][frame][5] * height_))), (255, 0, 0), 1)

                    #display prediction
                    cv2.rectangle(img, (int(pred[pred_count][frame][0] * width_), int(pred[pred_count][frame][1] * height_)), (
                    (int(pred[pred_count][frame][0] * width_) + int(pred[pred_count][frame][2] * width_)),
                    (int(pred[pred_count][frame][1] * height_) + int(pred[pred_count][frame][3] * height_))), (0, 255, 0), 1)

                pred_count += 1
            out.release()





