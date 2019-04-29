import skimage.io
import os
import numpy as np
import math

def get_geometric_person_interaction(obs, paths, path_to_images, annotations):

    #look at 30 pedestrians in a frame maximum
    neighbors = 30
    eps = 0.00001
    person_relations_ = []

    for i in range(len(obs)):

        #get gt data in format [frame_ID, ped_ID, x,y,w,h]
        gt = annotations + os.path.splitext(os.path.basename(paths[i]))[0] + ".csv"
        gt_data = np.genfromtxt(gt, delimiter=',')
        frames = gt_data[0,:]

        for person in range(obs[i].shape[0]):
            person_relations = np.zeros((obs[i].shape[1], neighbors, 4))
            #print(person_relations)

            for frame in range(obs[i].shape[1]):

                image = skimage.io.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")
                height_, width_, _ = image.shape

                #get all the people in the current frame
                people_in_frame = np.where(frames == obs[i][person][frame][1])[0]

                people = 0
                #if more than 1 person in the frame
                if len(people_in_frame) > 1:
                    for ind in (people_in_frame):
                        person_id = gt_data[1,ind]
                        if person_id != obs[i][person][frame][0]:

                            x_current = obs[i][person][frame][2] * width_
                            y_current = obs[i][person][frame][3] * height_
                            w_current = obs[i][person][frame][4] * width_
                            h_current = obs[i][person][frame][5] * height_

                            x_neighbor = gt_data[2, ind] * width_
                            y_neighbor = gt_data[3, ind] * height_
                            w_neighbor = gt_data[4, ind] * width_
                            h_neighbor = gt_data[5, ind] * height_

                            #compute geometric relation
                            # G = [log( | xb−xk | / wb), log( | yb−yk | /  hb), log(wk / wb), log(hk / hb)]

                            person_relations[frame][people][0] = math.log((math.fabs(x_current - x_neighbor) / w_current) + eps)
                            person_relations[frame][people][1] = math.log((math.fabs(y_current - y_neighbor) / h_current) + eps)
                            person_relations[frame][people][2] = math.log(w_current / w_neighbor)
                            person_relations[frame][people][3] = math.log(h_current / h_neighbor)

                            people += 1

            person_relations = np.reshape(person_relations, [obs[0].shape[1], neighbors*4])
            person_relations_.append(person_relations)

    person_relations_ = np.reshape(person_relations_, [len(person_relations_), obs[0].shape[1], neighbors*4])

    return person_relations_