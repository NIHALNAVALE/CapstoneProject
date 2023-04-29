import cv2
import numpy as np
from utils.general import get_iou
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

# self.features = np.zeros((256,1))
# self.conf_box = []    

def PersonToTrack(detection,config_box):
    best_iou = 0.0
    best_box,best_mask = None, None
    for one_mask, bbox, cls, conf in detection:
            iou = get_iou(config_box,bbox)
            if(iou>best_iou):
                best_iou = iou
                best_box,best_mask = bbox,one_mask
    return best_box,best_mask

def draw(image,bbox=None,mask=None,color=(0,0,0)):  
    if mask is not None and mask.any():         
        image[mask] = image[mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
    
    if bbox is not None and len(bbox) >= 4:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
     
def CropnMask(image,bbox,mask,color=(0,255,0)):
    img_croppped = image[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    mask = (mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]).astype(np.uint8)*255
    foreground = cv2.bitwise_or(img_croppped, img_croppped, mask=mask)
            
    mask = cv2.bitwise_not(mask)
    color_background = np.zeros_like(img_croppped, dtype=np.uint8)
    color_background[:] = color
    background = cv2.bitwise_or(color_background, color_background, mask=mask)

    img_masked = cv2.bitwise_or(foreground, background)

    return img_masked


def nn_match_two_way(desc1, desc2, nn_thresh=0.9, ratio_thresh=0.9):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.
      ratio_thresh - Optional ratio between first and second nearest neighbor distances
                     below which a match is considered good.
    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    if ratio_thresh < 0.0 or ratio_thresh > 1.0:
        raise ValueError('\'ratio_thresh\' should be between 0 and 1')

    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))

    # Get NN indices and scores.
    nn_idx = np.argmin(dmat, axis=1)
    nn_scores = dmat[np.arange(dmat.shape[0]), nn_idx]

    # Compute second-nearest neighbor distances and ratios.
    nn2_dist = np.partition(dmat, 1, axis=1)[:, 1]
    nn_ratio = nn_scores / nn2_dist

    # Threshold the NN matches using both distance and ratio.
    good_matches = np.logical_and(nn_scores < nn_thresh, nn_ratio < ratio_thresh)

    # Check if nearest neighbor goes both directions and keep those.
    nn2_idx = np.argmin(dmat, axis=0)
    bi_matches = np.arange(len(nn_idx)) == nn2_idx[nn_idx]
    good_matches = np.logical_and(good_matches, bi_matches)

    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[good_matches]
    m_idx2 = nn_idx[good_matches]
    scores = nn_scores[good_matches]

    # Populate the final 3xN match data structure.
    matches = np.zeros((3, m_idx1.shape[0]))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores

    return matches


#     def mask_bg(self,image,bbox,one_mask,color=(0,255,0)):
#         img_croppped = image[bbox[1]:bbox[3],bbox[0]:bbox[2]]
#         mask = (one_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]).astype(np.uint8)*255
#         foreground = cv2.bitwise_or(img_croppped, img_croppped, mask=mask)
                
#         mask = cv2.bitwise_not(mask)
#         color_background = np.zeros_like(img_croppped, dtype=np.uint8)
#         color_background[:] = color
#         background = cv2.bitwise_or(color_background, color_background, mask=mask)

#         img_masked = cv2.bitwise_or(foreground, background)

#         return img_masked
        
        
#     def draw(self,image,bbox=None,mask=None,color=(0,0,0)):  
#         if mask is not None and mask.any():         
#             image[mask] = image[mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        
#         if bbox is not None and len(bbox) >= 4:
#             cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2) 
