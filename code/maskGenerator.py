from distutils import extension
from cv2 import threshold
import torch
import torchvision
import cv2
from io import BytesIO
import numpy as np
import os
import torchvision.transforms as T
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms

from turtle import shape
import cv2
import numpy as np
import random
import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names 

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(1,3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    # print(outputs)
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    
    # print(masks)
    # print(masks.shape)
    # print(masks[0])

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    masks = masks.astype(np.uint8)
    newScores = [i for i in scores if i > threshold]
    # print(newScores)
    return masks, boxes, labels, newScores

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 1 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        # masks[i] = cv2.dilate(masks[i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30)))
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]

        colorMask = [255, 255, 255]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = colorMask
        # combine all the masks into a single image
        try:
          segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        except:
          pass
        # cv2_imshow(masks[0])
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        # cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
        #               thickness=2)
        # put the label text above the objects
        # cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
        #             thickness=2, lineType=cv2.LINE_AA)
    
    return image


def mask_expand(masks, valWidth, valHeight):
  newMask = []
  # for key, value in image_dictionary.items():
  #   data = image_dictionary[key]
  #   for subVal in data:
  #       if 'mask' in subVal:
  #           mask = subVal['mask']
  #           print(mask)
  #           maskNew = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (valWidth, valHeight)))
  #           newMask.append(maskNew)
    
  for i in range(len(masks)):
    mask = cv2.dilate(masks[i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (valWidth, valHeight)))
    newMask.append(mask)
  return np.array(newMask)


def tree_Segmentation(weight_loc, sorce_loc, conf_thresh):
  count = 0
  threshold = conf_thresh
  path = sorce_loc

  # initialize the model
  model = torch.load(weight_loc, map_location=torch.device('cpu'))
  # set the computation device
  device = torch.device('cpu')
  # load the modle on to the computation device and set to eval mode
  model.to(device).eval()

  # transform to convert the image to tensor
  transform = transforms.Compose([
      transforms.ToTensor()
  ])

  outPut_dictionary = {}

  for dirname, dirs, files in os.walk(path):
      for filename in files:
          filename_without_extension, extension = os.path.splitext(filename)
          if extension == ".jpg":
              print('found image')
              print(filename_without_extension)
              count = count +1
              print(f"count = {count}")
              image_path = os.path.join(dirname, filename)
              
              image = Image.open(image_path).convert('RGB')
              # keep a copy of the original image for OpenCV functions and applying masks
              orig_image = image.copy()
              name = filename_without_extension
              # transform the image
              image = transform(image)
              # add a batch dimension
              image = image.unsqueeze(0).to(device)

              masks, boxes, labels, scores = get_outputs(image, model, threshold)
              masks = mask_expand(masks, 30, 30)

              try:
                result = draw_segmentation_map(orig_image, masks, boxes, labels)
              except:
                pass
              
                ########Dict##################
              if masks.size:
                masks = masks.astype(int)
                outPut_dictionary[name] = [{'bbox' : boxes[i], 'mask': masks[i], 'conf': scores[i], 'class': 'tree'} for i in range(masks.shape[0])]
              else:
                count = count - 1
                print('No mask Image -- > deleting count..')
              # print(masks.shape)
              
              # print(outPut_dictionary.keys())
              
              # print(len(scores))
              # print(masks)
              # visualize the image
              try:
                print(result.shape)
                cv2.imshow(result)
              except:
                print('Unable to Display')
              # cv2.waitKey(0)
              # print(result)
              # set the save path
              save_path = f"Images/out.png"
              cv2.imwrite(save_path, result)
          else:
              print("Image NotFound")

  
  # return outPut_dictionary    


weight_loc = 'code/mask_rcnn_Tree2.pt'
image_source = 'maskInput'
thresh = 0.85

tree_Segmentation(weight_loc, image_source, thresh)

