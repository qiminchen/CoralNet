# CoralNet

## Data

### Organization

  
    -- label_set.json             # contains all the labels id and corresponding label names
    
    -- s102:
      -- meta.json                # contains source informations including number of images, create date,
                                    source name, number of valid robots, longitude, latitude, affiliation,
                                    number of confirmed images and source description.
      -- images:
        |-- i60774.jpg            # image 1, a high resolution RGB image
        |-- i60774.meta.json      # contains some image informations including height in cm, image name,
                                    aux1, aux2, aux3, photo date, etc.
        |-- i60774.anns.json      # contains annotations information of images, each annotation is composed
                                    of label, col and row (coordinate in image).
        |-- i60774.features.json  #
        
        |-- ...                   # image 2
        |-- ...
        |-- ...
        |-- ...
        
        |-- ...
    
    -- s109: ...
      -- meta.json
      -- images:
        -- ...
        
    -- ...
    -- ...

### Preprocess

  1. Divide the data **sources** into "source" and "target" chunks. Make sure that "target" chunk does not include any sources Oscar gave me earlier this year.
  
  2. There are around 1 million images in coralnet with more than 1000 labels, should talk to NOAA about identifying and merging them to 1000 (TBD) labels that are most common and used across the most sources, also discard the rare labels.
  
  3. Once the labels are set, use 224x224 windows to crop the image centering at each annotation coordinate. Two ways to do it: one is cropping the images in advance and storing them in different folders according to their labels, the other is cropping the images when loading the Dataset, I suggest cropping the annotation in advance.
  
  4. Write the images path and "source"/"target" chunks to *all_images.txt* and *is_train.txt* for convenient data loading.

## Training

  1. Use ResNet50 pretrained on ImageNet as pretrained model and replace the last fully connected layer.
  
  2. Use the "source" chunks to fine tune the entire network.
  
  3. Use the "target" chunks to evaluate the fine-tuned ResNet50. For each target source, split it into training set and testing set, replace the last layer of ResNet50 with <img src="https://latex.codecogs.com/gif.latex?k" title="k" /> classifier layer, where <img src="https://latex.codecogs.com/gif.latex?k" title="k" /> is the number of the labels of each target source. Then only fine-tune the classifier with training set and evaluate the overall accuracy and per-class accuracy with testing set.
  
  4. For each target source, after fine-tuning the fully connected layer, compare to using the features from the current network. Need to talk to Oscar about this.
  
## Comparison

  1. "target" chunks on ResNet50 pretrained on ImageNet **vs.** pretrained on CoralNet (ResNet50 after fine-tuning on "source" chunk). Both only fine-tune on the last fully connected layer.
  
  2. "target" chunks on fine-tuned ResNet50: 224x224 receptive field size **vs.** 168x168 receptive field size. Both only fine-tune on the last fully connected layer.
  
  3. Try EfficientNets if time is available.
