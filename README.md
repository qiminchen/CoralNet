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

  1. There are around 1 million images in coralnet with more than 1000 labels, should talk to NOAA about identifying and merging them to 1000 (TBD) labels that are most common and used across the most sources, also discard the rare labels.
  
  2. Once the labels are set, use 224x224 windows to crop the image centering at each annotation coordinate. Two ways to do it: one is cropping the images in advance and storing them in different folders according to their labels, the other is cropping the images when loading the Dataset, I suggest cropping the annotation in advance.
  
  3. Write the images path and "source"/"target" chunks to *all_images.txt* and *is_train.txt* for convenient data loading.

## Training
