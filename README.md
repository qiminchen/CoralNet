# CoralNet

## Data

### Organization

Take s102 as example:
  
    -- label_set.json             # contains all the labels id and specfic name.
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
  
  2. Once the labels are set, 
