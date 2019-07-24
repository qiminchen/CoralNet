# CoralNet

## Data

### Organization

Take s102 as example:
  
    -- s102:
      -- meta.json
      -- images:
        |-- i60774.jpg          # image 1
        |-- i60774.meta.json
        |-- i60774.anns.json
        |-- i60774.features.json
        
        |-- ...                 # image 2
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
