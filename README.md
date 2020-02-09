# Nautilus

## Upload file to S3

```
aws s3 sync /local/path/to/data s3://qic003/target/dir --profile prp  # specify user profile (check ~/.aws/cridential)
```

## Cephfs

```
k create -f ./kubectl/cephfs_pvc.yml    # create only once and can delete when no needed anymore
```

## Pod

```
k create -f ./kubectl/pod_example.yml`  # create pod
k get pods                              # get pod information
k describe pod pod_name(qic003-pod)     # pod description
k exec -it pod_name(qic003-pod) bash    # launch pod
k delete pod pod_name(qic003-pod)       # delete pod
```

## Job

```
k create -f ./kubectl/job_example.yml   # create job
k get jobs                              # get job information
k describe pod qic003-job               # get job/pod description
k exec -it pod_name_in_this_job(qic003-job-zprb4) bash  # launch pod
k delete job job_name(qic003-jod)       # delete pod, remember to delete job once training is finished
```

## Environment setup

Export conda environment coralnet: `conda env export > environment.yml  # only export once or environment changed`
1. Create a Job on kubectl with computing resources
2. Clone thie repo to target directory: `git clone https://github.com/qiminchen/CoralNet.git`
3. Create conda environment coralnet: `conda env create -f environment.yml`
4. Configure AWS: 
```
NOTE: Do NOT recommend installing the awscli awscli-plugin-endpoint in local environment, instead 
      install the awscli awscli-plugin-endpoint in conda environment

pip install awscli awscli-plugin-endpoint
aws configure set plugins.endpoint awscli_plugin_endpoint

vim ~/.aws/credentials

# Add profile to ~/.aws/credentials
[default]
aws_access_key_id = xxx
aws_secret_access_key = xxx

# Add profile to ~/.aws/config
aws configure set s3api.endpoint_url https://s3.nautilus.optiputer.net
aws configure set s3.endpoint_url https://s3.nautilus.optiputer.net

aws s3 ls s3://qic003/ --profile prp   # List dir to test s3 connection

# Sync repository from s3
aws s3 sync s3://qic003/CoralNet ./qic003

```

# CoralNet

## Extract features

```
cd ../CoralNet/
./scripts/extract_features.sh --net resnet --net_version "resnet50" --net_path /path/to/well/trained/model --source "sxxx" --logdir /path/to/save/features/dir
```

## Train Logistic Regression classifier

NOTE that `outdir` in `extract_features.py` has to be the same as `data_root` in `eval_local.py` for evaluating the new CoralNet purpose.
```
cd ../CoralNet/
./scripts/eval.sh --outdir ../path/to/save/training/status/dir --epochs 10
```

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

  1. Divide the data sources into "source" and "target" sets. Make sure that "target" set does not include any sources Oscar gave me earlier this year. In particular, make sure source 16 (https://coralnet.ucsd.edu/source/16/) is in the "target" set. This is to enable a comparison on the PLC dataset.
  
  2. There are around 1 million images in coralnet with more than 1000 labels, should talk to NOAA about identifying and merging them to 1000 (TBD) labels that are most common and used across the most sources, also discard the rare labels.
  
  3. Once the labels are set, use 224x224 windows to crop the image centering at each annotation coordinate: cropping the images when loading the Dataset. Be cautious about `__getitem__` function.
  
  4. Write the images path and "source"/"target" chunks to *all_images.txt* and *is_train.txt* for convenient data loading.

## Training

  1. Use the EfficientNet b4 configuration pre-trained on ImageNet as the pre-trained model and replace the last fully connected layer.
  
  2. Use the "source" chunks to fine tune the entire network.
  
  3. Use the "target" set to evaluate the fine-tuned EfficientNet. For each target source, split it into the training set and testing set (Use 1/8 for testing, 7/8 for training). Push all patches through the base-network, store the features then train a logistic regression classifier (e.g. scikit-learn toolkit). Evaluate the classifier on the test set.
  
  4. [BONUS] replace the last layer in pre-trained net and fine-tune on each source.
  
## Key results plot

  1. "Target" set performance versus the beta features. **The key plot is a histogram of difference in performance, per source, compared to the beta features.** This plot would be directly relevant to the performance of CoralNet.
  
  2. A comparison on Pacific Labeled Corals. Results in the same format as following woule be great:
    
      a) Intra-expert: self-consistency of experts.
      
      b) Inter-expert: agreement level across experts.
      
      c) Texton: Oscar's CVPR 2012 classifier (HOG+SVM).
      
      d) ImageNet feat: features from ImageNet backbone. Logistic regression classifier trained on each source.
      
      e) ImageNet BP: net pre-trained on ImageNet. Fine-tuned on each source in the "target" set.
      
      f) CoralNet feat: net pre-trained on CoralNet data. Logistic regression classifier trained on each source.
      
      g) CoralNet BP: net pre-trained on CoralNet data. Fine-tuned on each source in the "target" set.
      
      h) **[THIS PROJECT]**: Res (Efficient) Net pre-trained on new CoralNet data export. Logistic regression classifier trained on each source.
      
      i) **[THIS PROJECT]**: Res (Efficient) Net pre-trained on new CoralNet data export. Fine-tuned on each source in the "target" set. (Same as training step 4)

## Analysis and hyper-parameter sweeps (BONUS)

  1. Create base-net with smaller / larger receptive field.
  2. Try ResNet50 or deeper ResNet
  
## Progress

  1. Successfully ran the code with EfficientNet and ResNet50 models on Moorea Labelled Dataset.
  
  2. Confusion matrix of EfficientNet and ResNet50:
  
      <img src="/images/EfficientNet-B4.png"  width="400" height="400"><img src="/images/ResNet50.png"  width="400" height="400">
      
  3. CPU runtime
  
      **ResNet50**: maximum of 11 seconds spent of all batches with a batch size of 32, an average of 6 seconds spent in all batches with a batch size of 32, overall 10m 49s spent in forwarding 100 batches (3200 images).
      
      <img src="/images/resnet50_cpu_runtime.png">
      
      **EfficientNet**: maximum of 20 seconds spent of all batches with a batch size of 32, an average of 8 seconds spent in all batches with a batch size of 32, overall 13m 6s spent in forwarding 100 batches (3200 images).
      
      <img src="/images/efficientnet_cpu_runtime.png">
      
  4. Going on ...
