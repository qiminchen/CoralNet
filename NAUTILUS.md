# Nautilus Tutorial

## Get access
Refer to [Nautilus document](https://ucsd-prp.gitlab.io/userdocs/start/get-access/) for more details, here are the basic steps:

  1. Browse [PRP Nautilus portal](https://nautilus.optiputer.net/)
  
  2. Click Login at the top right corner
  3. You will be redirected to the "CILogon" page: select "Google" and login with google email
  4. On first login you become a **guest**, register at [Rocketchat](https://rocket.nautilus.optiputer.net/home), pin the admin in the # general channel to upgrade your account from **guest** to **user**
  5. Ask Professor Kriegman to create an namespace and add you to the namespace. Alternatively, you can request to be promoted to the **admin** so you can create any number of namespaces and invite other users to your namespace
  
## Before using Nautilus
**IMPORTANT**: Read the [document](https://ucsd-prp.gitlab.io/userdocs/start/quickstart/) very carefully before using the Nautilus, here are the basic setups summarized from the document:
  
  1. [Install](https://kubernetes.io/docs/tasks/tools/install-kubectl/) the kubectl tool (kubectl is a command-line tool that allows you to run commands against Nautilus clusters)
  2. Login to [PRP Nautilus portal](https://nautilus.optiputer.net/), click the **Get Config** link on top right corner to download the configuration file
  3. Create `/.kube` folder using the command line `mkdir ~/.kube` if not exist and copy the configuration file to this folder
  4. Use the command line to test if kubectl can connect to the Nautilus cluster
  
          $ kubectl get pods
     If you have multiple namespaces, specify the namespace to check the cluster
          
          $ kubectl get pods -n your_namespace
     If you’ve got `No resources found`., this indicates your namespace is empty and you can start running in it.
     
## Create Nautilus cluster
### Storage
#### < Ceph Posix >
Refer to [Storage](https://ucsd-prp.gitlab.io/userdocs/storage/toc-storage/) for more details. Before spawning a cluster, you need to create PersistentVolumeClaim (PVC) so it can be attached when spawning the cluster. Note the PVC will not be destroyed when the cluster is terminated so it could be used for storing the intermediate results. The PVC can be used for storing training dataset but I personally do not recommend as intensive I/O operations in PVC is extremely slow, especially for tasks where the dataset is larger than 50Gi and the size per training data is small.

Copy and paste the following code to `e.g. ceph.yml` file, change the `name` and `storage e.g. 500Gi` based on your needs:
    
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: ChangeNameHere
    spec:
      storageClassName: rook-cephfs
      accessModes:
      - ReadWriteMany
      resources:
        requests:
          storage: 500Gi

Use the command line to create a PersistentVolumeClaim (PVC), this is one time operation and of course you can create multiple PVC based on your needs:

        $ kubectl create -f ceph.yml
        
#### < Ceph S3 >
Refer to [Storage](https://pacificresearchplatform.org/userdocs/storage/ceph-s3/) for more details. I recommend using nautilus S3 for dataset storage. It used the same S3 protocol but is not related to Amazon. 
    
  1. Login to [Rocketchat](https://rocket.nautilus.optiputer.net/home), request `aws_access_key_id` and `aws_secret_access_key` from the admin
    
  2. Highly recommend installing `awscli` and `awscli-plugin-endpoint` in conda environment using
    
          $ pip install awscli awscli-plugin-endpoint
          $ aws configure set plugins.endpoint awscli_plugin_endpoint
  
  3. Configure `~/.aws/credentials`
  
          $ vim ~/.aws/credentials
          # Add the following profile to ~/.aws/credentials
          [default]
          aws_access_key_id = aws_access_key_id_requested_from_admin
          aws_secret_access_key = aws_secret_access_key_requested_from_admin
          
  4. Configure `~/.aws/config` by running the command line:
  
          $ aws configure set s3api.endpoint_url https://s3.nautilus.optiputer.net
          $ aws configure set s3.endpoint_url https://s3.nautilus.optiputer.net
          
  5. Create a bucket by running the command line, change the `--region` based on your location:
  
          $ aws s3api create-bucket --bucket bucket-name --region us-west-2
          
  6. Test the connection by running the command line:
  
          $ aws s3 ls s3://bucket-name/
          
### Spawn cluster
Refer to [Running](https://pacificresearchplatform.org/userdocs/running/jupyter/) for more details. Nautilus has two types of cluster: `Pod` and `Job`

1. **`Pod`**: You can only request 2-GPU and 8G CPU RAM at most as the pod example will be automatically destroyed in 6 hours. Using `Pod` for interactive debugging is suggested. Refer to [POD](https://pacificresearchplatform.org/userdocs/running/jupyter/) for more details.
  
    Copy and save the following code to `e.g. pod_example.yml` file, change the configuration to which the arrows point and remove arrows and symbols before saving the file:
      
        apiVersion: v1
        kind: Pod
        metadata:
          name: PodName  <------ change here
        spec:
          restartPolicy: Never
          containers:
          - name: gpu-container
            image: gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest  <------ default image provided by nautilus, change to your own image if needed
            args: ["sleep", "infinity"]
            volumeMounts:
            - mountPath: /PathName  <------ change name here
              name: Name_1  ---------
            - mountPath: /dev/shm   |
              name: dshm            |
            resources:              |
              limits:               |
                memory: 8Gi         |  <------ these two names should be the same, do not forget to delete these symbols
                nvidia.com/gpu: 1   |
              requests:             |
                memory: 8Gi         |
                nvidia.com/gpu: 1   |
          volumes:                  |
            - name: Name_1  ---------
              persistentVolumeClaim:
                claimName: Name_2  <------ name here should be the same as the name in ceph.yml above
            - name: dshm
              emptyDir:
                medium: Memory
  
    Spawn a pod:
  
        $ kubectl create -f ./path/to/pod_example.yml
  
    Get pod list:
      
        $ kubectl get pods
      
    Get pod description:
      
        $ kubectl describe pod pod_name
      
    Launch pod in interactive mode:
  
        $ kubectl exec -it pod_name bash
      
    Delete pod:

        $ kubectl delete pod pod_name

2. **`Job`**: `Jobs` in Nautilus are not limited in runtime, you can only run jobs with meaningful `command` field. Running in manual mode (`sleep infinity command` and manual start of computation) is **prohibited**. Refer to [JOB](https://pacificresearchplatform.org/userdocs/running/jobs/) for more details.

    Copy and save the following code to `e.g. job_example.yml` file, change the configuration to which the arrows point and remove arrows and symbols before saving the file:
        
        apiVersion: batch/v1
        kind: Job
        metadata:
           name: JobName  <------ change here
        spec:
          template:
            spec:
              containers:
              - name: gpu-container
                image: gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest  <------ default image provided by nautilus, change to your own image if needed
                command: ["/bin/bash", "-c"]                     |
                args:                                            |
                  - source ../miniconda/etc/profile.d/conda.sh;  |   <----- one way to run your program automatically, please change
                    conda activate coralnet;                     |
                    chmod +x scripts/train.sh;                   |
                    ./scripts/train.sh;                          |
                volumeMounts:
                - mountPath: /PathName  <------ change name here
                  name: Name_1  ----------
                - mountPath: /dev/shm    |   <------ these two names should be the same
                  name: dshm             |
                resources:               |
                  limits:                |
                    memory: 96Gi         |
                    nvidia.com/gpu: 1    |
                    cpu: "6"             |
                  requests:              |
                    memory: 64Gi   <------------- only require resources you need
                    nvidia.com/gpu: 1   <------------- only require resources you need
                    cpu: "3"    <------------- only require resources you need
              volumes:                   |
                - name: Name_1  ----------
                  persistentVolumeClaim:
                    claimName: Name_2  <------ name here should be the same as the name in ceph.yml above
                - name: dshm
                  emptyDir:
                    medium: Memory
                    
    Spawn a job:
  
        $ kubectl create -f ./path/to/job_example.yml
  
    Get job list:
      
        $ kubectl get jobs
        
    Get pod list:
      
        $ kubectl get pods
    when spawning a job, a pod without runtime limit will be spawned too and it will NOT be completed until your scripts running is finished. All the computational resources will be released once the job is completed.
      
    Get pod description:
      
        $ kubectl describe pod pod_name
      
    Launch pod in interactive mode:
  
        $ kubectl exec -it pod_name bash
      
    Delete pod:

        $ kubectl delete pod pod_name
        
Please refer to [Nautilus Document](https://pacificresearchplatform.org/userdocs/start/toc-start/) for original tutorial.
