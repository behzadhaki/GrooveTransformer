# UPF-HPC-TUTORIAL

## Install Environments Via Conda


Now 









------
Tutorial and Cheatsheet for Using DTIC Clusters

  <h5>References 
  
    [1] https://jonykoren.medium.com/slurm-and-high-performance-computing-hpc-8de939871c2c  
    [2] https://gerardmjuan.github.io/2018/07/16/Guide-For-Conda-Environment/
    [3] https://guiesbibtic.upf.edu/recerca/hpc
  
  </h5>
  
  ------------------------
  
  <h4> INSTALL SOFTWARES FOR ACCESSING THE CLUSTERS REMOTELY</h4>
  <h5> 1. Install Fortinet VPN As Discussed Here:
       
        https://www.upf.edu/en/web/biblioteca-informatica/serveis-pdi/-/asset_publisher/g2SvxJwKuVBy/content/id/114979410/maximized
  
  Connect to vpn using your dtic research 
  </h5>
  
  <h5> 2. Install FileZila from Here:
  
        https://filezilla-project.org/
        
    
   Go to File > Site Manager and set up as following
   
        Connect to Host: hpc.s.upf.edu
   
  <img src="https://github.com/behzadhaki/UPF-HPC-TUTORIAL/blob/main/images/filezilla%20setup.png" width="700"/>

   Now, you can remotely access your personal files located in the home-dtic/{username} folder
   </h5>
  
  ------------------------
  -------------------------
  
  <h5> To use the cluster, we do the following in order:
  
  1. Connect to HPC via Fortinet vpn
  
  2. Connect to the login node using (in this node, you can't do any computations, but you can access your storage on server as well as make requests for computational jobs/resources
           
           ssh -X user@hpc.s.upf.edu
      
  <img src="https://github.com/behzadhaki/UPF-HPC-TUTORIAL/blob/main/images/cluster_structure.png" height="400"/>

  3. To do computations, we can either submit a remote job or request an interactive resource via
    
    a. For interactive jobs, use srun/scancel commands: 
  
    b. For remote jobs, use sbatch job.sh, where job.sh is a bash script of the resources to request, environments/softwares to load and the code to run. See examples in job request samples folder
  
  
  <b>  <h6> 3A. Interactive Job Requests Via srun/scancel  </h6> </b>
  
  Make a request using (last two files are needed for accessing available modules in interactive mode)
          
```commandline
srun --nodes=1 --partition=short --gres=gpu:1 --cpus-per-task=4 --mem=8g --pty bash -i
source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh
```





  
  In the case of interactive resources (via srun), once you make a request, wait until you get allocated the requested resources. When this happens, <b>user@login01</b> will change to <b>user@nodeXX</b>
    
The <b>--partition</b> argument, we specify for how long we will need a resource. partition can be short, medium or long as shown in the following figure
  
 <img src="https://github.com/behzadhaki/UPF-HPC-TUTORIAL/blob/main/images/Resources.png" height="200"/>
  
  If you need access to Jupyter notebooks, refer to the tutorial in [1]
  
  <b>  <h6> 3B. Remote Jobs </h6> </b>

For remote jobs, use <b> sbatch job.sh </b>, where job.sh is a bash script of the resources to request, environments/softwares to load and the code to run. See examples in job request samples folder. An example of a job.sh file below (from [1]):


      #!/bin/bash
      #SBATCH -J job_name # your name for the job
      #SBATCH -p high # medium high or low 
      #SBATCH -N 1 # number of nodes
      #SBATCH --workdir=/homedtic/username/ # Working directory
      #SBATCH --gres=gpu:1 # GPU request
      #SBATCH --mem=125g # CPU request
      #SBATCH -o /homedtic/username/%N.%J.job_name.out # STDOUT # extract output from the running file to this directory
      #SBATCH -e /homedtic/username/%N.%J.job_name.err # STDOUT # extract errors from the running file to this directory

      export PATH="$HOME/project/anaconda3/bin:$PATH"
      export PATH="$/homedtic/ikoren/project/anaconda3/envs/name_of_env:$PATH"
      source activate name_of_env
      cd /homedtic/username/another/folder/
      python train.py



  #### NOTE: 
  DO NOT REQUEST MORE RESOURCES THAN YOU NEED. REQUESTING MORE THAN NEEDED WILL MAKE IT HARDER TO GET ACCESS NEXT TIMES YOU MAKE A REQUEST

  
 
   ------------------------
  -------------------------
  
### Setting Up Anaconda Environments On the Cluster

<h5> Install Miniconda [1] </h5>
After logging into the login nodes (no need to connect to computational nodes yet), run the following commands

        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/project/anaconda3
        # Activate
        export PATH="$HOME/project/anaconda3/bin:$PATH"
        
You can see that a <b> project </b> folder is created in your home directory in which the conda environments are kept. By exporting the path, you will be able to activate these environments via the terminal

<h5> Creating an environment [1] </h5>

      # give your conda env. a name ('name_of_env'), and select your python version
      conda create -y -n torch_thesis python=3.6 anaconda 
      # Activate the Environment
      export PATH="$/homedtic/{username}/project/anaconda3/envs/torch_thesis:$PATH"
      source activate torch_thesis


-------------------
-------------------
<h4> Cancelling Jobs </h4>
1. run squeue, then find your Job ID.
2. scancel Job_ID

  
-----------------
------------------
to get available gpu instances on the clusters:
      
    sinfo -o "%P %G %D %N"

example if you need quadro rtx 6000, use 
    
  --gres=gpu:quadro:1
  
<img width="466" alt="Screen Shot 2021-07-29 at 12 11 19 AM" src="https://user-images.githubusercontent.com/35939495/127406420-fee58288-e681-4b2f-a459-a4d4194afd94.png">

<img width="283" alt="Screen Shot 2021-07-29 at 12 43 27 AM" src="https://user-images.githubusercontent.com/35939495/127406327-fe38e714-52ac-49a3-8d07-a780b49cb9a4.png">


  pascal --> GTX 1080 ti, tesla --> Tesla T4, Quadro --> rtx 6000

  
----------
---------
