# Session 4.1 CONDA Environments and Jupyter Notebook on Expanse: Scalable & Reproducible Data Exploration and ML  

**Date:** Thursday, June 27, 2024

**Summary**: Set up reproducible and transferable software environments and scale up calculations to large datasets using parallel computing.

**Presented by:** [Peter Rose](https://www.sdsc.edu/research/researcher_spotlight/rose_peter.html) (pwrose at ucsd.edu) 

### Reading and Presentations:
* **Lecture material:**
   * Presentation slides:
* **Source Code/Examples:** [df-parallel](https://github.com/sbl-sdsc/df-parallel.git)

-----

## TASK 1: Launch Jupyter Lab on Expanse using a CONDA environment
For this task you will launch a Jupyter Lab session on an Expanse GPU node using a CONDA environment

1. Open a Terminal Window ("expanse Shell Access") through the [Expanse Portal](https://portal.expanse.sdsc.edu/training)

2. Clone the Git repository df-parallel
```
git clone https://github.com/sbl-sdsc/df-parallel.git
```
  
3. Launch Jupyter Lab using the Galyleo script on a GPU node

   This script will generate a URL for your Jupyter Lab session.
```
galyleo launch --account ${CIML24_ACCOUNT} --reservation ${CIML24_RES_GPU} --qos ${CIML24_QOS_GPU} --partition gpu-shared --cpus 10 --memory 92 --gpus 1 --time-limit 01:30:00 --conda-env df-parallel-gpu --conda-yml "${HOME}/df-parallel/environment-gpu.yml" --mamba --quiet
```

> The arguments ```--reservation ${CIM24_RES_GPU} --qos ${CIM24_QOS_GPU}``` are only active during the CIML workshop. Remove these arguments when running this example outside of the workshop and specify your project account number.

4. Open a new tab in your web browser and paste the Jupyter Lab URL.  

> You should see the Satellite Reserve Proxy Service page launch in your browser.

------

## TASK 2: Run Jupyter Lab Interactively
For this task you will compare the runtime for a simple data analysis using 5 dataframe libraries.

1. Go to the Jupyter Lab session launched in TASK 1

    Navigate to the ```df-parallel/notebooks``` directory.

2. Copy a dataset to the local scratch disk on the GPU node

    Run the ```1-FetchLocalData.ipynb``` notebook

3. Run the Dataframe notebooks

    Run the following Dataframe notebooks and write down the runtime shown at the bottom of each notebook.
```
2-PandasDataframe.ipynb
3-DaskDataframe.ipynb
4-SparkDataframe.ipynb
5-CudaDataframe.ipynb
6-DaskCudaDataframe.ipynb
```

-----

## TASK 3: Assess Parallel Efficiency
In this task you will assess how runtime scales with the number of CPU cores.

1. Run the notebook ```7-ParallelEfficiency``` with the default file format ```csv``` and the dataframe library ```Dask```.

Review the Parallel Efficiency plot. How well does Dask scale for this example?

2. Use the widgets in the notebook to rerun the analysis with a different dataframe libraries and file format and create a Parallel Efficiency plot. Describe what you found out.
 
-----

## TASK 4: Run a Jupyter Notebook in Batch
In this task you learn how to parameterize a notebook and run it in batch. This batch job compares the runtime using `csv` vs. `parquet` files for 4 dataframe libraries: Pandas (1 CPU), Dask (8 CPUs), Spark (8 CPUs), Cuda (1 GPU).

1. Parameterize the dataframe notebook
```
2-PandasDataframe.ipynb
```

The dataframe notebooks in this repo have already been parameterized, however to learn how to parameterize a notebook, first remove the current ```parameters``` tag, then add it back.
   
   - Select cell ```[3]``` in the ```2-PandasDataframe.ipynb```

   - Click the property inspector in the right sidebar (double gear icon at the top right)
  
   - Expand the ```COMMON TOOLS``` section

   - Note, the tag ```parameters``` has already been set. Remove it, then add it back

   - Type ```parameters``` in the “Add Tag” box and hit ```Enter```
   
   - Save the notebook


2. Edit the ```problem.sh``` batch script in the ```df-parallel directory```). Look at the bottom of the file for instructions.
   - Add a papermill statement for each dataframe notebook to use the ```parquet```  and ```csv``` file_format and save the executed notebook in the ```${RESULT_DIR}```

   > Check your script by comparing it with the solution.sh script.

3. Shutdown Jupyter Lab!

    ```File -> Shutdown``` to terminate the process

   > It is important to Shutdown Jupyter Lab now! Otherwise we will be running out of available GPUs for the next step.

4. Change directory to ```df-parallel```
   
5. Submit the ```problem.sh``` batch script using ```sbatch```

6. Monitor the progress of the job in the Expanse Portal

   > This jobs takes about 5 minutes to complete

7. When the job has completed, review the benchmark results in the file: `df-parallel/notebooks/results/benchmark.csv`.

   > Which dataframe libraries perform the best?
   
   > Which file format is most efficient?


[Back to Top](#top)
