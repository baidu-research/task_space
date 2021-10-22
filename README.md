# Exploiting a Zoo of Checkpoints for Unseen Tasks  

<p align="left">
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp
<img src="imgs/baidu-research-logo.png" width="220">
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp
<img src="imgs/purdue_logo.png" width="150">
</p>

This repo includes code to reproduce all results in the above Neurips paper.

## Dependencies
We used python 3.8.5, but other versions close to that should also work. Install all required packages by
```
pip install --upgade pip
pip install -r requirements.txt
```
We used cuda 10.2.89, but any version that meets pytorch's requirement should also work.


## Experiments
Check LMs/README.md for reproducing results on computational linguistics

Check vision/README.md for reproducing results on computer vision

***Note:*** This project requires running many small jobs. So it will be very useful if you have a cluster powered by slurm, which can launch jobs in parallel. Therefore in the job-launching scripts, you can see multiple commands like
```
sbatch -p $partition --gres=gpu:1 --wrap "python run.py" -o $job_log_path
```
If you do not have such a cluster, just use
```
python run.py > $job_log_path
```
instead.