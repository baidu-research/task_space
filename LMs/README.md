# This folder includes all code to reproduce

## 1. Estimation of ![formula](https://render.githubusercontent.com/render/math?math=\kappa) shown in fig.1, fig.2
For your convenience, we provide our estimated ![formula](https://render.githubusercontent.com/render/math?math=\kappa) at `estimate_kappa/kappa.wiki2-train-noless-10.npy`

![formula](https://render.githubusercontent.com/render/math?math=\kappa) is the input to step 2. If you are eager to get results in step 2, just use our estimate of ![formula](https://render.githubusercontent.com/render/math?math=\kappa) and ignore the remaining of this section.

### Details on estimating ![formula](https://render.githubusercontent.com/render/math?math=\kappa)

1) First we have to extract contextualized word embeddings for all checkpoints. 
This may take a while and quite a bit disk space
```
cd estimate_kappa;
./extract.sh text/wiki.train.len_noLess_10.tokens your/feature/path
```

2) Then we can compute ![formula](https://render.githubusercontent.com/render/math?math=\kappa) among all 34 checkpoints
```
python estimate_kappa.py your/feature/path ../ckpts.txt kappa.wiki2.all.npy
```
Fig.1 can be produced by
```
python show_dendrogram.py kappa.wiki2.all.npy ../ckpts.txt
```

3) We can also estimate ![formula](https://render.githubusercontent.com/render/math?math=\kappa) using only a subset of probing data, e.g.,
```
python estimate_kappa.py your/feature/path ../ckpts.txt kappa.wiki2.all.npy --max-sentence 128
```
Then we can check how quickly ![formula](https://render.githubusercontent.com/render/math?math=\kappa) converges w.r.t. number of words in probe data (fig. 2)
```
python convergence.py --data wikitext2 --metric kl --iso
```
    
## 2. All results in fig.3, and tab. 2
The train-valid-test data for all tasks can be downloaded by
```
cd probe_tasks
./get_data_and_features.sh
```

1) First prepare word representations for these tasks
```
./prepare_contextualizer.sh $one_of_the_34_ckpt  #e.g., bert-large-cased
```
This should take a while. A folder named as `contextualizers` should be created
Inside it are word representations by each checkpoint, and for every task. 

For your convenience, we have provided all pre-extracted features, using all ckpts on each task. Running `./get_data_and_features.sh` should already download these pre-extracted features under `contextualizers`.

2) Then we can run jobs for each task,
 ```
 ./chuncking.sh
 ./ner.sh
 ./pos-ptb.sh
 ./st.sh
 ./ef.sh
 ./syn-p.sh
 ./syn-gp.sh
 ./syn-ggp.sh
 ```   
This will create 8 task dirs under `task_logs`. Inside each of them are many job logs.

3) Run `./fig3.sh` to produce fig.3
