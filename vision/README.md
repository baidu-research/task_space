# This folder includes all code to reproduce the Results in Section 5.2

Imaging an admin that has to handle many new tasks, each an N-way classification problem.
Each task has a small amount of S training data (e.g., 20) per class.

The remaining training data are taken by K agents.
Each agent only sees C out of the total 100 class.
When C is big (e.g., 80), the agents' tasks are similar but slightly differ.
When C is small, the agents' tasks should differ more.

For every of its tasks, admin reports accuracy on the standard valid set (excluding unseen classes in that task)

Download cifar100 by
```
cd images;
./download_data.sh;
```
All experimental pipeline are included in `./exp.sh`. You are suggested to run it code block by code block to get the result in fig.4
