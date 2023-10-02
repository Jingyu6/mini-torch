# Minitorch Implementation

This repo contains my solution to excercises from [minitorch](https://github.com/minitorch/minitorch). All tasks are implemented except 4.4b due to my limited GPU access at the time of completion. 

For tasks 3.3 and 3.4 in *Module Efficiency* which require CUDA, I used google colab T4. A relatvely easy way to do so is to upload the code into your google drive, and execute the same test command in colab notebook like: 
```
!python pytest -m task3_3
```
after mounting your google drive and do ```%cd $PATH_TO_MINITORCH```. 

Some tricky parts I found during implementation:

* permute: this can introduce hidden bugs due to lack of testing, which end up showing in later tasks. 
* pooling: be careful when doing *view* operation.
* GEMM: I looked at similar examples from [NVIDIA](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) which helps me understand why tiling (or blocking) helps with memory IO saving.
* "All indices use numpy buffers": this means that you could create a temporary *np.array* at outer loops which can reassigned with new values in inner loops. 
