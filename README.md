# CUDA Assignment
This assignment is made to test your proficiency in problem solving in CUDA

## Objective
To get the sum of a large array in lesser time by leveraging parallelism

Traditional Reduction involves `n` operations to sum up the elements of an array. But by using the Tree method of Sum reduction we can reduc this to `logn` operations.

## Steps
- Half the threads add their value to the corresponding value from the other half (stride apart).
- The number of active threads is halved each time.

Eg. We have an array of 8 elements `[1,2,3,4,5,6,7,8]`

- 1st Iteration
```c
data[0] += data[4];
data[1] += data[5];
data[2] += data[6];
data[3] += data[7];

```
this reduces the problem size to 4 elements

- 2nd Iteration(4 elements)

```c
data[0] += data[2];
data[1] += data[3];
```

- 3rd Iteration(2 elements)

```c
data[0]+=data[1];
```

This will lead to `data[0]` holding the sum of all the elements

# References
- [CUDA Programmer's Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Parallel Reduction](https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8)
