# Distributed training on Azure with CIFAR10 and PyTorch

This tutorial is largely inspired by [this tutorial](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/pipelines/2b_train_cifar_10_with_pytorch). It expands on the concepts [previously learned](../1_mnist-pytorch-lit/mnist-pytorch-lit.ipynb) and adds the challenge of distributed training. More specifically, we automate the different training tasks using pipelines and define them in a YAML format. 

This tutorial assumes you have:
    - An Azure account with an active subscription **with a quota of at least 20 core**. 
    - An Azure ML workspace with two compute clusters (one CPU-based and the other GPU-enabled).
    - A Python environment with Azure Machine Learning Python v2 installed.

By the end of this tutorial, you should be able to:
- Define `CommandComponent` using YAML.
- Create basic `Pipeline` using component from local YAML file.
- Run distributed jobs using PyTorch's native distributed training capabilities.

## Resources
- [Azure tutorial: Train CIFAR10 with PyTorch](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/pipelines/2b_train_cifar_10_with_pytorc)
- [Blog post: How distributed training works](https://theaisummer.com/distributed-training-pytorch/)