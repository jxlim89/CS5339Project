# CS5339Project
Project on testing DNN robustness using Jacobian Regularization

### 1. Train own Models using jacobian regularization
```
MNIST_Jacobian.ipynb

CifarModelJacobian.ipynb
```

### 2. Test trained models using DeepFool algorithm
```
test_deepfool_cifar.py 

test_deepfool_MNIST.py
```
### 3. Requirements

```
Python 3.7.6

Pytorch / torchvision

matplotlib

PIL
```

### 4. Misc
Pre-trained models are available in Model directory

Test Images are provided in Images directory

Run the following directly to see results on pre-trained models

```python ./test_deepfool_MNIST.py ```

or

```python ./test_deepfool_cifar.py ```
