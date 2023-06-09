# Evaluating Unsupervised Denoising Requires Unsupervised Metrics 
## Problem

In many real-world imaging applications, such as microscopy and astronomy, it is not possible to obtain ground-truth clean images. This makes it difficult to evaluate the performance of unsupervised denoising methods, which are trained on noisy data without access to ground-truth.

## Solution

We propose two novel metrics for evaluating unsupervised denoising methods: the unsupervised mean squared error (MSE) and the unsupervised peak signal-to-noise ratio (PSNR). These metrics are computed using only noisy data, and they are shown to be asymptotically consistent estimators of the supervised MSE and PSNR. This means that they provide accurate approximations of the performance of a denoising method, even when ground-truth is not available.

The proposed metrics provide a way to evaluate unsupervised denoising methods in a fair and consistent manner. They can be used to compare the performance of different denoising methods, and they can be used to track the progress of denoising methods over time. The proposed metrics are a valuable tool for the development and evaluation of unsupervised denoising methods.

## Details

* The unsupervised MSE (uMSE) is computed as the mean squared error between the noisy image and the denoised image plus a correction term to remove the bias introduced by using the noisy image as reference.
* The unsupervised PSNR (uPSNR) is computed from the uMSE value.
* Both metrics are asymptotically consistent estimators of the supervised MSE and PSNR, which means that they provide accurate approximations of the performance of a denoising method as the amount of data increases.

## Evaluation

In our paper, we evaluate the proposed metrics on a variety of synthetic and real-world datasets. The results show that the proposed metrics are accurate and can be used to improve the development of unsupervised denoising methods.

## Conclusion

The proposed metrics are a valuable tool for the development and evaluation of unsupervised denoising methods. They can be used to compare the performance of different denoising methods, and they can be used to track the progress of denoising methods over time.
