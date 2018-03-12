## Deep Learning Study with Gluon

Deep learning to learn while making


## Schedule

| Topic      | Date/Time | Location |   Presentor | 
|-----------|----------|--------|----------|
| Gluon Introduction |  12/22,17:00   | 분석실  |haven-jeon ,seujung     |   
| [Convolution](Convolution/cats_and_dogs_conv.ipynb) & [Transfer Learning](Convolution/cats_and_dogs_conv_transfer.ipynb) |  12/28,17:00   | 분석실  |haven-jeon  |  
| [Intro with MNIST(Logistic, FC,](Basic/MNIST_Basic.ipynb)[ CNN)](Convolution/MNIST.ipynb),[Embedding](Embedding/word2vec_skipgram_with_gluon.ipynb), [Intro Audio Data Analysis](Recurrent/Introduction%20of%20Audio%20Data.ipynb)  |  01/04,18:00   | 분석실  |soohwanjo, soeque1, hongdm7 |  
| [Style Transfer](Style_transfer/style_transfer_vgg19_gluon.ipynb)(0h20m), [Fashion MNIST](Convolution/fashion_mnist.ipynb)(0h10m), [Traffic Sign Recognition](Convolution/Traffic%20Sign.ipynb)(0h15m), [Intro MXNet](Intro%20mxnet/Intro%20mxnet%20NDArray,%20Symbol,%20Model.ipynb)(0h10m) |  01/11,18:00   | 분석실  |seujung, haven-jeon, June-H, hyemin15  | 
| [Intro to GAN](GAN/GAN_1D_Array.ipynb)(0h15m),  [Intro to VAE](VAE/notebooks/VAE.ipynb)(0h30m) |  01/16,18:00   | 분석실  |soeque1, kionkim   | 
| [Multi GPU](Basic/multi_gpu_intro.ipynb)(0h20m), [Autoencoder](autoencoder/Autoencoder_w_gluon.ipynb)(0h30m), [Transfer Learning](Convolution/FCN_Alexnet_using_Gluon.ipynb) (0h40m)  |  01/25,18:00   | 분석실  |haven-jeon, ljy3795, su-park  | 
| [RNNs with Audio Classification](Recurrent/)(0h30m), [CAM with Traffic Sign Classification](Convolution/Traffic%20Sign.ipynb)(0h 20m) | 02/01,18:00  | 분석실 |hongdm7, June-H  |  
| [Pix2Pix](GAN/pix2pix.ipynb)(0h25m), [Deep Matrix Factorization](https://github.com/ski-net/dl_study_with_gluon/blob/master/Recommendation/180214_Deep_Matrix_Factorizaiton.ipynb) (0h20m) | 02/08,18:00  | 분석실 |soeque1, ljy3795, kionkim   |
| [introduction Deep Learning references](reference/dl_reference.md)  | 03/14,12:00  | 분석실 |  seujung |
| [Soft Decision Tree](soft_decision_tree/notebooks/soft_decision_tree_ver_2.ipynb) (0h30m),   [Deep Dream](Deep_dream/deep_dream.ipynb) (0h20m)  | 02/22,18:00  | 분석실 |kionkim, soohwanjo |
| [Korean-English Neural Machine Translater](https://github.com/haven-jeon/ko_en_neural_machine_translation)(0h40m)  | 02/28,18:00  | 분석실 |haven-jeon  |
| [IntegratedGradients](Convolution/multi_gpu_transfer_cats_and_dogs_cam_grad_cam_integrated-gradients.ipynb) (0h20m), [BEGAN](GAN/BEGAN) (0h20m),[QA- Multimodal Compact Bilinear Pooling](QA/notebooks) (0h45m) | 03/08,18:00  | 분석실 | kionkim, seujung , hyemin15, soeque1  |
| 1차 롤 아웃/2차 스터디 계획 수립 | 03/13,12:00  | 분석실 |  |
|  Deep Q-Network(0h30m), Double Deep Q-Network, Policy Gradient  | 03/15,18:00  | 분석실 |hongdm7, soohwanjo, June-H |
| Anomaly detection w Autoencoder, CapsNet(0h30m)  | 03/22,18:00  | 분석실 |ljy3795, kionkim   |
| relational network,  | 03/29,18:00  | 분석실 | seujung |



- [Attendance](https://docs.google.com/spreadsheets/d/1SCedAxS5-8sB-WqNi0bNPFh-R9IXHNOnx2k2eoDbsYg/edit?usp=sharing)



## Topic

### Linear algebra

- https://github.com/fastai/numerical-linear-algebra (@seujung)

### numpy exercise

- https://github.com/Kyubyong/numpy_exercises (@seujung)

### Gluon Basic

- Gluon Introduction, 12/22
- [Intro mxnet NDArray, Symbol, Model](Intro%20mxnet/Intro%20mxnet%20NDArray,%20Symbol,%20Model.ipynb) (@hyemin15)
- intro Grad
- [intro with MNIST](Basic/MNIST_Basic.ipynb) (@soohwanjo)


### Fully Connected  

- [Classification model with MNIST](Fully_Connected) (@seujung)

### Convolution

- [Image classification with Convolution](Convolution/cats_and_dogs_conv.ipynb) (@haven-jeon, cats and dogs )
- [Fashion MNIST](Convolution/fashion_mnist.ipynb)
- [Transfer Leraning](Convolution/cats_and_dogs_conv_transfer.ipynb) (@haven-jeon, cats and dogs classification)
- [CAM and Grad CAM](Convolution/multi_gpu_transfer_cats_and_dogs_cam_grad_cam.ipynb) (@haven-jeon, cats and dogs classification)


- [Image classification with MNIST](Convolution/MNIST.ipynb) (@soohwanjo)

- Image classification with 62 classes Traffic Sign (@June-H)

- [Medical Image Segmentation](Convolution/FCN_Alexnet_using_Gluon.ipynb) (@supark)

- Super resolution (@soohwanjo)

- AlexNet
  - https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
- ResNet
  - https://arxiv.org/pdf/1512.03385.pdf
- Inception V4
  - https://arxiv.org/pdf/1602.07261.pdf
- DenseNet
  - https://arxiv.org/pdf/1608.06993.pdf
- CapsNet
  - https://arxiv.org/pdf/1710.09829.pdf



### Recurrent
- [Introduction of Audio Data](Recurrent/) (@hongdm7, Whale Sound Data)
- [RNNs with Audio Classification](Recurrent/) (@hongdm7, Whale Sound Data)
- Stock Price Prediction with Amazon Stock Data (@hyemin15)


### Image

### Deep Dream
- [Deep Dream](Deep_dream/deep_dream.ipynb) (@soohwanjo)

### Neural Style Transfer
- [neural style transfer](Style_transfer/style_transfer_vgg19_gluon.ipynb) (@seujung)
  - https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
- fast neural style transfer (@seujung)

### pix2pix
- pix2pix (@hjkim)

### NLP

- Sentiment Classification (@supark, 네이버 영화 평점)
- seq2seq (@haven-jeon, Addition model)
- seq2seq with Attention Mechanism (@haven-jeon, Addition model)
- Text Generation(@EVA01)
- Text summarization (@haven-jeon,)
- [Korean-English Neural Machine Translater](https://github.com/haven-jeon/ko_en_neural_machine_translation) (@haven-jeon)

### high-performance learning

### Autoencoder
- [simple & conv. Autoencoder](autoencoder/Autoencoder_w_gluon.ipynb) (@ljy3795)
- Anomaly detection with Autoencoder (@ljy3795)

### Audio
- wavenet

### Recommandation
- [Deep Matrix Factorization](https://github.com/ski-net/dl_study_with_gluon/blob/master/Recommendation/180214_Deep_Matrix_Factorizaiton.ipynb) (@ljy3795)
- [Deep Matrix Factorization -- Fix InnerProduct Calculations](https://github.com/su-park/dl_study_with_gluon/blob/master/Recommendation/Deep%20Matrix%20Factorization%20using%20Gluon.ipynb) (@supark)

### XAI
- The Bayesian Case Model: A Generative Approach for Case-Based Reasoning and Prototype Classification (@kionkim)
  - https://arxiv.org/pdf/1503.01161.pdf
- Distilling a Neural Network Into a Soft Decision Tree (@kionkim)
  - https://arxiv.org/pdf/1711.09784.pdf


### GAN(Generative Adversarial Networks)
- [GAN](GAN/GAN_1D_Array.ipynb) (@hjkim)
  - https://arxiv.org/pdf/1406.2661.pdf
- [DCGAN](GAN) (@seujung)
  - https://arxiv.org/pdf/1511.06434.pdf
- DiscoGAN
  - https://arxiv.org/pdf/1703.05192.pdf
- WGAN
  - https://arxiv.org/pdf/1701.07875.pdf
- [BEGAN](GAN/BEGAN_Example_dim64_gluon.ipynb) (@seujung)
  - https://arxiv.org/pdf/1703.10717.pdf
- BiGAN (@hjkim)
  - https://arxiv.org/pdf/1605.09782.pdf

### VAE(Variational Auto Encoder)
- [Introduction to VAE](VAE/notebooks/VAE.ipynb)
- Tutorial on Variational Autoencoders (@kionkim)
  - https://arxiv.org/pdf/1606.05908.pdf

### Embedding
- [Word2Vec](Embedding/word2vec_skipgram_with_gluon.ipynb) (@hjkim, text8)
  - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.764.2227&rep=rep1&type=pdf

### Computer Age Statistical Inference(CASI)
- https://web.stanford.edu/~hastie/CASI

### QA
- relational network (@seujung)
  - https://arxiv.org/pdf/1706.01427.pdf
- [Visual Question Answering](QA/notebooks) (@kionkim, @hyemin15)

### Reinforcement Learning
- DQN(Deep Q-Network) (@hongdm7)
- DDQN(Double Deep Q-Network) (@soohwanjo)

## Reference

- https://github.com/zackchase/gluon-slides/blob/master/sept18-gluon.pdf
- https://github.com/zackchase/mxnet-the-straight-dope
- https://github.com/SherlockLiao/mxnet-gluon-tutorial
- https://github.com/gluon-api/gluon-api
- http://blog.creation.net/mxnet-part-1-ndarrays-api#.WjyR21SFiu7
