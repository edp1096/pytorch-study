
```powershell
pip install matplotlib ; pip install numpy ; pip install pandas ; pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Lecture, Tutorial
* https://tutorials.pytorch.kr/beginner/basics/intro.html
    * https://pytorch.org/tutorials/beginner/basics/intro.html
* https://atcold.github.io/pytorch-Deep-Learning/ko
* https://www.youtube.com/playlist?list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv
    * https://deeplearningzerotoall.github.io/season2/lec_pytorch.html
* https://wikidocs.net/book/2788
* https://github.com/Harry24k/Pytorch-Basic
* https://github.com/mrdbourke/pytorch-deep-learning

* https://github.com/teddylee777/machine-learning
* https://fleuret.org/dlc

## Fashion-MNIST

t10k = Test set 10,000

https://github.com/zalandoresearch/fashion-mnist

Each training and test example is assigned to one of the following labels:

| Label | Description |
| ---   | ---         |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |


## MNIST

* Source:
    * https://github.com/teavanist/MNIST-JPG
        * https://wikidocs.net/60324
        * https://wikidocs.net/63565
        * https://tensorflow.blog/2017/01/26/pytorch-mnist-example
        * https://deep-learning-study.tistory.com/459
        * https://github.com/pytorch/examples/tree/main/mnist
        * https://uding.tistory.com/44
        * https://korchris.github.io/2019/08/23/mnist
        * https://discuss.pytorch.org/t/how-to-save-wrong-prediction-results-of-a-cnn-on-mnist/66576
        * https://github.com/msiddalingaiah/MachineLearning/tree/master/MNIST
        * https://github.com/krishna-tx/mnist-pytorch
            * https://www.youtube.com/watch?v=ijaT8HuCtIY
        * https://junstar92.tistory.com/116
        * https://ninejy.tistory.com/26
    * https://www.kaggle.com/code/kkaiwwang/99-67-no-cheating-multi-simple-cnn-model
        * https://paperswithcode.com/paper/an-ensemble-of-simple-convolutional-neural
    * https://ddtxrx.tistory.com/entry/PyTorch-MNIST-1


## Bees or ants
* Source:
    * https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html
    * https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    * https://blogofth-lee.tistory.com/268


## Movie posters - Custom image dataset

* Source:
    * https://androidkt.com/load-custom-image-datasets-into-pytorch-dataloader-without-using-imagefolder
    * https://www.kaggle.com/datasets/raman77768/movie-classifier
    * https://github.com/raman77768/Movie-Genre-Classification
    * https://youtu.be/ACmuBbuXn20
        * https://intrepidgeeks.com/tutorial/using-vgg16-to-realize-transfer-learning-with-pytorch
        * https://deep-learning-study.tistory.com/521
        * https://dacon.io/codeshare/2354
    * https://youtu.be/is_Vw-aJMg4
    * https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203
    * https://stackoverflow.com/questions/52855843/multi-label-classification-in-pytorch
    * https://ko.taphoamini.com/multi-label-classification-yeje


## Cats vs Dogs

* Source:
    * https://junstar92.tistory.com/121
        * https://blog.jovian.ai/image-classification-between-dog-and-cat-using-resnet-in-pytorch-fdd9fdfac20a
        * https://www.microsoft.com/en-us/download/details.aspx?id=54765
        * https://nuguziii.github.io/dev/dev-002
        * https://inside-machinelearning.com/en/easy-tutorial-object-detection-on-image-in-pytorch-part-2/


## Todo

* CRNN, CTC, RNN/LSTM/GRU
    * https://github.com/GitYCC/crnn-pytorch
    * https://github.com/PAN001/MNIST-digit-sequence-recognition
    * https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
    * https://github.com/dredwardhyde/crnn-ctc-loss-pytorch
        * https://medium.com/swlh/multi-digit-sequence-recognition-with-crnn-and-ctc-loss-using-pytorch-framework-269a7aca2a6
    * https://github.com/srihari-humbarwadi/crnn_ocr_keras
    * https://velog.io/@cha-suyeon/%EB%94%A5%EB%9F%AC%EB%8B%9D-OCR3-
    * https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/#step-1-loading-mnist-train-dataset
    * https://medium.com/@nutanbhogendrasharma/pytorch-recurrent-neural-networks-with-mnist-dataset-2195033b540f
    * https://velog.io/@sjinu/Pytorch-Implementation-code
    * https://coding-yoon.tistory.com/131
    * https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial

* Sliding window
    * https://github.com/rustagiadi95/OCR-sliding_window-pytorch

* Detectron2
    * https://github.com/facebookresearch/detectron2

* YOLO
    * https://foss4g.tistory.com/1646
    * https://velog.io/@minkyu4506/PyTorch%EB%A1%9C-YOLOv1-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0
    * https://velog.io/@skhim520/YOLO-v1-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-%EB%B0%8F-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84
    * https://wandb.ai/wandb_fc/korean/reports/Windows-YOLOv5---VmlldzoxODc3MjE1
    * https://wandb.ai/wandb_fc/korean/reports/PyTorch-YOLOv5---VmlldzoxODc3Mzk5
    * https://leedakyeong.tistory.com/entry/Object-Detection-YOLO-v1v6-%EB%B9%84%EA%B5%902
    * https://arxiv.org/abs/2207.02696
    * https://github.com/PaddlePaddle/PaddleDetection

* SSD - Single Shot MultiBox Detector, VOC, R-CNN
    * https://towardsdatascience.com/learning-note-single-shot-multibox-detector-with-pytorch-part-1-38185e84bd79
    * https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
        * https://github.com/dldldlfma/pytorch_tutorial_ssd
    * https://github.com/amdegroot/ssd.pytorch
    * https://taeu.github.io/paper/deeplearning-paper-ssd
    * https://herbwood.tistory.com/15
    * https://www.kaggle.com/code/thongnon1996/object-detection-mnist-ssd-pytorch-from-scratch
    * https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    * https://deep-learning-study.tistory.com/612

* Transformer
    * https://github.com/huggingface/transformers
    * https://github.com/moyix/fauxpilot
    * https://github.com/microsoft/CodeBERT
        * https://ebbnflow.tistory.com/151
        * https://tutorials.pytorch.kr/intermediate/dynamic_quantization_bert_tutorial.html
        * https://velog.io/@lgd1820/pytorch%EB%A1%9C-BERT-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B013
        * https://inhyeokyoo.github.io/project/nlp/bert-issue

* ViT
    * https://github.com/jeonsworld/ViT-pytorch
    * https://blog.kubwa.co.kr/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-a-convnet-for-the-2020s-9b45ac666d04

* Dataset
    * MNIST - http://yann.lecun.com/exdb/mnist
    * EMNIST - https://www.nist.gov/itl/products-and-services/emnist-dataset
    * TextOCR - https://textvqa.org/textocr/dataset
    * Text Recognition Data - https://www.robots.ox.ac.uk/~vgg/data/text/
    * CIFAR - https://www.cs.toronto.edu/~kriz/cifar.html
        * Mirror - http://data.brainchip.com/dataset-mirror
    * Imagenet - Kaggle
    * Tiny-Imagenet200 - http://cs231n.stanford.edu/tiny-imagenet-200.zip
    * COCO - https://cocodataset.org/
        * https://github.com/cocodataset/cocoapi

* Etc.
    * https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html
        * https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    * https://www.kaggle.com/code/famadio/ocr-with-pytorch/notebook
    * https://github.com/facebookresearch/faiss
    * https://codetorial.net/tensorflow/index.html
    * https://github.com/albumentations-team/albumentations
    * https://deview.kr/data/deview/2019/presentation/[115]%EC%96%B4%EB%94%94%EA%B9%8C%EC%A7%80+%EA%B9%8E%EC%95%84%EB%B4%A4%EB%8B%88_%EB%AA%A8%EB%B0%94%EC%9D%BC+%EC%84%9C%EB%B9%84%EC%8A%A4%EB%A5%BC+%EC%9C%84%ED%95%9C+%EA%B0%80%EB%B2%BC%EC%9A%B4+%EC%9D%B4%EB%AF%B8%EC%A7%80+%EC%9D%B8%EC%8B%9D_%EA%B2%80%EC%B6%9C+%EB%94%A5%EB%9F%AC%EB%8B%9D+%EB%AA%A8%EB%8D%B8.pdf
    * https://github.com/YuvalNirkin/fsgan
    * https://arxiv.org/pdf/1810.00736.pdf
    * https://velog.io/@0ju-un/PyTorch%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-FPN%EC%9C%BC%EB%A1%9C-%ED%95%9C%EA%B8%80-%EC%86%90%EA%B8%80%EC%94%A8%EC%97%90%EC%84%9C-%EC%9E%90%EB%AA%A8-%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0-dataset-%EC%A0%9C%EC%9E%91%EB%B6%80%ED%84%B0-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5-%EC%98%88%EC%B8%A1%EA%B9%8C%EC%A7%80
    * https://cvml.tistory.com/22
    * https://github.com/fchollet/deep-learning-models
    * https://www.aihub.or.kr/
    * https://keep-steady.tistory.com/36
