# ERNIE3.0

## Description

This model is fine-tuned on [**ERNIE 3.0-_Medium_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams) using [this dataset](https://paddlenlp.bj.bcebos.com/datasets/divorce.tar.gz), and is used for multi-label tasks.

## Model

| Model                   | Download                          | Micro F1(%)   | Macro F1(%)    |
|:------------------------|:----------------------------------|:--------------|:---------------|
|ernie-3.0-medium-zh      |[424 M](ernie-3.0-medium-zh.onnx)  |90.57          |79.36           |

## Dataset

[CAIL2019—Marriage and Family Element Extraction Task Dataset](https://paddlenlp.bj.bcebos.com/datasets/divorce.tar.gz)

## References

* [ERNIE3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)
* [ERNIE3.0 for Multi-label Tasks](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label)

## License

Apache License 2.0
