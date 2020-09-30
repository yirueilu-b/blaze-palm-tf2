# MediaPipe BlazePalm Model

An unofficial Implementation BlazePalm ( hand detector ) using Tensorflow 2.0.

## Usage

### Install dependencies

```
pip install -r requirements.txt
```

### Training

**Prepare training and validation data**

- Directory Structure

    In directory `dataset`, create `image` and `annotation` folders

    ```
    dataset
        |----train_image
        |----train_annotation
        |----val_image
        |----val_annotation
    ```
    
- Annotation Format

    Each image need a corresponding Json file that contains all hand items in the image.

    Each item in a json file should have 3 attributes `label`, `bounding_box` and `key_points`
    
    `label`: always be `1` (only one class)
    
    `bounding_box`: normalized 4 values, x_min, y_min, x_max, y_max
    
    `key_points`: 7 key points in total, [x1, y1, x2, y2,... x7, y7]
    
    ```
    [
      {
        "label": "1",
        "bounding_box": [
          0.739062488079071,
          0.3499999940395355,
          0.839062511920929,
          0.4833333194255829
        ],
        "key_points": [
          0.7484375,
          0.47708333333333336,
          0.8140625,
          0.35625,
          0.828125,
          0.39166666666666666,
          0.8359375,
          0.43125,
          0.8390625,
          0.4708333333333333,
          0.7421875,
          0.425,
          0.775,
          0.37083333333333335
        ]
      }
    ]
    ```

**Run `train.py`**

- Confirm the `Training Config` in `train.py` is correct then run.

### Inference

TODO

## Result

![](https://i.imgur.com/F9Wh4T9m.png)

![](https://i.imgur.com/40YP8OBm.png)

![](https://i.imgur.com/KKSmqXUm.png)

## TODO

- Refactor code

- Train a model on open dataset

- Decoder for prediction result

- Inference and visualize

## Reference

- [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214)
- [ssd_keras](https://github.com/pierluigiferrari/ssd_keras)
- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)

