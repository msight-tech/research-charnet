# Convolutional Character Networks

This project hosts the testing code for CharNet, described in our paper:

    Convolutional Character Networks
    Linjie Xing, Zhi Tian, Weilin Huang, and Matthew R. Scott;
    In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

   
## Updates

1. Added Visualization
2. icdar_2015 Test Dataset


## Installation

```
pip install torch torchvision
python setup.py build develop
```


## Dataset
Download icdar_2015 Test Dataset [here](https://drive.google.com/file/d/1mPRmedDdC9UPlNQAEohMCMWlnTRoiqfC/view?usp=share_link)


## Run
1. Please run `bash download_weights.sh` to download our trained weights. 
2. For ICDAR 2015, please run the following command line. Please replace `images_dir` with the directory containing ICDAR 2015 testing images. The results will be in `results_dir`.

    ```
    python tools/test_net.py configs/icdar2015_hourglass88.yaml <images_dir> <results_dir>
    ```

## Results
![](https://github.com/hamzakhalil798/research-charnet/blob/master/images/result_1.jpg)
![](https://github.com/hamzakhalil798/research-charnet/blob/master/images/result_2.jpg)


## Citation

If you find this work useful for your research, please cite as:

    @inproceedings{xing2019charnet,
    title={Convolutional Character Networks},
    author={Xing, Linjie and Tian, Zhi and Huang, Weilin and Scott, Matthew R},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
    }
    
## Contact

For any questions, please feel free to reach: 
```
github@malongtech.com
```


## License

CharNet is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. It is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact sales@malongtech.com.
