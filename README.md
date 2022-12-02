# A Novel Occlusion-aware Vote Cost for Light Field Depth Estimation
### [Project Page](https://imkanghan.github.io/projects/OAVC/main) | [Paper](https://ieeexplore.ieee.org/document/9517020)

This repository is an implementation of light field depth estimation described in the paper "A Novel Occlusion-aware Vote Cost for Light Field Depth Estimation". 

[Kang Han](https://imkanghan.github.io/)<sup>1</sup>, [Wei Xiang](https://scholars.latrobe.edu.au/wxiang)<sup>2</sup>, [Eric Wang](https://research.jcu.edu.au/portfolio/eric.wang/)<sup>1</sup>, [Tao Huang](https://www.taoicclab.com/)<sup>1</sup>

<sup>1</sup>James Cook University, <sup>2</sup>La Trobe University

### Abstract
Conventional light field depth estimation methods build a cost volume that measures the photo-consistency of pixels refocused to a range of depths, and the highest consistency indicates the correct depth. This strategy works well in most regions but usually generates blurry edges in the estimated depth map due to occlusions. Recent work shows that integrating occlusion models to light field depth estimation can largely reduce blurry edges. However, existing occlusion handling methods rely on complex edge-aided processing and post-refinement, and this reliance limits the resultant depth accuracy and impacts on the computational performance. In this paper, we propose a novel occlusion-aware vote cost (OAVC) which is able to accurately preserve edges in the depth map. Instead of using photo-consistency as an indicator of the correct depth, we construct a novel cost from a new perspective that counts the number of refocused pixels whose deviations from the central-view pixel are less than a small threshold, and utilizes that number to select the correct depth. The pixels from occluders are thus excluded in determining the correct depth. Without the use of any explicit occlusion handling methods, the proposed method can inherently preserve edges and produces high-quality depth estimates. Experimental results show that the proposed OAVC outperforms state-of-the-art light field depth estimation methods in terms of depth estimation accuracy and the computational performance.


### Setup
Python 3 dependencies:

- OpenCV
- Numpy
- Joblib
- Multiprocessing

### Dataset

Download datasets from 4D Light Field Benchmark or Inria synthetic light field datasets.


### Running code

Change the file path in main.py, and run

```
python main.py
```

### Citation
If you find this code useful in your research, please cite:

    @ARTICLE{9517020,
        author={Han, Kang and Xiang, Wei and Wang, Eric and Huang, Tao},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
        title={A Novel Occlusion-Aware Vote Cost for Light Field Depth Estimation}, 
        year={2022},
        volume={44},
        number={11},
        pages={8022-8035},
        doi={10.1109/TPAMI.2021.3105523}
    }
