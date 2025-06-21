# Text-guided Class-Incremental LiDAR Semantic Segmentation with Category Distribution Constraint


Our codebase is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## Installation
```
conda env create -f clpcss.yml

pip install -r requirements.txt
```

#### Training

- **init text token**
    ```
    CUDA_VISIBLE_DEVICES=0 python ./model/word/word_embedding.py
    ```

- **step 0**:
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --CL True --setup "Sequential" --CLstep 0 --test_name "tcil_step_0"
    ```

- **TCIL step 1**: 
    ```
    bash ./scripts/tcil_step_1.sh    
    ```
  
- **TCIL step 2**: 
    ```
    bash ./scripts/tcil_step_2.sh
    ```

- **TCIL test**
   ```
   bash ./scripts/tcil_test.sh
   ```

### Acknowledgement
This project is based on [CL-PCSS](https://github.com/LTTM/CL-PCSS). Thanks for their wonderful work.
