# SMPVideo

**All environment setup steps are included in the shell script, so there is no need to explicitly show environment installation instructions.**
**merge_result.csv is the submission file**

**Environment Requirements:**  
- GPU: 3090 24GB cu118  

---
All code executions are performed within the `SMPVideo` folder:

~~~
SMPVideo/
----MMRA/
----bash/
----machine_learning/
----requirements/
----xxxx
~~~

## 1. Quickly Obtain Competition Results (Integrated Testing Script)

You can quickly reproduce the competition results by running the integrated testing script.  
We provide all the weight files and the feature files directly related to training and testing.

Except for the final_easy_excel folder(size too large), all other files are already available in the GitHub repository. However, to prevent them from being overwritten during subsequent training, we also provide Baidu Cloud links as a backup.

| File Name                             | Place to Put the File   | Baidu Cloud Link                                                     | File about|
|---------------------------------------|-------------------------|----------------------------------------------------------------------|-------------------------------|
| final_easy_excel           |   MMRA/datasets/tiktok  | [Download](https://pan.baidu.com/s/1RYzMuZDwv8A2UCTbmoX-PQ?pwd=3be8) |  train+test features  |
| train_MMRA_tiktok_MSE |   MMRA/train_result     | [Download](https://pan.baidu.com/s/1IIp3CoPK5zVGASqwRDAguw?pwd=cxm6) | train.pkl (with train.log) |

~~~
SMPVideo/
├── MMRA/
│ ├── datasets/
│ │ ├── tiktok/
│ │ │ ├── final_easy_excel/
│ ├── train_result/
│ │ ├── train_MMRA_tiktok_MSE/
~~~

| File Name                             | Place to Put the File   | Baidu Cloud Link                                                     | File about|
|---------------------------------------|-------------------------|----------------------------------------------------------------------|-------------------------------|
| machine_learning       |  machine_learning/machine_learning  | [Download](https://pan.baidu.com/s/12hD0CSYQaD8Eq_WqLdk4PA?pwd=am6v) |  catboost+lightbgm+mlp best params.json  |
| merge_test_data.csv |   machine_learning/output     | [Download](https://pan.baidu.com/s/1vpFccibCB5FWTp5-sH1IiA?pwd=p428) | test features |

~~~
SMPVideo/
├── machine_learning/
│ ├── machine_learning/
│ ├── output/
│ │ ├── merge_test_data.csv
~~~

Execute the following command:  
`bash ./bash/test/run_all.sh`

In theory, using the same environment, identical inputs, and the provided weights, you should be able to reproduce the results submitted during the competition.  
If for any reason the results are not consistent with those in the competition, we kindly ask the competition organizers to notify us via email. 

## 2. Easy to Reproduce Training

Because repair scripts and intermediate feature generation take a long time, we provide corresponding intermediate files for rapid reproduction.

After training, the weight files will be overwritten, so we recommend using the integrated testing script above for quick validation.

In theory, all of these files are provided in our GitHub repository, except for the train and test files under MMRA (as they are too large).

| File Name                  | Place to Put the File   | Baidu Cloud Link                                   | File about |
|----------------------------|------------------------|---------------------------------------------------|-----|
| kfold | machine_learning/kfold      | [Download](https://pan.baidu.com/s/12eTXfM03S8FsFg7cA9d7Yg?pwd=qk6w)   | catboost+mlp use |
| kfold_time  | machine_learning/kfold_time      | [Download](https://pan.baidu.com/s/1OOhgPKWT1JcY2k8m2WLHtg?pwd=x81j)        | lightbgm use |
| final_easy_excel           |   MMRA/datasets/tiktok  | [Download](https://pan.baidu.com/s/1RYzMuZDwv8A2UCTbmoX-PQ?pwd=3be8) |  train+test features  |

---

The tests revealed that LightGBM is highly related to time, while CatBoost + MLP is more related to the volume of data. Therefore, the data was divided as mentioned above.

Then Execute the following command:  
`bash ./bash/Easy_train/run_all.sh`

If you want to run a test, 
simply execute `bash ./bash/test/run_all.sh`


**Note:** : After a simple training session, the original weight files will be overwritten.

Moreover, due to the randomness introduced by multithreading, we observed that even with fixed seeds, the training processes of CatBoost, XGBoost, and MLP may still yield slightly different results. In contrast, MMRA's training process does not utilize multithreading, allowing it to consistently produce the same output given identical input.

We believe these minor variations have no significant impact on the final outcome. In theory, repeating the experiment multiple times should eventually reproduce the competition results.

## 3. Complete to Reproduce Training

```markdown
⚠️ **This step is NOT recommended to run**:

The overall feature extraction process is time-consuming. Although we provide the `FFmpeg`-repaired videos for download, the entire pipeline still takes significantly more time compared to the `easy_train` steps. Additionally, due to the use of multi-threading during feature extraction, the randomness in the results cannot be completely eliminated.

We recommend following **Easy to Reproduce Training**, which is faster and introduces slightly less randomness.
```

1. Download the official dataset from [video_file.zip](https://drive.google.com/drive/folders/1F37YsuZPngqTIDTe-I_yoSFEvLSoVLnt).
2. Unzip the downloaded file.
3. Rename the extracted folder to `Video`.
4. Move the `Video` folder to the `dataset/` directory .

~~~
SMPVideo/
├── dataset/
│ ├── Video/
│ │ ├── train/
│ │ ├── test/
~~~

Then execute the following command:  
`bash ./bash/Complete_train/run_all.sh`

This project uses ffmpeg with libx264 encoding to repair videos, which is time-consuming (about 3+ days).  
To save time, we provide repaired videos via Baidu Netdisk:

1. Download [repair.zip](https://pan.baidu.com/s/1TcGMh-6MQAPLpi2LKiUoSw?pwd=k8jp) and [repair_1.zip](https://pan.baidu.com/s/19EjBwQRUIalk0KkYWQ4xDQ?pwd=3n87) from the provided Baidu Netdisk link.
2. Extract both files.
3. Rename the extracted folder to `repair_video`.
4. Move the `repair_video` folder into the `dataset` directory.
5. When asked, allow the system to overwrite any existing files.
~~~
SMPVideo/
├── dataset/
│ ├── repair_video/
│ │ ├── train/
│ │ ├── test/
~~~

Then execute the following command:  
`bash ./bash/Complete_train/run_all.sh --skip repair`

Finally, if you want to run a test, 
simply execute `bash ./bash/test/run_all.sh`

## External Code Dependencies

All required environment dependencies and Python packages are installed automatically by our one-click setup script—no manual intervention needed.

### Included Third-Party Code

- **MMRA** (Predicting Micro-video Popularity via Multi-modal Retrieval Augmentation)  
  We have integrated the MMRA implementation directly in this repository under `MMRA/`.  

  Original source: [https://github.com/ICDM-UESTC/MMRA](https://github.com/ICDM-UESTC/MMRA)  
