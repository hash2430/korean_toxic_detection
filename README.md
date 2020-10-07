# Large BERT-base Multi-task learning for Korean hate/gender-bias/any-bias detection
This repository is for Korean Kaggles using multi-task approach.
We have achieved better performance with multi-task learning compared to
single-task learning.
Each script generates csv format Kaggle output for corresponding task.

# Train
```
python run_classifier.py
--vocab_file={vocab_path} --checkpoint={checkpoint_path} --config_file={config_path} --data_dir={train_data_path} --task_name kortd

```
* 'td' is a name that I made and it is short for 'toxic detection'

# Inference
## Korean gender bias detection
```
python eval_gender_bool.py
--vocab_file={vocab_path} --checkpoint={checkpoint_path} --config_file={config_path} --data_dir={test_data_path} --task_name kortd
``` 
## Korean hate speech detection
```
python eval_hate.py
--vocab_file={vocab_path} --checkpoint={checkpoint_path} --config_file={config_path} --data_dir={test_data_path} --data_dir=/mnt/sdd1/text/korean-hate-speech --task_name kortd
```

## Korean bias detection
```
python eval_bias.py
--vocab_file={vocab_path} --checkpoint={checkpoint_path} --config_file={config_path} --data_dir={test_data_path} --task_name kortd
```

# Result
|                       | Single-task | Multi-task |
|:---------------------:|:-----------:|:----------:|
| Gender bias detection |    68.13%   |   68.36%   |
| Hate speech detection |    52.54%   |   56.53%   |
|   Any bias detection  |    63.26%   |   65.57%   |
* 'Hate' is more coarse concept than 'gender bias detection' or 'any bias detection'
* Thus, it seems reasonable that 'hate detection' benefits the most from multi-task learning
* 'Any bias detection' is also more coarse task than 'gender bias detection.'
* The tendency of coarser task benefitting from finer-grained task is observed in this experiment, which is coherent with recent studies.
* Limitation
    * The pretrained model is trained on literal style dataset (Korean wikipedia, newspaper) while test data is colloquial and obtained from Naver news comments.
    * This domain mis-match restricts the upper bound of this experiment.
    * Simply changing to different pretrained model that is pretrained on colloquial trainingset gives much higher performance
    * Hate detection performance goes up to 60% accuracy, simply by replacing the pretrained model.
    


