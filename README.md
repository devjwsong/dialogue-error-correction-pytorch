# dialogue-error-correction-pytorch
This is a project for correcting augmented dialogue data using pre-trained models.

The models use the corrupted utterance and the turn label which is originally intended to be generated as inputs and create the corrected output utterance based on the label.

The simplified structure of this procedure is as follows.

<img src="https://user-images.githubusercontent.com/16731987/133450505-36002338-ce25-49d1-96fe-8df84cec2020.PNG" alt="The description of correcting augmented dialogue data." />

<br/>

---

### Models

The correction models are based on Transformer[[1]](#1) encoder-decoder structure.

There are two ways to set the encoder and the decoder.

<br/>

1. *BERT*[[2]](#2)(encoder) + *GRU*[[3]](#3)(decoder) with Copy mechanism[[4]](#4)
2. *BART*[[5]](#5)(encoder & decoder)

<br/>

The detailed information on each model can be found from the original papers attached in the reference section.

The BERT encoder is able to be replaced by *Tod-BERT*[[6]](#6) and *ConvBERT*[[7]](#7).

If you want to adopt *ConvBERT*, you have to download the pretrained checkpoint [here](https://dialoglue.s3.amazonaws.com/index.html#convbert).

Make a directory name `convbert` in the root and put all the downloaded files in it. 

<br/>

Additionally, the model is also trained for binary classification which detect whether the input should be corrected or not.

The class $0$ means that the utterance does not have to be fixed and class $1$ is vice versa.

<br/>

---

### Dataset

This project uses MultiWoZ 2.1[[7]](#7) for training and evaluation.

You should download the data from [here](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip), unzip it, and put all files into `./trade-dst/data`. (If you don't have the directory named `data` in `./trade-dst`, just create the folder first.)

Make sure to have the structure as below.

```
trade-dst
└--data
    └--MultiWOZ_2.1
       └--attraction_db.json
       └--data.json
       └--hospital_db.json
       └--...
       └--valListFile.txt
```

<br/>

---

### Arguments

**Arguments for data processing**

| Argument           | Type    | Description                                 | Default            |
| ------------------ | ------- | ------------------------------------------- | ------------------ |
| `seed`             | `int`   | The random seed.                            | `0`                |
| `multiwoz_dir`     | `str`   | The directory path for multiwoz data files. | `"trade-dst/data"` |
| `data_dir`         | `str`   | The directory path to save pickle files.    | `"data"`           |
| `slot_change_rate` | `float` | The ratio of changed slot part.             | `0.8`              |
| `cut_rate`         | `float` | The ratio of truncation.                    | `0.3`              |
| `max_window_size`  | `int`   | The maximum size of window when truncated.  | `2`                |

<br/>

**Arguments for training**

| Argument             | Type         | Description                                                  | Default              |
| -------------------- | ------------ | ------------------------------------------------------------ | -------------------- |
| `seed`               | `int`        | The random seed.                                             | `0`                  |
| `data_dir`           | `str`        | The directory path to save pickle files.                     | `"data"`             |
| `model_name`         | `str`        | The model to train & test. (`"bert"`, `"todbert"`, `"convbert"`, `"bart-base"`, `"bart-large"`) | *YOU SHOULD SPECIFY* |
| `num_epochs`         | `int`        | The number of total epochs.                                  | `5`                  |
| `train_batch_size`   | `int`        | The batch size for training.                                 | `16`                 |
| `eval_batch_size`    | `int`        | The batch size for inferencing.                              | `4`                  |
| `num_workers`        | `int`        | The number of workers for data loading.                      | `0`                  |
| `max_encoder_len`    | `int`        | The maximum length of a source sequence.                     | `512`                |
| `num_decoder_layers` | `int`        | The number of layers for the GRU decoder.                    | `2`                  |
| `decoder_dropout`    | `float`      | The dropout rate for the GRU decoder.                        | `0.2`                |
| `max_decoder_len`    | `int`        | The maximum length of a target sequence.                     | `256`                |
| `learning_rate`      | `float`      | The starting learning rate.                                  | `5e-5`               |
| `warmup_prop`        | `float`      | The warmup step proportion.                                  | `0.1`                |
| `max_grad_norm`      | `float`      | The max gradient for gradient clipping.                      | `1.0`                |
| `use_copy`           | `store_true` | Using copy or not, when using GRU decoder?                   | X                    |
| `mtl_factor`         | `float`      | The loss factor for multi-task learning.                     | `1.0`                |
| `loss_reduction`     | `str`        | How to reduce the LM loss value?                             | `"mean"`             |
| `beam_size`          | `int`        | The beam size for the beam search when inferencing.          | `4`                  |
| `num_samples`        | `int`        | The number of test samples to show.                          | `20`                 |
| `gpus`               | `nargs="+"`  | The indices of GPUs to use. (Multiple indices should be given separated in white space. ex: 0 1) | `["0"]`              |
| `num_nodes`          | `int`        | The number of machine.                                       | `1`                  |

<br/>

---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Parse the raw data. This process uses the modified version of parsing script in *TRADE*[[9]](#9) repository.

   ```shell
   cd trade-dst
   python create_data.py
   cd ..
   ```

   <br/>

3. Pre-process the parsed data.

   ```shell
   sh exec_data_process.sh
   ```

   <br/>

4. Train and evaluate the correction model.

   ```shell
   sh exec_main.sh
   ```

   <br/>

The trained checkpoint is saved as a form of Pytorch Lightning module.

In order to extract only model parameter to use in other Pytorch environment, run `src/extract_model.py`.

```shell
python src/extract_model.py --log_idx=LOG_IDX --ckpt_name=CKPT_NAME
```

- `log_idx`: This is the version index of saved log. This log files are saved in `./lightning_logs`.
- `ckpt_name`: This is the checkpoint name without the extension `.ckpt`. If you do not specify this argument, the lastest checkpoint saved in `./lightning_logs/version_{log_idx}/checkpoints` will automatically be chosen.

<br/>

---

### References

<a id="1">[1]</a> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998-6008). ([https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf))

<a id="2">[2]</a> Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*. ([https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf))

<a id="3">[3]</a> Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*. ([https://arxiv.org/pdf/1406.1078v3.pdf](https://arxiv.org/pdf/1406.1078v3.pdf))

<a id="4">[4]</a> See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. *arXiv preprint arXiv:1704.04368*. ([https://arxiv.org/pdf/1704.04368.pdf](https://arxiv.org/pdf/1406.1078v3.pdf))

<a id="5">[5]</a> Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019). Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *arXiv preprint arXiv:1910.13461*. ([https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf))

<a id="6">[6]</a> Wu, C. S., Hoi, S., Socher, R., & Xiong, C. (2020). TOD-BERT: pre-trained natural language understanding for task-oriented dialogue. *arXiv preprint arXiv:2004.06871*. ([https://arxiv.org/pdf/2004.06871.pdf](https://arxiv.org/pdf/2004.06871.pdf))

<a id="7">[7]</a> Mehri, S., Eric, M., & Hakkani-Tur, D. (2020). Dialoglue: A natural language understanding benchmark for task-oriented dialogue. *arXiv preprint arXiv:2009.13570*. ([https://arxiv.org/pdf/2009.13570.pdf](https://arxiv.org/pdf/2009.13570.pdf))

<a id="8">[8]</a> Eric, M., Goel, R., Paul, S., Kumar, A., Sethi, A., Ku, P., ... & Hakkani-Tur, D. (2019). MultiWOZ 2.1: A consolidated multi-domain dialogue dataset with state corrections and state tracking baselines. *arXiv preprint arXiv:1907.01669*. ([https://arxiv.org/pdf/1907.01669.pdf](https://arxiv.org/pdf/1907.01669.pdf))

<a id="9">[9]</a> Wu, C. S., Madotto, A., Hosseini-Asl, E., Xiong, C., Socher, R., & Fung, P. (2019). Transferable multi-domain state generator for task-oriented dialogue systems. *arXiv preprint arXiv:1905.08743*. ([https://arxiv.org/pdf/1905.08743.pdf](https://arxiv.org/pdf/1905.08743.pdf))
