import os
from typing import List, Union
import pickle
import random
import argparse
import torch
from torch.utils.data import IterableDataset
from models.modeling_bert import Model, Config, MlmNspModel
### You can import any Python standard libraries or pyTorch sub directories here
from torch.utils.data import DataLoader
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
### END YOUR LIBRARIES

import utils_pretrain
from utils.tokenization import BertTokenizer
from bpe import BytePairEncoding
from model import MLMandNSPmodel
from data import ParagraphDataset
print(torch.cuda.is_available())
# You can use tqdm to check your progress
from tqdm import tqdm, trange

# TODO: 2 data path designation
class PretrainDataset(IterableDataset):
    def __init__(self, max_seq_len, comment_dataset: ParagraphDataset, title_dataset: ParagraphDataset, tokenizer: BertTokenizer):
        """ Maked Language Modeling & Next Sentence Prediction dataset initializer
        Use below attributes when implementing the dataset

        Attributes:
        dataset -- Paragraph dataset to make a MLM & NSP sample
        """
        self.comment_dataset = comment_dataset
        self.title_dataset = title_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len



    @property
    def token_num(self):
        return self.comment_dataset.token_num

    # TODO: nsp pair sample/example create/feature create(=tokenize)
    def __iter__(self):
        """ Masked Language Modeling & Next Sentence Prediction dataset
        Sample two sentences from the dataset, and make a self-supervised pretraining sample for MLM & NSP

        Note: You can use any sampling method you know.

        Yields:
        source_sentences -- Sampled sentences
        MLM_sentences -- Masked sentences
        MLM_mask -- Masking for MLM
        NSP_label -- NSP label which indicates whether the sentences is connected.

        Example: If 25% mask with 50 % <msk> + 25% random + 25% same -- this percentage is just a example.
        source_sentences = ['<cls>', 'He', 'bought', 'a', 'gallon', 'of', 'milk',
                            '<sep>', 'He', 'drank', 'it', 'all', 'on', 'the', 'spot', '<sep>']
        MLM_sentences = ['<cls>', 'He', '<msk>', 'a', 'gallon', 'of, 'milk',
                         '<sep>', 'He', 'drank', 'it', 'tree', 'on', '<msk>', 'spot', '<sep>']
        MLM_mask = [False, False, True, False, False, False, False,
                    False, True, False, False, True, False True, False, False]
        NSP_label = True
        """
        # Special tokens
        CLS = 2
        SEP = 3
        MSK = 4

        # The number of tokens
        TOKEN_NUM = self.token_num

        while True:
            ### YOUR CODE HERE (~ 40 lines)
            source_sentences: List[int] = None
            MLM_sentences: List[int] = None
            MLM_mask: List[bool] = None
            NSP_label: bool = None
            source_sentences = []
            MLM_mask = []

            # SAMPLE NSP PAIR
            nsp = torch.multinomial(torch.FloatTensor([0.5, 0.5]), 1)

            index = random.randint(0, len(self.comment_dataset)-1)
            if nsp == torch.LongTensor([1]):
                NSP_label = True
                sentence1 = self.comment_dataset[index]
                sentence2 = self.title_dataset[index]
            else:
                NSP_label = False
                title_index = random.randint(0, len(self.comment_dataset)-1)
                while index == title_index:
                    title_index = random.randint(0, len(self.comment_dataset) - 1)
                sentence1 = self.comment_dataset[index]
                sentence2 = self.title_dataset[title_index]
            prgr = [sentence1, sentence2]

            # GENERATE EXAMPLE, TOKENIZE, MASK
            i = 0
            src_sentence = ['[CLS]']
            for sentence in prgr:
                src_sentence += self.tokenizer.tokenize(sentence)
                i += len(sentence)
                src_sentence += ['[SEP]']
            src_sentence = self.tokenizer.convert_tokens_to_ids(src_sentence)
            if len(src_sentence) > self.max_seq_len:
                src_sentence = src_sentence[:self.max_seq_len-1]
                src_sentence.append(SEP)
            uniform_prob = 1.0/float(i)
            mask = torch.FloatTensor(src_sentence) == SEP
            mask2 = torch.FloatTensor(src_sentence) == CLS
            mask = (mask + mask2)
            distribution = torch.FloatTensor(src_sentence)
            distribution[mask] = 0.
            distribution[~mask] = uniform_prob
            sample_num = int(len(src_sentence)*0.15)
            try:
                sampled_indices = torch.multinomial(distribution, sample_num, False)
                n_mask = int(float(sample_num) * 0.8)
                n_do_nothing = int(float(sample_num) * 0.1)
                mask_indices = sampled_indices[:n_mask]
                random_transform_indices = sampled_indices[n_mask + n_do_nothing:]
                random_tokens = random.sample(range(8, len(self.tokenizer.vocab)), len(random_transform_indices))
            except: #Exception occurs when the length of src_sentence is too short. If that's the case, I am masking one token.
                sampled_indices = [random.randint(1, len(src_sentence) - 2)]
                mask_indices = sampled_indices
                random_transform_indices = []
                random_tokens = []
            finally:
                mlm_mask = torch.BoolTensor(len(src_sentence)).fill_(False)
                for index in sampled_indices:
                    mlm_mask[index] = True
                mlm_sentence = src_sentence
                mlm_sentence = torch.IntTensor(mlm_sentence)

                mlm_sentence[mask_indices] = MSK
                for idx, random_token in zip(random_transform_indices, random_tokens):
                    while mlm_sentence[idx] == random_token:
                        random_token = random.randint(8, len(self.tokenizer.vocab))
                    mlm_sentence[idx] = random_token



            source_sentences = src_sentence
            MLM_mask = mlm_mask.tolist()
            MLM_sentences = mlm_sentence.tolist()

            assert len(source_sentences) == len(MLM_sentences) == len(MLM_mask)
            yield source_sentences, MLM_sentences, MLM_mask, NSP_label

def calculate_losses(
    model: MlmNspModel,
    source_sentences: torch.Tensor,
    MLM_sentences: torch.Tensor,
    MLM_mask: torch.Tensor,
    NSP_label: torch.Tensor
):
    """ MLM & NSP losses calculation
    Use cross entropy loss to calculate both MLN and NSP losses.
    MLM loss should be an average loss of masked tokens.

    Arguments:
    model -- MLM & NSP model
    source_sentences -- Source sentences tensor in torch.long type
                        in shape (sequence_length, batch_size)
    MLM_sentences -- Masked sentences tensor in torch.long type
                        in shape (sequence_length, batch_size)
    MLM_mask -- MLM mask tensor in torch.bool type
                        in shape (sequence_length, batch_size)
    NSP_label -- NSP label tensor in torch.bool type
                        in shape (batch_size, )

    Returns:
    MLM_loss -- MLM loss in scala tensor
    NSP_loss -- NSP loss in scala tensor
    """
    ### YOUR CODE HERE (~4 lines)
    MLM_loss: torch.Tensor = None
    NSP_loss: torch.Tensor = None
    if isinstance(MLM_mask, List):
        MLM_mask = torch.stack(MLM_mask)
        MLM_sentences = torch.stack(MLM_sentences)
        source_sentences = torch.stack(source_sentences)
    masked_indices = MLM_mask.flatten().nonzero()
    masked_indices = masked_indices.squeeze(1)

    # MLM
    mlm_out, nsp_out = model(MLM_sentences, None)
    predicted = mlm_out.view(mlm_out.shape[0]*mlm_out.shape[1],-1).index_select(0, masked_indices)
    gt = source_sentences.flatten().index_select(0, masked_indices)

    ce_MLM = torch.nn.CrossEntropyLoss()
    #TODO: convert predicted to e
    MLM_loss = ce_MLM(predicted, gt)

    # NSP
    ce_NSP = torch.nn.CrossEntropyLoss()
    NSP_label.contiguous()
    nsp_label = torch.zeros(NSP_label.shape[0], dtype=torch.int64)
    for i in range(NSP_label.shape[0]):
        if NSP_label[i]:
            nsp_label[i]=torch.LongTensor([1])
    NSP_loss = ce_NSP(nsp_out, nsp_label.squeeze(1))


    ### END YOUR CODE
    assert MLM_loss.shape == NSP_loss.shape == torch.Size()
    return MLM_loss, NSP_loss

def pretraining(
    model: MLMandNSPmodel,
    model_name: str,
    train_dataset: PretrainDataset,
    val_dataset: PretrainDataset,
):
    """ MLM and NSP pretrainer
    Implement MLN & NSP pretrainer with the given model and datasets.

    Note 1: Don't forget setting model.train() and model.eval() before training / validation.
            It enables / disables the dropout layers of our model.

    Note 2: There are useful tools for your implementation in utils.py

    Note 3: Training takes almost 3 minutes per a epoch on TITAN RTX. Thus, 200 epochs takes 10 hours.
    For those who don't want to wait 10 hours, we attaches a model which has trained over 200 epochs.
    You can use it on the IMDB training later.

    Memory tip 1: If you delete the loss tensor explictly after every loss calculation like "del loss",
                  tensors are garbage-collected before next loss calculation so you can cut memory usage.

    Memory tip 2: If you use torch.no_grad when inferencing the model for validation,
                  you can save memory space of gradient. 

    Memory tip 3: If you want to keep batch_size while reducing memory usage,
                  creating a virtual batch is a good solution.
    Explanation: https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672

    Useful readings: https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/ 

    Arguments:
    model -- Pretraining model which need to be trained
    model_name -- The model name. You can use this name to save your model
    train_dataset -- Pretraining dataset for training
    val_dataset -- Pretraining dataset for validation

    Variables:
    batch_size -- Batch size
    learning_rate -- Learning rate for the optimizer
    epochs -- The number of epochs
    steps_per_a_epoch -- The number of steps in a epoch.
                        Because there is no end in IterableDataset, you have to set the end of epoch explicitly.
    steps_for_val -- The number of steps for validation

    Returns:
    MLM_train_losses -- List of average MLM training loss per a epoch
    MLM_val_losses -- List of average MLM validation loss per a epoch
    NSP_train_losses -- List of average NSP training loss per a epoch
    NSP_val_losses -- List of average NSP validation loss per a epoch
    """
    # Below options are just our recommendation. You can choose different options if you want.
    batch_size = 16
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 200 # 200 if you want to feel the effect of pretraining
    steps_per_a_epoch: int=2000
    steps_for_val: int=200
    # epochs = 7
    # steps_per_a_epoch = 1
    # steps_for_val = 1

    ### YOUR CODE HERE
    MLM_train_losses: List[float] = None
    MLM_val_losses: List[float] = None
    NSP_train_losses: List[float] = None
    NSP_val_losses: List[float] = None
    MLM_train_losses = []
    MLM_val_losses = []
    NSP_train_losses = []
    NSP_val_losses = []
    train_data_iterator = iter(
        torch.utils.data.dataloader.DataLoader(train_dataset, collate_fn=utils_pretrain.pretrain_collate_fn, batch_size=batch_size, num_workers=16, shuffle=False))
    eval_data_iterator = iter(
        torch.utils.data.dataloader.DataLoader(val_dataset, collate_fn=utils_pretrain.pretrain_collate_fn, batch_size=batch_size, num_workers=16, shuffle=False))

    loss_log = tqdm(total=0, bar_format='{desc}')
    i = 0
    for epoch in trange(epochs, desc="Epoch", position=0):
        i += 1
        # Run batches for 'steps_per_a_epoch' times
        MLM_loss = 0
        NSP_loss = 0
        model.train()
        device = 'cpu'
        for step in trange(steps_per_a_epoch, desc="Training steps"):
            optimizer.zero_grad()
            src, mlm, mask, nsp = next(train_data_iterator)
            src = src.to(torch.device(device))
            mlm = mlm.to(torch.device(device))
            mask = mask.to(torch.device(device))
            nsp = nsp.to(torch.device(device))
            mlm_loss, nsp_loss = calculate_losses(model, src, mlm, mask, nsp)
            MLM_loss += mlm_loss
            NSP_loss += nsp_loss
            loss = mlm_loss + nsp_loss
            loss.backward()
            optimizer.step()
            des = 'Loss: {:06.4f}'.format(loss.cpu())
            loss_log.set_description_str(des)

        # Calculate training loss
        MLM_loss = MLM_loss / steps_per_a_epoch
        NSP_loss = NSP_loss / steps_per_a_epoch
        MLM_train_losses.append(float(MLM_loss.data))
        NSP_train_losses.append(float(NSP_loss.data))

        # Calculate valid loss
        model.eval()
        valid_mlm_loss = 0.
        valid_nsp_loss = 0.

        for step in trange(steps_for_val, desc="Evaluation steps"):
            src, mlm, mask, nsp = next(eval_data_iterator)
            mlm_loss, nsp_loss = calculate_losses(model, src, mlm, mask, nsp)
            valid_mlm_loss += mlm_loss
            valid_nsp_loss += nsp_loss

        valid_mlm_loss = valid_mlm_loss / steps_for_val
        valid_nsp_loss = valid_nsp_loss / steps_for_val

        MLM_val_losses.append(float(valid_mlm_loss.data))
        NSP_val_losses.append(float(valid_nsp_loss.data))
        torch.save(model.state_dict(), os.path.join('/data2/projects/AI605/hw3/models',model_name + str(i)+'.pth'))

    ### END YOUR CODE

    assert len(MLM_train_losses) == len(MLM_val_losses) == epochs and \
           len(NSP_train_losses) == len(NSP_val_losses) == epochs

    assert all(isinstance(loss, float) for loss in MLM_train_losses) and \
           all(isinstance(loss, float) for loss in MLM_val_losses) and \
           all(isinstance(loss, float) for loss in NSP_train_losses) and \
           all(isinstance(loss, float) for loss in NSP_val_losses)

    return MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses

##############################################################
# Testing functions below.                                   #
#                                                            #
# We only checks MLM & NSP dataset and loss calculation.     #
# We do not tightly check the correctness of your trainer.   #
# You should attach the loss plot to the report              #
# and submit the pretrained model to validate your trainer.  #
# We will grade the score by running your saved model.       #
##############################################################

def MLM_and_NSP_dataset_test():
    print("======MLM & NSP Dataset Test Case======")
    CLS = BytePairEncoding.CLS_token_idx
    SEP = BytePairEncoding.SEP_token_idx
    MSK = BytePairEncoding.MSK_token_idx

    class Dummy(object):
        def __init__(self):
            self.paragraphs = [[[10] * 100, [11] * 100], [[20] * 100, [21] * 100], [[30] * 100]]

        @property
        def token_num(self):
            return 100

        def __len__(self):
            return len(self.paragraphs)

        def __getitem__(self, index):
            return self.paragraphs[index]
        
    dataset = PretrainDataset(Dummy())

    count = 0
    nsp_true_count = 0
    combinations = set()
    for src, mlm, mask, nsp in dataset:
        # First test
        assert src[0] == mlm[0] == CLS and src[101] == mlm[101] == src[-1] == mlm[101] == SEP and \
               not mask[0] and not mask[101] and not mask[-1], \
                "CLS and SEP should not be masked."
        
        # Second test
        assert not nsp ^ (src[1] // 10 == src[102] // 10 and src[1] % 10 + 1 == src[102] % 10), \
                "Your result does not match NSP label."
        
        # Third test
        assert all(src[1] == src[i] for i in range(2, 100)) and all(src[102] == src[i] for i in range(103, 201)), \
                "You should not modify the source sentence."

        # Forth test
        assert all((w1 == w2 or m) for w1, w2, m in zip(src, mlm, mask)), \
                "Only masked position can have a different token."

        # Fifth test
        temp1 = sum(mask)
        assert .145 < sum(mask) / len(src) < .155, \
                "The number of the masked tokens should be 15%% of the total tokens."

        # Sixth test
        temp2 = sum(word == MSK for word in mlm)
        assert .795 < sum(word == MSK for word in mlm) / sum(mask) < .805, \
                "80%% of the masked tokens should be converted to MSK tokens"
        
        # Seventh test
        assert .095 < sum(w1 != w2 for w1, w2 in zip(src, mlm) if w2 != MSK) / sum(mask) < .105, str(count)
                # "10%% of the masked tokens should be converted to random tokens"
        
        combinations.add((src[1], src[102]))
        nsp_true_count += nsp
        
        count += 1
        print(count)
        if count > 10000:
            break

    # Eighth test
    print(str(nsp_true_count / 10000))
    assert .45 < nsp_true_count / 10000 < .55, \
            "Your NSP label is biased. Buy a lottery if you failed the test with a correct database."

    # Nineth test
    print(len(combinations))
    assert len(combinations) >= 18, \
            "The number of sentence combination is too limited."

    print("MLM & NSP dataset test passed!")

def loss_calculation_test():
    print("======MLM & NSP Loss Calculation Test Case======")
    CLS = BytePairEncoding.CLS_token_idx
    SEP = BytePairEncoding.SEP_token_idx
    MSK = BytePairEncoding.MSK_token_idx
    
    torch.manual_seed(1234)
    model = MLMandNSPmodel(100)

    samples = []

    src = [CLS] + [10, 10, 10, 10, 10, 10] + [SEP] + [20, 20, 20, 20, 20] + [SEP]
    mlm = [CLS] + [10, 10, 10, 10, MSK, 10] + [SEP] + [MSK, 20, 20, 15, 20] + [SEP]
    mask = [False, False, True, False, False, True, False, False, True, False, False, True, False, False]
    nsp = True
    samples.append((src, mlm, mask, nsp))

    src = [CLS] + [30, 30, 30] + [SEP] + [40, 40, 40, 40] + [SEP]
    mlm = [CLS] + [MSK, 30, 30] + [SEP] + [40, 45, 40, 40] + [SEP]
    mask = [False, True, False, True, False, False, True, False, False, False]
    nsp = False
    samples.append((src, mlm, mask, nsp))

    src = [CLS] + [10, 20, 30, 40] + [SEP] + [50, 40, 30, 20, 10] + [SEP]
    mlm = [CLS] + [10, MSK, 30, 40] + [SEP] + [50, MSK, 30, 25, 10] + [SEP]
    mask = [False, False, True, False, False, False, False, True, False, True, False ,False]
    nsp = True
    samples.append((src, mlm, mask, nsp))

    src, mlm, mask, nsp = utils.pretrain_collate_fn(samples)

    MLM_loss, NSP_loss = calculate_losses(model, src, mlm, mask, nsp)

    # First test
    assert MLM_loss.allclose(torch.scalar_tensor(5.12392426), atol=1e-2), \
        "Your MLM loss does not match the expected result"
    print("The first test passed!")

    # Second test
    assert NSP_loss.allclose(torch.scalar_tensor(0.59137219), atol=1e-2), \
        "Your NSP loss does not match the expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def pretrain_model():
    print("======MLM & NSP Pretraining======")
    """ MLM & NSP Pretraining 
    You can modify this function by yourself.
    This function does not affects your final score.
    """
    tokenizer = BertTokenizer(args.vocab_file, max_len=args.max_seq_length)
    comment_train_dataset = ParagraphDataset(os.path.join('data', 'korean-hate-speech', 'unlabeled'))
    title_train_dataset = ParagraphDataset(os.path.join('data', 'korean-hate-speech', 'news_title'))
    train_dataset = PretrainDataset(args.max_seq_length, comment_train_dataset, title_train_dataset, tokenizer)
    comment_val_dataset = ParagraphDataset(os.path.join('data', 'korean-hate-speech', 'unlabeled', 'dev'))
    title_val_dataset = ParagraphDataset(os.path.join('data', 'korean-hate-speech', 'news_title', 'dev'))
    val_dataset = PretrainDataset(args.max_seq_length, comment_val_dataset, title_val_dataset, tokenizer)
    config = Config(args.config_file)
    device = torch.device('cpu')
    model = MlmNspModel(config).to(torch.device(device))
    model.bert.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model_name = 'pretrained'

    MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses \
            = pretraining(model, model_name, train_dataset, val_dataset)

    torch.save(model.state_dict(), model_name+'_final.pth')

    with open(model_name+'_result.pkl', 'wb') as f:
        pickle.dump((MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses), f)

    utils_pretrain.plot_values(MLM_train_losses, MLM_val_losses, title=model_name + "_mlm")
    utils_pretrain.plot_values(NSP_train_losses, NSP_val_losses, title=model_name + "_nsp")

    print("Final MLM training loss: {:06.4f}".format(MLM_train_losses[-1]))
    print("Final MLM validation loss: {:06.4f}".format(MLM_val_losses[-1]))
    print("Final NSP training loss: {:06.4f}".format(NSP_train_losses[-1]))
    print("Final NSP validation loss: {:06.4f}".format(NSP_val_losses[-1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--config_file",
                        type=str,
                        required=True,
                        help="model configuration file")
    parser.add_argument("--vocab_file",
                        type=str,
                        required=True,
                        help="tokenizer vocab file")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="pretrained model checkpoint")
    # Other parameters
    parser.add_argument("--output_dir",
                        default="output",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    # MLM_and_NSP_dataset_test()
    # loss_calculation_test()
    pretrain_model()
