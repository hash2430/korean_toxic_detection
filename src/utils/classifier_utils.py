import os
import sys
import csv
import logging
import pandas as pd

from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class KorTDTestProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        news_title_path = os.path.join(data_dir, 'news_title', 'train.news_title.txt')
        with open(news_title_path, 'r', encoding='utf-8-sig') as f:
            titles = f.readlines()
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'labeled','train.tsv')),
            titles,
            "train")



    def get_dev_examples(self, data_dir):
        news_title_path = os.path.join(data_dir, 'news_title', 'dev.news_title.txt')
        with open(news_title_path, 'r', encoding='utf-8-sig') as f:
            titles = f.readlines()

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'labeled', 'dev.tsv')),
            titles,
            "dev"
        )

    def get_test_examples(self, data_dir):
        news_title_path = os.path.join(data_dir, 'news_title', 'test.news_title.txt')
        with open(news_title_path, 'r', encoding='utf-8-sig') as f:
            titles = f.readlines()

        csv = pd.read_csv(os.path.join(data_dir, 'labeled', 'test.gender_bias.no_label.csv')).values
        return self._create_examples(
            csv,
            titles,
            "test"
        )

    def get_labels(self):
        return [['True','False'],['gender','others','none'],['hate','offensive','none']]

    def _create_examples(self, lines, titles, set_type):
        examples = []
        lines = lines[:,0]
        for line, title in zip(lines, titles):
            guid = "%s-%s" % (set_type, line)
            text_a = line
            text_b = title.strip()
            label_gender_bias = 'False'
            label_bias = 'none'
            label_hate = 'none'
            label = [label_gender_bias, label_bias, label_hate]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class KorTDProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        news_title_path = os.path.join(data_dir, 'news_title', 'train.news_title.txt')
        with open(news_title_path, 'r', encoding='utf-8-sig') as f:
            titles = f.readlines()
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'labeled','train.tsv')),
            titles,
            "train")

    def get_dev_examples(self, data_dir):
        news_title_path = os.path.join(data_dir, 'news_title', 'dev.news_title.txt')
        with open(news_title_path, 'r', encoding='utf-8-sig') as f:
            titles = f.readlines()

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'labeled', 'dev.tsv')),
            titles,
            "dev"
        )

    def get_test_examples(self, data_dir):
        news_title_path = os.path.join(data_dir, 'news_title', 'test.news_title.txt')
        with open(news_title_path, 'r', encoding='utf-8-sig') as f:
            titles = f.readlines()
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'labeled', 'test.tsv')),
            titles,
            "test"
        )

    def get_labels(self):
        return [['True','False'],['gender','others','none'],['hate','offensive','none']]

    def _create_examples(self, lines, titles, set_type):
        examples = []
        lines = lines[1:]
        for line, title in zip(lines, titles):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            text_b = title.strip()
            label_gender_bias = line[1]
            label_bias = line[2]
            label_hate = line[3]
            label = [label_gender_bias, label_bias, label_hate]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
class KorNLIProcessor(DataProcessor):
    """Processor for the KorNLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "multinli.train.ko.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "xnli.dev.ko.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "xnli.test.ko.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class KorSTSProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sts-train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sts-dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sts-test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[5]
            text_b = line[6]
            label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map0 = {}
    label_map1 = {}
    label_map2 = {}
    if output_mode == "classification_multitask":
        for i, label in enumerate(label_list[0]):
            label_map0[label]=i
        for i, label in enumerate(label_list[1]):
            label_map1[label] = i
        for i, label in enumerate(label_list[2]):
            label_map2[label] = i

    else: label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification_multitask":
            label_id = [label_map0[example.label[0]], label_map1[example.label[1]], label_map2[example.label[2]]]
        elif output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            if output_mode == "classification_multitask":
                logger.info("label: {}, {}, {}, (id={}, {}, {})".format(
                    example.label[0], example.label[1], example.label[2], label_id[0], label_id[1], label_id[2]))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    if task_name == "kornli":
        assert len(preds) == len(labels)
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "korsts":
        assert len(preds) == len(labels)
        return pearson_and_spearman(preds, labels)
    elif task_name == "kortd":
        assert len(preds[0]) == len(labels)
        return {"acc0": simple_accuracy(preds[0], labels[:,0]),
                "acc1": simple_accuracy(preds[1], labels[:,1]),
                "acc2": simple_accuracy(preds[2], labels[:,2])}
    else:
        raise KeyError(task_name)