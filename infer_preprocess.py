import os
import sys
import tokenization
import tensorflow as tf
import collections
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, "r") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) > 30:
                        # split if the sentence is longer than 30
                        while len(words) > 30:
                            tmplabel = labels[:30]
                            for iidx in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = ' '.join(
                                [label for label in labels[:len(tmplabel) + 1] if len(label) > 0])
                            w = ' '.join(
                                [word for word in words[:len(tmplabel) + 1] if len(word) > 0])
                            lines.append([l, w])
                            words = words[len(tmplabel) + 1:]
                            labels = labels[len(tmplabel) + 1:]

                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)
            return lines
        
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        
        
class BC5CDRProcessor(DataProcessor):

    def get_test_examples(self, data_dir, file_name="input.tsv"):
        return self._create_example(
            self._read_data(os.path.join(data_dir, file_name)), "test")

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples
    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask
        
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    label2id_file = 'label2id.pkl'
    if not os.path.exists(label2id_file):
        with open(label2id_file, 'wb') as w:
            pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("guid: %s" % (example.guid))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.compat.v1.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.compat.v1.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # write_tokens(ntokens, label_ids, mode)
    return feature
def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                         mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
#         tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#         writer.write(tf_example.SerializeToString())
    return features