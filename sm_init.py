#! usr/bin/env python3
# -*- coding:utf-8 -*-

# !pip install spacy
# !python -m spacy download en_core_web_sm


import tensorflow as tf
# from horovod.tensorflow.compression import Compression
import horovod.tensorflow as hvd
import tokenization

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1" 
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "NER", "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "albert_base_zh/albert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_export", False, "Whether to export model to pb format.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")





# Create the output directory where all the results are saved.
# output_dir = os.path.join(working_dir, 'output')
# tf.gfile.MakeDirs(output_dir)

# The config json file corresponding to the pre-trained BERT model.
# This specifies the model architecture.
bert_config_file = os.path.join(DATA_DIR_BIOBERT, 'bert_config.json')

# The vocabulary file that the BERT model was trained on.

init_checkpoint = os.path.join(DATA_DIR_BIOBERT, 'model.ckpt')

batch_size = 1
params = dict([('batch_size', batch_size)])
  
# The maximum total input sequence length after WordPiece tokenization. 
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
max_seq_length = 128

output_dir = os.path.join(working_dir, 'results')

train_batch_size=64

num_train_epochs=1
num_warmup_steps=0
max_seq_length=128
hvd_rank = 0

verbose_logging = True
# Set to True if the dataset has samples with no answers. For SQuAD 1.1, this is set to False
version_2_with_negative = False

# The total number of n-best predictions to generate in the nbest_predictions.json output file.
n_best_size = 20

# The maximum length of an answer that can be generated. 
# This is needed  because the start and end predictions are not conditioned on one another.
max_answer_length = 30

# The initial learning rate for Adam
learning_rate = 5e-6

# Total batch size for training
train_batch_size = 3

# Proportion of training to perform linear learning rate warmup for
warmup_proportion = 0.1

# # Total number of training epochs to perform (results will improve if trained with epochs)
num_train_epochs = 1

from utils.utils import LogEvalRunHook, LogTrainRunHook, setup_xla_flags


#hvd
hvd.init()
tmp_filenames = [os.path.join(notebooks_dir, "train.tf_record{}".format(i)) for i in range(hvd.size())]
num_examples_per_rank = len(train_examples) // hvd.size()
remainder = len(train_examples) % hvd.size()
if hvd.rank() < remainder:
    start_index = hvd.rank() * (num_examples_per_rank+1)
    end_index = start_index + num_examples_per_rank + 1
else:
    start_index = hvd.rank() * num_examples_per_rank + remainder
    end_index = start_index + (num_examples_per_rank)

tf.compat.v1.logging.info("hvd.size()",hvd.size())


#tokenize
vocab_file = os.path.join(DATA_DIR_BIOBERT, 'vocab.txt')

# Should be True for uncased models and False for cased models.
# The BioBERT available in NGC is uncased
do_lower_case = True

# Validate the casing config consistency with the checkpoint name.
tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

# Create the tokenizer.
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

bert_config_file = os.path.join(DATA_DIR_BIOBERT, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

processor = BC5CDRProcessor()
label_list = processor.get_labels()

id2label = {}
for (i, label) in enumerate(label_list, 1):
    id2label[i] = label


# config = tf.ConfigProto(log_device_placement=True) 
# run_config = tf.estimator.RunConfig(
#       model_dir=output_dir,
#       session_config=config,
#       save_checkpoints_steps=1000,
#       keep_checkpoint_max=1)


train_examples = processor.get_test_examples(notebooks_dir, file_name='train.tsv')
num_train_steps = 1 #int(len(train_examples) / train_batch_size * num_train_epochs)

start_index = 0
end_index = len(train_examples)
tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]


filed_based_convert_examples_to_features(train_examples[start_index:end_index], 
                                         label_list, max_seq_length, tokenizer, tmp_filenames[hvd_rank])

# predict_examples = processor.get_test_examples(notebooks_dir, file_name='input.tsv')
# tf.compat.v1.logging.info("***** Running training *****")
# tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))

train_input_fn = file_based_input_fn_builder(
        input_file=tmp_filenames, #train_file,
        batch_size=train_batch_size,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True,
        hvd=None if not FLAGS.horovod else hvd)

global_batch_size = train_batch_size
training_hooks = []
training_hooks.append(LogTrainRunHook(global_batch_size, 0))

model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        hvd=None if not FLAGS.horovod else hvd,
        amp=FLAGS.amp)

print("***** Running training *****")
print("  Num examples = %d", len(train_examples))
print("  Batch size = %d", FLAGS.train_batch_size)
print("  Num steps = %d", num_train_steps)
