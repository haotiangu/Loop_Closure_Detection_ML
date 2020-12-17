import torch

class Args(object):
    log_dir = 'log/'
    results_dir = 'results/'
    positives_per_query = 2
    negatives_per_query = 18
    max_epoch = 10
    batch_num_queries = 2
    learning_rate = 0.000005
    momentum = 0.9
    optimizer = 'adam'
    decay_step = 200000
    margin_1 = 0.5
    margin_2 = 0.2
    loss_function = 'quadruplet' # ['triplet', 'quadruplet']
    loss_not_lazy = 'store_false'
    loss_ignore_zero_batch = True
    triplet_use_best_positives = True



class Config(object):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99
    HARD_NEGATIVES = {}
    TRAINING_LATENT_VECTORS = []
    TOTAL_ITERATIONS = 0
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # GLOBAL
    NUM_POINTS = 4096
    FEATURE_OUTPUT_DIM = 256
    RESULTS_FOLDER = "results/"
    OUTPUT_FILE = "results/results.txt"

    LOG_DIR = 'log/'
    MODEL_FILENAME = "model.ckpt"

    DATASET_FOLDER = 'benchmark_datasets/'

    # TRAIN
    BATCH_NUM_QUERIES = 2
    TRAIN_POSITIVES_PER_QUERY = 2
    TRAIN_NEGATIVES_PER_QUERY = 18
    DECAY_STEP = 200000
    DECAY_RATE = 0.7
    BASE_LEARNING_RATE = 0.000005
    MOMENTUM = 0.9
    OPTIMIZER = 'ADAM'
    MAX_EPOCH = 20
    MARGIN_1 = 0.5
    MARGIN_2 = 0.2

    RESUME = False

    TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
    TEST_FILE = 'generating_queries/test_queries_baseline.pickle'

    # LOSS
    LOSS_FUNCTION = 'quadruplet'
    LOSS_LAZY = True
    TRIPLET_USE_BEST_POSITIVES = False
    LOSS_IGNORE_ZERO_BATCH = False

    # EVAL6
    EVAL_BATCH_SIZE = 2
    EVAL_POSITIVES_PER_QUERY = 4
    EVAL_NEGATIVES_PER_QUERY = 12

    EVAL_DATABASE_FILE = 'generating_queries/oxford_evaluation_database.pickle'
    EVAL_QUERY_FILE = 'generating_queries/oxford_evaluation_query.pickle'



args = Args()
cfg = Config()