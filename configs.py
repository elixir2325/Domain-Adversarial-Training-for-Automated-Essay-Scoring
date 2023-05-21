class Configs:

    data_path = 'bert_dann/data/training_set_rel3.xlsx'

    lm_path = 'bert-base-uncased'
    seq_len = 512
    encoder_dim = 768
    dropout = 0.2
    # max_iters = 1000

    epochs = 10
    iters_per_epoch = 200
    max_iters = epochs * iters_per_epoch
    batch_size = 3
    lr = 2e-5
    mu = 0.005