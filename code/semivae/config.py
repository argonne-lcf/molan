import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=256,
                           help='Encoder h dimensionality')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.25,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=3,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0.25,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=287,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_d_h',
                           type=int, default=681,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')

    model_arg.add_argument('--y_dropout',
                           type=float, default=0.05,
                           help='y layers dropout')
    model_arg.add_argument('--y_act',
                           type=str, default='relu',
                           help='y activation layers dropout')
    model_arg.add_argument('--y_n_layers',
                           type=int, default=2,
                           help='y num layers')
    model_arg.add_argument('--y_layer_ratio_size',
                           type=float, default=0.5,
                           help='y layer size ratio')
    model_arg.add_argument('--y_batchnorm',
                           type=bool, default=False,
                           help='Use batchnorm for y')

    # Train
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    train_arg.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    train_arg.add_argument('--kl_cycle_length',
                           type=int, default=15,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_cycles_constant',
                           type=float, default=3,
                           help='Initial kl weight value')
    train_arg.add_argument('--y_start',
                           type=int, default=0,
                           help='Epoch to start change y weight from')
    train_arg.add_argument('--y_end',
                           type=int, default=60,
                           help='Epoch to end change y weight from')
    train_arg.add_argument('--y_w_start',
                           type=float, default=0,
                           help='Initial y weight value')
    train_arg.add_argument('--y_w_end',
                           type=float, default=1,
                           help='Maximum y weight value')
    train_arg.add_argument('--lr',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers',
                           type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
