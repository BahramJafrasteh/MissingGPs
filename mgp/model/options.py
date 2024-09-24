import argparse
import os


class options:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser = self.initialize(parser)
        self.parser = parser

    def initialize(self, parser):
        parser.add_argument("--dataset_name", default=None, required=True)
        parser.add_argument("--name", default="nn", help="DSVI_Latent_Inverse | DSVI")

        parser.add_argument("--nolayers", type=int, default=0, help="number of players")

        parser.add_argument(
            "--n_samples", default=20, type=int, help="number of samples"
        )

        parser.add_argument(
            "--M", type=int, default=10, help="number of inducing points"
        )
        parser.add_argument(
            "--M2", type=int, default=10, help="number of inducing points missing"
        )
        parser.add_argument(
            "--scaling",
            type=str,
            default="MeanStd",
            help="scaling method [MeanStd|MinMax|MaxAbs|Robust|None]",
        )

        parser.add_argument(
            "--imputation",
            type=str,
            default="median",
            help="imputation method [medina|meann|knn|mice|None]",
        )

        parser.add_argument(
            "--Ptype",
            type=str,
            default="reg",
            help="Problem type (regression or classification) [reg|class]",
        )

        parser.add_argument(
            "--kernel", type=str, default="matern", help="matern|rbf|nsrbf"
        )

        parser.add_argument(
            "--no_iterations", type=int, default=1000, help="number of inducing points"
        )

        parser.add_argument(
            "--var_noise", type=float, default=1e-4, help="variance noise"
        )

        parser.add_argument("--lrate", type=float, default=1e-2, help="learning rate")

        parser.add_argument(
            "--likelihood_var",
            type=float,
            default=0.01,
            help="variance noise gaussian likelihood",
        )

        parser.add_argument(
            "--minibatch_size", type=int, default=100, help="mini-batch size"
        )

        parser.add_argument("--n_epoch", type=int, default=100, help="mini-batch size")

        parser.add_argument(
            "--numThreads", type=int, default=1, help="number of threads"
        )

        parser.add_argument(
            "--fitting", action="store_true", help="fitting the model to the data"
        )

        parser.add_argument(
            "--missing", action="store_true", help="shared parameters in nueral network"
        )

        parser.add_argument(
            "--consider_miss",
            action="store_true",
            help="shared parameters in nueral network",
        )

        parser.add_argument("--split_number", type=int, default=0, help="split number")

        parser.add_argument("--nGPU", type=int, default=0, help="GPU number")

        return parser

    def parse(self, write_conf=True):
        opt = self.parser.parse_args()

        if opt.missing:
            add = "missing"
        else:
            add = "non_missing"
        if write_conf:
            path_config = "results/{}/{}/s{}_{}_{}_{}/".format(
                opt.name.lower(),
                opt.dataset_name,
                opt.split_number,
                add,
                opt.imputation,
                opt.nolayers,
            )
            # path_config = "results/{}/{}/s{}/".format(opt.dataset_name, opt.name.upper(), opt.split_number)
            if not os.path.exists(path_config):
                os.makedirs(path_config)

            message = ""
            message += "----------------- Options ---------------\n"
            for k, v in sorted(vars(opt).items()):
                default = self.parser.get_default(k)
                comment = "\t[default: {}] \t[help: {}]\t ".format(
                    str(default), str(self.parser._option_string_actions["--" + k].help)
                )
                message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
            message += "----------------- End -------------------"
            print(message)
            with open(path_config + "conf.txt", "w") as file:
                file.write(message)
        self.opt = opt
        return opt
