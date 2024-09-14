import argparse
from rsarl.agents import KSPAgentFactory

from rsarl.evaluator import Evaluator
from rsarl.logger import Logger

from rsarl.networks import SingleFiberNetwork
from rsarl.envs import make_serial_vector_env
from example.env import MultiMetricEnv


from rsarl.requester import UniformRequester


import rsarl

hparams = {}
hparams["seed"] = 0
hparams["num_envs"] = 5

# nsf
hparams["n_slot"] = 80
hparams["avg_service_time"] = 10
hparams["avg_request_arrival_rate"] = 12

hparams["n_requests"] = 10000
hparams["warmup_n_requests"] = 3000

rsarl.utils.set_random_seed(hparams["seed"])


class Slot1_4UniformRequester(UniformRequester):

    def bandwidth(self):
        return self.rand_generator.randint(25, 51)


def get_env(nw_name: str):
    # network
    net = SingleFiberNetwork(nw_name, hparams["n_slot"], is_weight=True)
    req = Slot1_4UniformRequester(
        net.n_nodes,
        avg_service_time=hparams["avg_service_time"],
        avg_request_arrival_rate=hparams["avg_request_arrival_rate"],
    )

    # build env
    env = MultiMetricEnv(net, req)
    vec_env = make_serial_vector_env(env, hparams["num_envs"], hparams["seed"], False)
    test_vec_env = make_serial_vector_env(env, hparams["num_envs"], 0, True)

    return env, vec_env, test_vec_env


def get_args():
    parser = argparse.ArgumentParser()
    # general exp settings
    parser.add_argument(
        "-exp", "--exp_name", default=None, help="Name of the experiment"
    )
    parser.add_argument(
        "-k", "--k", type=int, default=5, help="The number of paths to consider"
    )
    # algorithm & network settings
    parser.add_argument(
        "-sa", "--sa", default="ff", help="Name of spectrum assignment algorithm"
    )
    parser.add_argument("-nw", "--nw", default="nsf", help="Name of network")
    # logger settings
    parser.add_argument(
        "--save", default=False, action="store_true", help="Enable save"
    )
    parser.add_argument("-db", "--db", default="rsa-rl.db", help="Name of the database")
    parser.add_argument(
        "-ow",
        "--overwrite",
        default=False,
        action="store_true",
        help="Enable overwrite DB",
    )
    return parser.parse_args()


def main():
    args = get_args()

    env, _, test_vec_env = get_env(args.nw)

    if args.exp_name is None:
        exp_name = f"{args.k}-sp-{args.sa}-{args.nw}"
    else:
        exp_name = args.exp_name

    # agent
    agent = KSPAgentFactory.create(args.sa, args.k)
    # pre-calculate all path related to all combination of a pair of nodes
    agent.prepare_ksp_table(env.net)

    logger = Logger(
        exp_name,
        db_name=args.db,
        save_experience=args.save,
        is_overwrite=args.overwrite,
        use_tensorboard=True,
    )

    logger.save_experiment(env, agent, hparams)

    evaluator = Evaluator(
        test_vec_env,
        warming_up_steps=hparams["warmup_n_requests"],
        evalutate_steps=hparams["n_requests"],
        logger=logger,
    )

    evaluator(agent)


if __name__ == "__main__":
    main()
