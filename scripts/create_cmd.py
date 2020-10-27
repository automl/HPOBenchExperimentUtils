import argparse
import os

expset_dc = {
    "NAS201": ["Cifar10NasBench201Benchmark", "Cifar10ValidNasBench201Benchmark", "Cifar100NasBench201Benchmark", "ImageNetNasBench201Benchmark"],
    "NAS101": ["NASCifar10ABenchmark", "NASCifar10BBenchmark", "NASCifar10CBenchmark"],
    "NASTAB": ["SliceLocalizationBenchmark", "ProteinStructureBenchmark", "NavalPropulsionBenchmark", "ParkinsonsTelemonitoringBenchmark"],
    "rl": ["cartpolereduced"],
    "learna": ["metalearna", "learna"],
}

opt_set = {
    "def": ["hpbandster_bohb_eta_3", "smac_hb_eta_3", "randomsearch", "dragonfly_default"],
}


def main(args):
    exp = args.exp
    opt = args.opt
    nrep = args.nrep

    if not os.path.isdir(args.out_cmd):
        os.mkdir(args.out_cmd)

    val_fl = "%s/val_%s_%s_%d.cmd" % (args.out_cmd, exp, opt, nrep)
    run_fl = "%s/run_%s_%s_%d.cmd" % (args.out_cmd, exp, opt, nrep)
    eval_fl = "%s/eval_%s_%s_%d.cmd" % (args.out_cmd, exp, opt, nrep)
    evalu_fl = "%s/evalunv_%s_%s_%d.cmd" % (args.out_cmd, exp, opt, nrep)

    run_cmd = []
    val_cmd = []
    eval_cmd = []
    evalu_cmd = []

    base = "python %s/HPOlibExperimentUtils" % args.root

    for benchmark in expset_dc[exp]:
        for optimizer in opt_set[opt]:
            for seed in range(1, nrep+1):
                cmd = "%s/run_benchmark.py --output_dir %s --optimizer %s --benchmark %s --rng %s" % (base, args.out_run, optimizer, benchmark, seed)
                run_cmd.append(cmd)

            cmd = "%s/validate_benchmark.py --output_dir %s/%s/%s --benchmark %s --rng %d" % (base, args.out_run, benchmark, optimizer, benchmark, 1)
            val_cmd.append(cmd)
        cmd = "%s/evaluate_benchmark.py --output_dir %s/ --input_dir %s/ --benchmark %s --rng %d" % (base, args.out_eval, args.out_run, benchmark, 1)
        eval_cmd.append(cmd)
        cmd += " --unvalidated"
        evalu_cmd.append(cmd)

    for c, f in [[run_cmd, run_fl], [val_cmd, val_fl], [eval_cmd, eval_fl], [evalu_cmd, evalu_fl]]:
        write_cmd(c, f)


def write_cmd(cmd_list, out_fl):
    if len(cmd_list) > 10000:
        ct = 0
        while len(cmd_list) > 10000:
            write_cmd(cmd_list[:10000], out_fl + "_%d" % ct)
            ct += 1
            cmd_list = cmd_list[10000:]
    else:
        with open(out_fl, "w") as fh:
            fh.write("\n".join(cmd_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', required=True, choices=expset_dc.keys())
    parser.add_argument('--opt', default="def", choices=opt_set.keys())
    parser.add_argument('--nrep', default=10, type=int)
    parser.add_argument('--out-run', default="./exp_outputs", type=str)
    parser.add_argument('--out-eval', default="./plots", type=str)
    parser.add_argument('--out-cmd', default="./", type=str)
    parser.add_argument('--root', default="./")
    
    args, unknown = parser.parse_known_args()
    main(args)