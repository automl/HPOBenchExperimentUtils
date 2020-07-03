from math import log10
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter, UniformFloatHyperparameter


# TODO: Add more hyperparameter types and type-specific handlers.

def handler_unknown(hyp):
    raise RuntimeError("No valid handler available for hyperparameter of type %s" % type(hyp))


def handle_uniform_float(hyper: UniformFloatHyperparameter):
    domain = {
        'name': hyper.name,
        'type': 'float',
        'min': log10(hyper.lower) if hyper.log else hyper.lower,
        'max': log10(hyper.upper) if hyper.log else hyper.upper
    }

    parser = (lambda x: 10 ** x) if hyper.log else (lambda x: x)
    return domain, parser


handlers = {
    UniformFloatHyperparameter: handle_uniform_float
}


def _configspace_to_dragonfly_domain(hypers: list[Hyperparameter]):
    domain = {}
    parser = []
    for hyp in hypers:
        d, p = handlers.get(type(hyp), handler_unknown)(hyp)
        domain[hyp.name] = d
        parser.append((hyp.name, p))

    return domain, parser


def configspace_to_dragonfly(cs: ConfigurationSpace, name="hpolib_benchmark"):
    domain, domain_parser = _configspace_to_dragonfly_domain(cs.get_hyperparameters())
    out = {'name': name, 'domain': domain}
    return out, domain_parser
    # TODO: Add support for converting constraints and fidelities