"""Neural PCFG integration based on DeepSynth's DreamCoder list model."""

from __future__ import annotations

import importlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import List, Optional, Sequence, Tuple

from .spec import DSLGenerationConfig, DSLProgramRecord

Expression = str


class TranslationError(RuntimeError):
    """Raised when a DeepSynth program cannot be translated to Python."""


@dataclass
class _SampledProgram:
    program: object  # DeepSynth Program
    arity: int
    inputs: Sequence[Sequence[List[int]]]
    outputs: Sequence[object]
    python_source: str
    dsl_repr: str


class DeepSynthNeuralSampler:
    """Wrapper that loads DeepSynth's neural PCFG and exposes sampling utilities."""

    def __init__(
        self,
        config: DSLGenerationConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng
        self._load_deepsynth()
        self._load_pcfg()

    # --------------------------------------------------------------------- #
    # Loading
    # --------------------------------------------------------------------- #
    def _load_deepsynth(self) -> None:
        root_hint = (
            Path(self.config.deepsynth_root).expanduser()
            if self.config.deepsynth_root
            else Path(__file__).resolve().parents[3] / "-IDL-DeepSynth"
        )
        if not root_hint.exists():
            raise FileNotFoundError(
                f"DeepSynth repository not found at {root_hint}. "
                "Set `deepsynth_root` in the config."
            )
        self.deepsynth_root = root_hint
        if str(self.deepsynth_root) not in sys.path:
            sys.path.insert(0, str(self.deepsynth_root))

        required = [
            "model_loader",
            "type_system",
            "cons_list",
            "dsl",
            "DSL.list",
        ]
        for module_name in required:
            importlib.import_module(module_name)

    def _load_pcfg(self) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for the neural PCFG integration. Make sure it is "
                "installed in the active environment."
            )
        from model_loader import build_dreamcoder_intlist_model
        from type_system import Arrow, INT, List

        dsl, cfg, model = build_dreamcoder_intlist_model(max_program_depth=self.config.max_depth)

        dummy_examples = [
            ([self._dummy_list()], self._dummy_list()),
            ([self._dummy_list()], self._dummy_list()),
        ]
        batch_ios = [
            [([inp], out) for inp, out in dummy_examples],
        ]
        with torch.no_grad():
            predictions = model(batch_ios)
            grammars = model.reconstruct_grammars(predictions)
        self.dsl = dsl
        self.pcfg = grammars[0]
        self.pcfg.normalise()
        self.pcfg.init_vose()
        self.type_request = Arrow(List(INT), List(INT))

        from cons_list import tuple2constlist, cons_list2list

        self._tuple2constlist = tuple2constlist
        self._cons_list2list = cons_list2list

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def sample_record(self) -> Optional[DSLProgramRecord]:
        attempts = 0
        while attempts < self.config.neural_max_attempts:
            attempts += 1
            program = self._sample_program()
            try:
                sampled = self._build_sample(program)
            except TranslationError:
                continue
            if self._validate_outputs(sampled.outputs):
                return DSLProgramRecord(
                    arity=sampled.arity,
                    depth=self.config.max_depth,
                    dsl=sampled.dsl_repr,
                    python=sampled.python_source,
                    inputs=sampled.inputs,
                    outputs=sampled.outputs,
                    metadata={"source": "deepsynth_neural"},
                )
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sample_program(self):
        sampler = self.pcfg.sampling()
        return next(sampler)

    def _build_sample(self, program) -> _SampledProgram:
        arity = self.config.arity
        inputs, outputs = self._run_examples(program, arity)
        python_body = self._translate_program(program, arity)
        python_source = self._emit_python(python_body, arity)
        return _SampledProgram(
            program=program,
            arity=arity,
            inputs=inputs,
            outputs=outputs,
            python_source=python_source,
            dsl_repr=repr(program),
        )

    def _run_examples(self, program, arity: int) -> Tuple[List[Sequence[List[int]]], List[object]]:
        examples: List[Sequence[List[int]]] = []
        outputs: List[object] = []
        trials = max(self.config.input_trials, 1)
        for _ in range(trials * 2):
            args: List[List[int]] = [self._sample_list() for _ in range(arity)]
            env = self._tuple2constlist(tuple(args))
            result = program.eval_naive(self.dsl, env)
            if result is None:
                continue
            result_py = self._materialise(result)
            examples.append(tuple(args))
            outputs.append(result_py)
            if len(examples) >= trials:
                break
        if len(examples) < trials:
            raise TranslationError("insufficient valid IO pairs generated")
        return examples, outputs

    def _materialise(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return self._cons_list2list(value)
        return value

    def _translate_program(self, program, arity: int) -> Expression:
        translator = _ProgramTranslator(arity)
        expr = translator.translate(program)
        if not isinstance(expr, _Expression):
            raise TranslationError("expected expression after translation")
        return expr.code

    def _emit_python(self, body_expr: Expression, arity: int) -> str:
        params = ", ".join(f"a{i+1}" for i in range(arity))
        header = f"def f({params}):"
        body_lines: List[str] = []
        if "math." in body_expr:
            body_lines.append("import math")
        if body_expr.strip().startswith("return"):
            body_lines.append(body_expr.strip())
        else:
            body_lines.append(f"return {body_expr}")
        return "\n".join([header, indent("\n".join(body_lines), "    ")])

    def _validate_outputs(self, outputs: Sequence[object]) -> bool:
        if len(outputs) < 2:
            return False
        first = json.dumps(outputs[0], sort_keys=True)
        return any(json.dumps(o, sort_keys=True) != first for o in outputs[1:])

    def _sample_list(self) -> List[int]:
        length = self.rng.randint(3, 5)
        return [self.rng.randint(0, 5) for _ in range(length)]

    def _dummy_list(self) -> List[int]:
        return [0, 1, 2]


# ------------------------------------------------------------------------------
# Translation layer: DeepSynth Program -> Python expression
# ------------------------------------------------------------------------------


class _Expression:
    def __init__(self, code: str) -> None:
        self.code = code

    def __repr__(self) -> str:
        return f"_Expression({self.code!r})"


class _LambdaClosure:
    def __init__(self, translator: "_ProgramTranslator", body, env_stack: List[_Expression]) -> None:
        self.translator = translator
        self.body = body
        self.env_stack = env_stack

    def apply(self, arg: _Expression):
        new_stack = [arg] + self.env_stack
        return self.translator._translate(self.body, new_stack)

    def as_lambda(self) -> str:
        param = self.translator.fresh_lambda_arg()
        placeholder = _Expression(param)
        new_stack = [placeholder] + self.env_stack
        body_expr = self.translator._translate(self.body, new_stack)
        if not isinstance(body_expr, _Expression):
            raise TranslationError("lambda body did not resolve to expression")
        return f"lambda {param}: {body_expr.code}"


def _ensure_expression(value: object, translator: "_ProgramTranslator") -> _Expression:
    if isinstance(value, _Expression):
        return value
    if isinstance(value, _LambdaClosure):
        return _Expression(value.as_lambda())
    if isinstance(value, _PrimitiveFunction):
        return _Expression(value.as_lambda())
    raise TranslationError(f"Unsupported value type for expression conversion: {type(value).__name__}")


class _PrimitiveFunction:
    def __init__(
        self,
        name: str,
        total_arity: int,
        builder,
        translator: "_ProgramTranslator",
        args: Optional[List[object]] = None,
    ):
        self.name = name
        self.total_arity = total_arity
        self.builder = builder
        self.translator = translator
        self.args = args or []

    def apply(self, arg: _Expression):
        collected = self.args + [arg]
        if len(collected) == self.total_arity:
            converted = [_ensure_expression(item, self.translator) for item in collected]
            return self.builder(converted)
        return _PrimitiveFunction(
            self.name,
            self.total_arity,
            self.builder,
            self.translator,
            collected,
        )

    def as_lambda(self) -> str:
        remaining = self.total_arity - len(self.args)
        placeholders = [self.translator.fresh_lambda_arg() for _ in range(max(remaining, 0))]
        placeholder_exprs = [_Expression(name) for name in placeholders]
        full_args = self.args + placeholder_exprs
        converted = [_ensure_expression(item, self.translator) for item in full_args]
        result = self.builder(converted)
        if not isinstance(result, _Expression):
            raise TranslationError(f"Primitive {self.name} did not produce an expression")
        if remaining <= 0:
            return result.code
        params = ", ".join(placeholders)
        return f"lambda {params}: {result.code}"


PRIMITIVE_BUILDERS = {}


def _register(name, arity):
    def decorator(fn):
        PRIMITIVE_BUILDERS[name] = (arity, fn)
        return fn

    return decorator


@_register("empty", 0)
def _build_empty(_args):
    return _Expression("[]")


@_register("append", 2)
def _build_append(args):
    value, lst = args
    return _Expression(f"({lst.code}) + [{value.code}]")


@_register("cons", 2)
def _build_cons(args):
    head, tail = args
    return _Expression(f"[{head.code}] + ({tail.code})")


@_register("car", 1)
def _build_car(args):
    (lst,) = args
    return _Expression(f"({lst.code})[0]")


@_register("cdr", 1)
def _build_cdr(args):
    (lst,) = args
    return _Expression(f"({lst.code})[1:]")


@_register("length", 1)
def _build_length(args):
    (lst,) = args
    return _Expression(f"len({lst.code})")


@_register("index", 2)
def _build_index(args):
    idx, lst = args
    return _Expression(f"({lst.code})[{idx.code}]")


@_register("+", 2)
def _build_add(args):
    a, b = args
    return _Expression(f"({a.code}) + ({b.code})")


@_register("-", 2)
def _build_sub(args):
    a, b = args
    return _Expression(f"({a.code}) - ({b.code})")


@_register("*", 2)
def _build_mul(args):
    a, b = args
    return _Expression(f"({a.code}) * ({b.code})")


@_register("mod", 2)
def _build_mod(args):
    a, b = args
    return _Expression(f"({b.code}) % ({a.code})")


@_register("max", 2)
def _build_max(args):
    a, b = args
    return _Expression(f"max({a.code}, {b.code})")


@_register("min", 2)
def _build_min(args):
    a, b = args
    return _Expression(f"min({a.code}, {b.code})")


@_register("gt?", 2)
def _build_gt(args):
    a, b = args
    return _Expression(f"({a.code}) > ({b.code})")


@_register("le?", 2)
def _build_le(args):
    a, b = args
    return _Expression(f"({a.code}) <= ({b.code})")


@_register("eq?", 2)
def _build_eq(args):
    a, b = args
    return _Expression(f"({a.code}) == ({b.code})")


@_register("not", 1)
def _build_not(args):
    (a,) = args
    return _Expression(f"not ({a.code})")


@_register("if", 3)
def _build_if(args):
    cond, then, els = args
    return _Expression(f"({then.code}) if ({cond.code}) else ({els.code})")


@_register("range", 1)
def _build_range(args):
    (limit,) = args
    return _Expression(f"list(range({limit.code}))")


@_register("map", 2)
def _build_map(args):
    fn, lst = args
    if isinstance(fn, _LambdaClosure):
        lambda_src = fn.as_lambda()
    elif isinstance(fn, _Expression):
        lambda_src = fn.code
    else:
        raise TranslationError("unsupported map function operand")
    return _Expression(f"list(map({lambda_src}, {lst.code}))")


@_register("filter", 2)
def _build_filter(args):
    fn, lst = args
    if isinstance(fn, _LambdaClosure):
        lambda_src = fn.as_lambda()
    elif isinstance(fn, _Expression):
        lambda_src = fn.code
    else:
        raise TranslationError("unsupported filter function operand")
    return _Expression(f"list(filter({lambda_src}, {lst.code}))")


@_register("is-mod", 2)
def _build_is_mod(args):
    mod_base, value = args
    return _Expression(f"(({value.code}) % ({mod_base.code}) == 0)")


@_register("is-prime", 1)
def _build_is_prime(args):
    (val,) = args
    return _Expression(
        "all(value % i for i in range(2, int(math.sqrt(value)) + 1)) if value >= 2 else False".replace(
            "value", val.code
        )
    )


@_register("is-square", 1)
def _build_is_square(args):
    (val,) = args
    return _Expression(
        "(int(math.sqrt({0})) ** 2 == {0})".format(val.code)
    )


class _ProgramTranslator:
    def __init__(self, arity: int) -> None:
        self.initial_stack = [_Expression(f"a{i+1}") for i in range(arity)]
        self.lambda_counter = 0

    def fresh_lambda_arg(self) -> str:
        name = f"elem_{self.lambda_counter}"
        self.lambda_counter += 1
        return name

    def translate(self, program) -> _Expression:
        return self._translate(program, list(self.initial_stack))

    def _translate(self, program, env_stack: List[_Expression]):
        from program import BasicPrimitive, Function, Lambda, Variable

        if isinstance(program, Variable):
            if program.variable >= len(env_stack):
                raise TranslationError(f"unbound variable index {program.variable}")
            return env_stack[program.variable]
        if isinstance(program, BasicPrimitive):
            if program.primitive.isdigit():
                return _Expression(program.primitive)
            if program.primitive not in PRIMITIVE_BUILDERS:
                raise TranslationError(f"primitive {program.primitive} not supported")
            arity, builder = PRIMITIVE_BUILDERS[program.primitive]
            return _PrimitiveFunction(program.primitive, arity, builder, self)
        if isinstance(program, Function):
            fn_obj = self._translate(program.function, env_stack)
            for arg in program.arguments:
                arg_val = self._translate(arg, env_stack)
                arg_expr = _ensure_expression(arg_val, self)
                if isinstance(fn_obj, _PrimitiveFunction):
                    fn_obj = fn_obj.apply(arg_expr)
                elif isinstance(fn_obj, _LambdaClosure):
                    fn_obj = fn_obj.apply(arg_expr)
                else:
                    fn_expr = _ensure_expression(fn_obj, self)
                    fn_obj = _Expression(f"({fn_expr.code})({arg_expr.code})")
            return _ensure_expression(fn_obj, self)
        if isinstance(program, Lambda):
            return _LambdaClosure(self, program.body, env_stack)
        raise TranslationError(f"unsupported program node {type(program).__name__}")


try:
    import torch  # noqa: E402
except Exception:  # pragma: no cover
    torch = None
