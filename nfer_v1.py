#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Inspiration/Inspiracija - Upgrade/Nadogradnja 

Dragan Stošić, October 2026. 
https://github.com/gajaka

originalni hazard calculus i subalgebra
PPGraph / NFER: Hazard Analysis 

repo sa kompletnom prezentacijom
PPGRAPH/STATES, ALLOWED EDGES, PROTECTED REGION
kompletan formalni dokaz u pvs i lean

https://github.com/rv-tools/nfer#writing-specifications
"""



"""
Nfer_Loto_V1.py

Osnovna ideja:
    Nfer intervalne apstrakcije nad promenama distribucionih
    osobina celih Loto kombinacija.

Preneti operatori, prema pregledanom Scala kodu:
    before, during, start, finish, meet, overlap, coincide

Prediktivna nadogradnja:
    Deterministička Nadaraya-Watson kernel-regresija
    741 odnosa parova brojeva narednog izvlačenja.

Nfer deo:
    - prepoznaje stanja distribucionih osobina;
    - prati trajanje aktivnih intervala;
    - zatvara interval tek kada je promena stvarno opažena;
    - izvodi složene intervale vremenskim operatorima;
    - ne koristi budući kraj aktivnog intervala.

Regresioni deo:
    - koristi sve prethodne raspoložive trening prelaze;
    - nema odsecanja trening istorije;
    - ne rangira ukupne frekvencije pojedinačnih brojeva;
    - ne pretpostavlja normalnu raspodelu;
    - nema klasifikacionog gubitka niti random postupaka.

Provera:
    - početnih 128 izvlačenja kalibriše pragove događaja;
    - 48 kola služi izboru širine kernela;
    - poslednja 64 kola služe odvojenoj proveri;
    - finalni NEXT koristi sve poznate istorijske prelaze.

HORIZON ograničava raspon pojedinačnog vremenskog pravila.
Ne ograničava broj istorijskih primera za regresiju.
"""


import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import argparse
import csv
import hashlib
import math
import sys

from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Potrebna biblioteka: numpy.\n"
        "Instalacija: python3 -m pip install numpy"
    ) from exc


DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

CALIBRATION = 128
VALIDATION = 48
TEST = 64
MIN_ROWS = 300

HORIZON = 64
BANDWIDTHS = (0.25, 1.0, 4.0)
BEAM_WIDTH = 64

CHANNELS = 5
STATES = 3
SYMBOLS = CHANNELS * STATES

OPERATORS = (
    "before",
    "during",
    "start",
    "finish",
    "meet",
    "overlap",
    "coincide",
)

RULE_COUNT = SYMBOLS * SYMBOLS * len(OPERATORS)

PAIR_I, PAIR_J = np.triu_indices(39, 1)
DRAW_PAIR_I, DRAW_PAIR_J = np.triu_indices(7, 1)

CHANNEL_NAMES = (
    "Zbir",
    "Raspon",
    "CDF_razdaljina_2",
    "CDF_razdaljina_8",
    "CDF_razdaljina_20",
)


def load_csv(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 kolona; dobijeno {len(row)}."
                )

            try:
                numbers = [int(value.strip()) for value in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "potrebni su celi brojevi."
                ) from exc

            if len(set(numbers)) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            if min(numbers) < 1 or max(numbers) > 39:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan opsega 1..39."
                )

            rows.append(sorted(numbers))

    if len(rows) < MIN_ROWS:
        raise ValueError(
            f"{path.name}: potrebno najmanje {MIN_ROWS} izvlačenja."
        )

    draws = np.asarray(rows, dtype=np.int64)

    canonical = "\n".join(
        ",".join(map(str, draw)) for draw in draws
    ).encode("ascii")

    fingerprint = hashlib.sha256(canonical).hexdigest()
    return path, draws, fingerprint


def describe_draw(draw):
    values = draw.astype(float)

    distances = np.abs(
        values[:, None] - values[None, :]
    )[DRAW_PAIR_I, DRAW_PAIR_J]

    profile = np.r_[
        values / 39.0,
        [np.mean(distances <= limit) for limit in (2, 5, 10, 20)],
        (values[-1] - values[0]) / 38.0,
    ]

    events = np.array([
        values.sum() / 273.0,
        (values[-1] - values[0]) / 38.0,
        np.mean(distances <= 2),
        np.mean(distances <= 8),
        np.mean(distances <= 20),
    ], dtype=float)

    return profile, events


@dataclass(frozen=True)
class Interval:
    symbol: int
    begin: int
    end: int


def infer_relations(a, b):
    """
    Uslovi i krajnje tačke odgovaraju pregledanom Nfer kodu.

    Operatori nisu međusobno isključivi.
    Jedan par intervala može zadovoljiti više operatora.
    """
    result = []

    both_atomic = a.begin == a.end and b.begin == b.end

    before = (
        a.end < b.begin
        if both_atomic
        else a.end <= b.begin
    )

    if before:
        result.append((0, a.begin, b.end))

    if a.begin >= b.begin and a.end <= b.end:
        result.append((1, b.begin, b.end))

    if a.begin == b.begin:
        result.append((2, a.begin, max(a.end, b.end)))

    if a.end == b.end:
        result.append((3, min(a.begin, b.begin), a.end))

    if a.end == b.begin:
        result.append((4, a.begin, b.end))

    if a.begin < b.end and b.begin < a.end:
        result.append((
            5,
            min(a.begin, b.begin),
            max(a.end, b.end),
        ))

    if a.begin == b.begin and a.end == b.end:
        result.append((6, a.begin, a.end))

    return result


class NferContext:
    def __init__(self, thresholds):
        self.thresholds = thresholds

        self.active_symbol = np.full(CHANNELS, -1, dtype=int)
        self.active_begin = np.zeros(CHANNELS, dtype=int)

        self.completed = []

        self.base_available = np.full(SYMBOLS, -1, dtype=int)
        self.base_duration = np.zeros(SYMBOLS, dtype=float)

        self.rule_available = np.full(RULE_COUNT, -1, dtype=int)
        self.rule_duration = np.zeros(RULE_COUNT, dtype=float)

        self.completed_count = 0
        self.inference_count = 0

    def publish_pair(self, a, b, now):
        # Relacije se izvode između različitih distribucionih kanala.
        if a.symbol // STATES == b.symbol // STATES:
            return

        total_span = max(a.end, b.end) - min(a.begin, b.begin)

        if total_span > HORIZON:
            return

        for operator, begin, end in infer_relations(a, b):
            rule = (
                (a.symbol * SYMBOLS + b.symbol)
                * len(OPERATORS)
                + operator
            )

            self.rule_available[rule] = now
            self.rule_duration[rule] = end - begin + 1
            self.inference_count += 1

    def submit(self, time_index, values):
        newly_completed = []

        for channel in range(CHANNELS):
            state = int(np.searchsorted(
                self.thresholds[channel],
                values[channel],
                side="right",
            ))

            symbol = channel * STATES + state

            if self.active_symbol[channel] < 0:
                self.active_symbol[channel] = symbol
                self.active_begin[channel] = time_index

            elif symbol != self.active_symbol[channel]:
                interval = Interval(
                    symbol=int(self.active_symbol[channel]),
                    begin=int(self.active_begin[channel]),
                    end=time_index - 1,
                )

                newly_completed.append(interval)

                self.active_symbol[channel] = symbol
                self.active_begin[channel] = time_index

        # Pravila imaju eksplicitni maksimalni raspon HORIZON.
        # Stariji intervali izvan tog raspona ne mogu zadovoljiti pravilo.
        self.completed = [
            interval for interval in self.completed
            if interval.end >= time_index - HORIZON
        ]

        for interval in newly_completed:
            self.base_available[interval.symbol] = time_index
            self.base_duration[interval.symbol] = (
                interval.end - interval.begin + 1
            )
            self.completed_count += 1

            for older in self.completed:
                self.publish_pair(older, interval, time_index)
                self.publish_pair(interval, older, time_index)

            self.completed.append(interval)

        active = np.zeros(SYMBOLS, dtype=float)

        for channel in range(CHANNELS):
            symbol = self.active_symbol[channel]
            duration = time_index - self.active_begin[channel] + 1

            active[symbol] = (
                math.log1p(duration) / math.log1p(HORIZON)
            )

        base_recency = np.zeros(SYMBOLS, dtype=float)
        base_duration = np.zeros(SYMBOLS, dtype=float)

        observed_base = self.base_available >= 0

        base_recency[observed_base] = np.exp(
            -(time_index - self.base_available[observed_base]) / HORIZON
        )

        base_duration[observed_base] = (
            np.log1p(self.base_duration[observed_base])
            / math.log1p(HORIZON)
        ) * base_recency[observed_base]

        rule_recency = np.zeros(RULE_COUNT, dtype=float)
        rule_duration = np.zeros(RULE_COUNT, dtype=float)

        observed_rules = self.rule_available >= 0

        rule_recency[observed_rules] = np.exp(
            -(time_index - self.rule_available[observed_rules]) / HORIZON
        )

        rule_duration[observed_rules] = (
            np.log1p(self.rule_duration[observed_rules])
            / math.log1p(HORIZON)
        ) * rule_recency[observed_rules]

        # Normalizacija veličine blokova osobina.
        # Ne određuje raspored Loto brojeva.
        return np.r_[
            active / math.sqrt(SYMBOLS),
            base_recency / math.sqrt(SYMBOLS),
            base_duration / math.sqrt(SYMBOLS),
            rule_recency / math.sqrt(RULE_COUNT),
            rule_duration / math.sqrt(RULE_COUNT),
        ]


def build_features(draws):
    described = [describe_draw(draw) for draw in draws]
    profiles = np.asarray([item[0] for item in described])
    event_values = np.asarray([item[1] for item in described])

    # Kalibracija je fiksna i poznata pre validacije i završne provere.
    thresholds = np.quantile(
        event_values[:CALIBRATION],
        (1.0 / 3.0, 2.0 / 3.0),
        axis=0,
    ).T

    engine = NferContext(thresholds)
    features = []

    for index in range(len(draws)):
        intervals = engine.submit(index, event_values[index])

        numerical = np.concatenate([
            profiles[index],
            profiles[max(0, index - 3):index + 1].mean(axis=0),
            profiles[max(0, index - 15):index + 1].mean(axis=0),
            profiles[max(0, index - 31):index + 1].mean(axis=0),
        ])

        features.append(np.r_[
            numerical / math.sqrt(len(numerical)),
            intervals,
        ].astype(np.float32))

        if (index + 1) % 500 == 0:
            print(
                f"  Nfer obrađeno: {index + 1}/{len(draws)}",
                flush=True,
            )

    return np.asarray(features), engine, thresholds


def make_targets(draws):
    targets = draws[1:]

    membership = np.zeros((len(targets), 39), dtype=float)
    membership[
        np.arange(len(targets))[:, None], targets - 1
    ] = 1.0

    pairs = (
        membership[:, PAIR_I] * membership[:, PAIR_J]
    )

    return targets, pairs


class IntervalKernelRegressor:
    """
    Kernel-regresija sa pozitivnim težinama svih prethodnih prelaza.

    Za prognozu prelaza t:
        x[t] je poznat kontekst posle poslednjeg poznatog izvlačenja;
        y[:t] sadrži samo već poznate naredne ishode tih konteksta.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.norm = np.sum(
            x.astype(np.float64) ** 2,
            axis=1,
        )

    def distance(self, transition):
        if transition < 1:
            raise ValueError("Nema prethodnih prelaza za regresiju.")

        query = self.x[transition]
        product = self.x[:transition] @ query

        distance = (
            self.norm[:transition]
            + self.norm[transition]
            - 2.0 * product.astype(float)
        )

        return np.maximum(distance, 0.0)

    def predict_many(self, transition, bandwidths):
        distance = self.distance(transition)

        positive = distance[distance > 1e-10]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-10)

        kernels = np.asarray([
            (
                1.0 + distance / (scale * float(multiplier))
            ) ** -2.0
            for multiplier in bandwidths
        ])

        kernels /= kernels.sum(axis=1, keepdims=True)

        prediction = kernels @ self.y[:transition]

        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("Nevažeći izlaz regresije.")

        effective = 1.0 / np.sum(kernels ** 2, axis=1)

        return prediction, effective


def pair_matrix(scores):
    matrix = np.zeros((39, 39), dtype=float)
    matrix[PAIR_I, PAIR_J] = scores
    matrix[PAIR_J, PAIR_I] = scores
    return matrix


def score_combination(combination, matrix):
    indices = np.asarray(combination, dtype=int)
    return float(
        matrix[np.ix_(indices, indices)].sum() / 2.0
    )


def improve_combination(initial, matrix):
    current = initial
    score = score_combination(current, matrix)

    for _ in range(16):
        indices = np.asarray(current, dtype=int)
        marginal = matrix[:, indices].sum(axis=1)

        exchange_scores = (
            score
            + marginal[None, :]
            - matrix[indices, :]
            - marginal[indices, None]
        )

        # Jedino ograničenje zamene: brojevi moraju biti različiti.
        exchange_scores[:, indices] = -np.inf

        flat = int(np.argmax(exchange_scores))
        removed_position, added = np.unravel_index(
            flat, exchange_scores.shape
        )

        candidate = tuple(sorted(
            current[:removed_position]
            + current[removed_position + 1:]
            + (int(added),)
        ))

        candidate_score = score_combination(candidate, matrix)

        if candidate_score <= score + 1e-12:
            break

        current = candidate
        score = candidate_score

    return current, score


def select_combination(scores):
    """
    Deterministički beam + lokalno poboljšanje.
    Nema zaokruživanja brojeva i naknadnih estetskih filtera.
    """
    scores = np.asarray(scores, dtype=float)

    if scores.shape != (len(PAIR_I),):
        raise ValueError("Neispravan broj regresionih skorova.")

    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("Nevažeći skorovi parova.")

    matrix = pair_matrix(scores)

    order = np.lexsort((PAIR_J, PAIR_I, -scores))

    beam = [
        (
            (int(PAIR_I[index]), int(PAIR_J[index])),
            float(scores[index]),
        )
        for index in order[:BEAM_WIDTH]
    ]

    for _ in range(3, 8):
        candidates = {}

        for combination, previous_score in beam:
            gains = matrix[:, combination].sum(axis=1)
            present = set(combination)

            for number in range(39):
                if number in present:
                    continue

                expanded = tuple(sorted(combination + (number,)))
                candidates[expanded] = (
                    previous_score + float(gains[number])
                )

        beam = sorted(
            candidates.items(),
            key=lambda item: (-item[1], item[0]),
        )[:BEAM_WIDTH]

    finalists = [
        improve_combination(combination, matrix)
        for combination, _ in beam[:4]
    ]

    finalists.sort(key=lambda item: (-item[1], item[0]))

    winner, score = finalists[0]
    return tuple(number + 1 for number in winner), score


def longest_run(combination):
    current = maximum = 1

    for previous, number in zip(combination[:-1], combination[1:]):
        current = current + 1 if number == previous + 1 else 1
        maximum = max(maximum, current)

    return maximum


def process_dataset(path, draws, fingerprint):
    print("\n" + "=" * 76)
    print(path.name)
    print(f"Izvlačenja: {len(draws)}")
    print(f"SHA-256: {fingerprint}")
    print("\n1. Kalibracija događaja i Nfer intervalni kontekst", flush=True)

    x, engine, thresholds = build_features(draws)
    targets, y = make_targets(draws)

    count = len(y)
    test_start = count - TEST
    validation_start = test_start - VALIDATION

    if validation_start < CALIBRATION:
        raise ValueError("Nedovoljno istorije posle početne kalibracije.")

    model = IntervalKernelRegressor(x, y)

    print(f"Dimenzija intervalnog konteksta: {x.shape[1]}")
    print(f"Zatvorenih osnovnih intervala: {engine.completed_count}")
    print(f"Izvedenih relacija: {engine.inference_count}")
    print(
        "Opaženih tipova pravila: "
        f"{int(np.sum(engine.rule_available >= 0))}"
    )

    print("\nPragovi događaja, kalibrisani na prvih 128 izvlačenja:")
    for name, limits in zip(CHANNEL_NAMES, thresholds):
        print(f"  {name}: {limits[0]:.6f}, {limits[1]:.6f}")

    print(
        "\nValidacija — CSV redovi: "
        f"{validation_start + 2}..{test_start + 1}"
    )
    print(
        "Završna provera — CSV redovi: "
        f"{test_start + 2}..{len(draws)}"
    )

    print("\n2. Izbor širine kernela na 48 kola", flush=True)

    validation_hits = np.zeros(len(BANDWIDTHS), dtype=float)
    validation_mse = np.zeros(len(BANDWIDTHS), dtype=float)

    for transition in range(validation_start, test_start):
        predictions, _ = model.predict_many(
            transition, BANDWIDTHS
        )

        for index, prediction in enumerate(predictions):
            combination, _ = select_combination(prediction)

            validation_hits[index] += len(
                set(combination).intersection(targets[transition])
            )

            validation_mse[index] += float(np.mean(
                (prediction - y[transition]) ** 2
            ))

    validation_hits /= VALIDATION
    validation_mse /= VALIDATION

    chosen_index = min(
        range(len(BANDWIDTHS)),
        key=lambda index: (
            -validation_hits[index],
            validation_mse[index],
            BANDWIDTHS[index],
        ),
    )

    bandwidth = BANDWIDTHS[chosen_index]

    for index, candidate in enumerate(BANDWIDTHS):
        print(
            f"  Faktor={candidate:g}: "
            f"pogodaka={validation_hits[index]:.4f}, "
            f"MSE={validation_mse[index]:.7f}"
        )

    print(f"Izabrani faktor širine: {bandwidth:g}")

    print("\n3. Odvojena provera poslednja 64 kola", flush=True)

    combinations = []
    hits = []
    control = []
    losses = []
    effective_counts = []

    copied_previous = 0

    for transition in range(test_start, count):
        prediction, effective = model.predict_many(
            transition, (bandwidth,)
        )

        combination, _ = select_combination(prediction[0])
        previous = draws[transition]
        actual = targets[transition]

        combinations.append(combination)
        hits.append(len(set(combination).intersection(actual)))
        control.append(len(set(previous).intersection(actual)))

        losses.append(float(np.mean(
            (prediction[0] - y[transition]) ** 2
        )))
        effective_counts.append(float(effective[0]))

        copied_previous += (
            tuple(combination) == tuple(previous)
        )

        if len(hits) % 8 == 0:
            print(f"  Provera {len(hits)}/{TEST}", flush=True)

    hits = np.asarray(hits)
    control = np.asarray(control)

    print("\nREZULTATI ODVOJENE PROVERE")
    print(f"Prosečno pogodaka: {hits.mean():.4f}")
    print(
        "Raspodela pogodaka 0..7: "
        f"{np.bincount(hits, minlength=8).tolist()}"
    )
    print(f"Kola sa 3+: {int(np.sum(hits >= 3))}")
    print(f"Kola sa 4+: {int(np.sum(hits >= 4))}")
    print(
        "Kontrola: prethodno izvuceno kolo — "
        f"{control.mean():.4f}"
    )
    print(
        "Razlika prema kontroli: "
        f"{hits.mean() - control.mean():+.4f}"
    )
    print(f"MSE regresije odnosa parova: {np.mean(losses):.7f}")
    print(
        "Prosečan efektivni broj ponderisanih primera: "
        f"{np.mean(effective_counts):.2f}"
    )
    print(f"Različite prognoze: {len(set(combinations))}/{TEST}")
    print(f"Kopije prethodnog kola: {copied_previous}")
    print(
        "Prognoze sa 5+ uzastopnih: "
        f"{sum(longest_run(item) >= 5 for item in combinations)}"
    )
    print(
        "Prognoze potpuno u jednoj polovini: "
        f"{sum(max(item) <= 19 or min(item) >= 20 for item in combinations)}"
    )

    print("\n4. Finalni NEXT iz svih istorijskih prelaza", flush=True)

    prediction, effective = model.predict_many(
        count, (bandwidth,)
    )

    combination, score = select_combination(prediction[0])

    print("FINALNI NEXT: " + ", ".join(map(str, combination)))
    print(f"Modelski skor: {score:.8f}")
    print(f"Upotrebljenih istorijskih prelaza: {count}")
    print(f"Efektivni broj ponderisanih primera: {effective[0]:.2f}")
    print(f"Najduži uzastopni niz: {longest_run(combination)}")

    lower = sum(number <= 19 for number in combination)
    print(f"Brojeva 1..19 / 20..39: {lower} / {7 - lower}")
    print(
        "Pretraga je deterministička i heuristička; "
        "nije iscrpno pregledanje svih kombinacija."
    )

    return combination


def main():
    parser = argparse.ArgumentParser(
        description="Nfer intervalni kontekst za Loto 7/39."
    )
    parser.add_argument("csv", nargs="*")
    args = parser.parse_args()

    filenames = args.csv or DEFAULT_FILES

    if len(filenames) != 3:
        raise ValueError("Navedi tačno tri CSV putanje.")

    datasets = [load_csv(filename) for filename in filenames]

    if len({path for path, _, _ in datasets}) != 3:
        raise ValueError("Potrebna su tri različita CSV fajla.")

    print("NFER LOTO V1")
    print("Intervalni operatori + kernel-regresija odnosa parova.")
    print("Bez random postupaka i klasifikatora.")
    print("Tri nezavisna CSV-a; jedna završna kombinacija po CSV-u.")
    print("Ulazni redosled: najstarije → najnovije.")
    print("Program ne menja i ne zapisuje fajlove.")

    results = []

    for path, draws, fingerprint in datasets:
        combination = process_dataset(path, draws, fingerprint)
        results.append((path.name, combination))

    print("\n" + "=" * 76)
    print("ZAVRŠNI NEXT PREGLED")

    for name, combination in results:
        print(f"{name}: " + ", ".join(map(str, combination)))

    grouped = {}

    for name, combination in results:
        grouped.setdefault(combination, []).append(name)

    for combination, names in grouped.items():
        if len(names) > 1:
            print(
                "\nIsti NEXT: "
                + " / ".join(names)
                + ": "
                + ", ".join(map(str, combination))
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrekinuto.", file=sys.stderr)
        sys.exit(130)
    except (
        OSError,
        ValueError,
        FloatingPointError,
        csv.Error,
    ) as exc:
        print(f"\nGREŠKA: {exc}", file=sys.stderr)
        sys.exit(1)



"""
NFER LOTO V1
Intervalni operatori + kernel-regresija odnosa parova.
Bez random postupaka i klasifikatora.
Tri nezavisna CSV-a; jedna završna kombinacija po CSV-u.
Ulazni redosled: najstarije → najnovije.
Program ne menja i ne zapisuje fajlove.

============================================================================
loto7_4704_k83.csv
Izvlačenja: 4704
SHA-256: 1decd748cc0c398e384d9b16fd7e7eb50a5396f59a2f2cf9e7cdd05bdd07ae09

1. Kalibracija događaja i Nfer intervalni kontekst
  Nfer obrađeno: 500/4704
  Nfer obrađeno: 1000/4704
  Nfer obrađeno: 1500/4704
  Nfer obrađeno: 2000/4704
  Nfer obrađeno: 2500/4704
  Nfer obrađeno: 3000/4704
  Nfer obrađeno: 3500/4704
  Nfer obrađeno: 4000/4704
  Nfer obrađeno: 4500/4704
Dimenzija intervalnog konteksta: 3243
Zatvorenih osnovnih intervala: 15185
Izvedenih relacija: 2652956
Opaženih tipova pravila: 1260

Pragovi događaja, kalibrisani na prvih 128 izvlačenja:
  Zbir: 0.481074, 0.570208
  Raspon: 0.736842, 0.868421
  CDF_razdaljina_2: 0.095238, 0.142857
  CDF_razdaljina_8: 0.333333, 0.380952
  CDF_razdaljina_20: 0.666667, 0.809524

Validacija — CSV redovi: 4593..4640
Završna provera — CSV redovi: 4641..4704

2. Izbor širine kernela na 48 kola
  Faktor=0.25: pogodaka=1.2500, MSE=0.0275529
  Faktor=1: pogodaka=1.2708, MSE=0.0275518
  Faktor=4: pogodaka=1.2500, MSE=0.0275515
Izabrani faktor širine: 1

3. Odvojena provera poslednja 64 kola
  Provera 8/64
  Provera 16/64
  Provera 24/64
  Provera 32/64
  Provera 40/64
  Provera 48/64
  Provera 56/64
  Provera 64/64

REZULTATI ODVOJENE PROVERE
Prosečno pogodaka: 1.0781
Raspodela pogodaka 0..7: [18, 26, 17, 3, 0, 0, 0, 0]
Kola sa 3+: 3
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.1406
Razlika prema kontroli: -0.0625
MSE regresije odnosa parova: 0.0275484
Prosečan efektivni broj ponderisanih primera: 4573.90
Različite prognoze: 2/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0

4. Finalni NEXT iz svih istorijskih prelaza
FINALNI NEXT: 8, 11, 22, 23, 26, 33, 34
Modelski skor: 0.68771283
Upotrebljenih istorijskih prelaza: 4703
Efektivni broj ponderisanih primera: 4607.58
Najduži uzastopni niz: 2
Brojeva 1..19 / 20..39: 2 / 5
Pretraga je deterministička i heuristička; nije iscrpno pregledanje svih kombinacija.

============================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974
SHA-256: fc5313380978e4826789413029140d99f09bd92c143892bf9b9027ff9f276be7

1. Kalibracija događaja i Nfer intervalni kontekst
  Nfer obrađeno: 500/2974
  Nfer obrađeno: 1000/2974
  Nfer obrađeno: 1500/2974
  Nfer obrađeno: 2000/2974
  Nfer obrađeno: 2500/2974
Dimenzija intervalnog konteksta: 3243
Zatvorenih osnovnih intervala: 9623
Izvedenih relacija: 1677420
Opaženih tipova pravila: 1258

Pragovi događaja, kalibrisani na prvih 128 izvlačenja:
  Zbir: 0.481074, 0.570208
  Raspon: 0.736842, 0.868421
  CDF_razdaljina_2: 0.095238, 0.142857
  CDF_razdaljina_8: 0.333333, 0.380952
  CDF_razdaljina_20: 0.666667, 0.809524

Validacija — CSV redovi: 2863..2910
Završna provera — CSV redovi: 2911..2974

2. Izbor širine kernela na 48 kola
  Faktor=0.25: pogodaka=1.2917, MSE=0.0275476
  Faktor=1: pogodaka=1.2917, MSE=0.0275482
  Faktor=4: pogodaka=1.2500, MSE=0.0275483
Izabrani faktor širine: 0.25

3. Odvojena provera poslednja 64 kola
  Provera 8/64
  Provera 16/64
  Provera 24/64
  Provera 32/64
  Provera 40/64
  Provera 48/64
  Provera 56/64
  Provera 64/64

REZULTATI ODVOJENE PROVERE
Prosečno pogodaka: 1.1562
Raspodela pogodaka 0..7: [16, 26, 18, 4, 0, 0, 0, 0]
Kola sa 3+: 4
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.3125
Razlika prema kontroli: -0.1562
MSE regresije odnosa parova: 0.0275447
Prosečan efektivni broj ponderisanih primera: 2725.37
Različite prognoze: 38/64
Kopije prethodnog kola: 3
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 4

4. Finalni NEXT iz svih istorijskih prelaza
FINALNI NEXT: 8, 10, 21, 22, 23, 34, 38
Modelski skor: 0.69679747
Upotrebljenih istorijskih prelaza: 2973
Efektivni broj ponderisanih primera: 2760.24
Najduži uzastopni niz: 3
Brojeva 1..19 / 20..39: 2 / 5
Pretraga je deterministička i heuristička; nije iscrpno pregledanje svih kombinacija.

============================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730
SHA-256: 1f4252375372cbbb88cd026892dc19eb5400e6386bfc5dc613ff7d19d9db0504

1. Kalibracija događaja i Nfer intervalni kontekst
  Nfer obrađeno: 500/1730
  Nfer obrađeno: 1000/1730
  Nfer obrađeno: 1500/1730
Dimenzija intervalnog konteksta: 3243
Zatvorenih osnovnih intervala: 5252
Izvedenih relacija: 850870
Opaženih tipova pravila: 1092

Pragovi događaja, kalibrisani na prvih 128 izvlačenja:
  Zbir: 0.466422, 0.549451
  Raspon: 0.763158, 0.894737
  CDF_razdaljina_2: 0.095238, 0.095238
  CDF_razdaljina_8: 0.333333, 0.380952
  CDF_razdaljina_20: 0.714286, 0.809524

Validacija — CSV redovi: 1619..1666
Završna provera — CSV redovi: 1667..1730

2. Izbor širine kernela na 48 kola
  Faktor=0.25: pogodaka=1.2708, MSE=0.0275467
  Faktor=1: pogodaka=1.3542, MSE=0.0275441
  Faktor=4: pogodaka=1.3750, MSE=0.0275429
Izabrani faktor širine: 4

3. Odvojena provera poslednja 64 kola
  Provera 8/64
  Provera 16/64
  Provera 24/64
  Provera 32/64
  Provera 40/64
  Provera 48/64
  Provera 56/64
  Provera 64/64

REZULTATI ODVOJENE PROVERE
Prosečno pogodaka: 1.2188
Raspodela pogodaka 0..7: [12, 30, 18, 4, 0, 0, 0, 0]
Kola sa 3+: 4
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.2812
Razlika prema kontroli: -0.0625
MSE regresije odnosa parova: 0.0275516
Prosečan efektivni broj ponderisanih primera: 1679.63
Različite prognoze: 7/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0

4. Finalni NEXT iz svih istorijskih prelaza
FINALNI NEXT: 7, 8, 11, 23, 27, 35, 37
Modelski skor: 0.71537667
Upotrebljenih istorijskih prelaza: 1729
Efektivni broj ponderisanih primera: 1710.07
Najduži uzastopni niz: 2
Brojeva 1..19 / 20..39: 3 / 4
Pretraga je deterministička i heuristička; nije iscrpno pregledanje svih kombinacija.

============================================================================
ZAVRŠNI NEXT PREGLED
loto7_4704_k83.csv: 8, 11, 22, 23, 26, 33, 34
loto7_4704_k83_loto_2974.csv: 8, 10, 21, 22, 23, 34, 38
loto7_4704_k83_loto_plus_1730.csv: 7, 8, 11, 23, 27, 35, 37
"""



"""
vremenske veze između složenih događaja 

Nfer od jednostavnih događaja gradi intervale i povezuje ih pravilima 
„pre”, „tokom”, „preklapa se”, „počinje” i „završava”. 
Tako opisuje višekoračne obrasce kroz istoriju.
Za moj Loto to bi značilo praćenje redosleda promena distribucije kombinacija, 
njihovog trajanja i međusobnog preklapanja - čuvam upravo taj intervalni mehanizam.

Nfer vremenski operatori i građenje intervalnog konteksta. 
Deterministička kernel-regresija odnosa parova, 
koja taj kontekst povezuje sa narednom kombinacijom. 
To je jasno odvojeno od izvornog Nfer mehanizma: 
https://github.com/rv-tools/nfer
"""
