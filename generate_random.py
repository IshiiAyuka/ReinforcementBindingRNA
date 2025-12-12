import csv
import random
from pathlib import Path
from typing import Iterable, Tuple


NUCLEOTIDES = ("A", "C", "G", "U")


def generate_random_sequence(min_len: int, max_len: int) -> str:
    """Return a random RNA sequence whose length is within [min_len, max_len]."""
    length = random.randint(min_len, max_len)
    return "".join(random.choice(NUCLEOTIDES) for _ in range(length))


def generate_random_sequences(
    count: int, min_len: int = 10, max_len: int = 100
) -> Iterable[Tuple[int, str, int]]:
    """Yield tuples of (id, sequence, length) for the requested number of sequences."""
    for i in range(1, count + 1):
        seq = generate_random_sequence(min_len, max_len)
        yield i, seq


def write_sequences_to_csv(path: Path, rows: Iterable[Tuple[int, str, int]]) -> None:
    """Write sequence rows to a CSV file with header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence"])
        writer.writerows(rows)


if __name__ == "__main__":
    output_path = Path("random_rna.csv")
    rows = list(generate_random_sequences(100, 10, 100))
    write_sequences_to_csv(output_path, rows)
    print(f"Wrote {len(rows)} sequences to {output_path.resolve()}")
