"""Joint row indices: contiguous blocks, with or without replacement.

Without replacement partitions the full history, shuffles those blocks and
uses every row once before constructing the next complete history cycle.
Descriptors are built before horizon streaming, so chunk width cannot alter RNG.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BlockLimits:
    minimum: int = 1
    maximum: int | None = None

    def resolve(self, n, horizon):
        if min(n, horizon) < 1 or self.minimum < 1:
            raise ValueError('Positive history, horizon and block limits required')
        support = min(n, horizon)
        upper = support if self.maximum is None else min(self.maximum, support)
        lower = min(self.minimum, support)
        if upper < lower:
            raise ValueError('Maximum block length is below supported minimum')
        return lower, upper


def partition(n, lower, upper, rng):
    """Chronological nonoverlapping partition; only its final block may be short."""
    if lower == upper:
        starts = np.arange(0, n, lower, dtype=np.int64)
        return starts, np.minimum(lower, n-starts)
    lengths, consumed = [], 0
    while consumed < n:
        count = max(8, int(2*(n-consumed)/((lower+upper)/2))+1)
        draws = rng.integers(lower, upper+1, count)
        cumulative = np.cumsum(draws)
        take = min(count, int(np.searchsorted(cumulative, n-consumed))+1)
        batch = draws[:take].copy()
        if cumulative[take-1] >= n-consumed:
            batch[-1] = n-consumed-int(cumulative[take-2] if take > 1 else 0)
        lengths.extend(batch.tolist())
        consumed += int(batch.sum())
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lengths[:-1])))
    return starts, lengths


class JointBlockStream:
    def __init__(self, n, horizon, paths, seed, *, replacement, limits=BlockLimits()):
        if paths < 1:
            raise ValueError('Positive simulation count required')
        lower, upper = limits.resolve(n, horizon)
        self.descriptors = []
        for child in np.random.SeedSequence(int(seed)).spawn(paths):
            rng = np.random.default_rng(child)
            starts, lengths, remaining = [], [], horizon
            while remaining:
                if replacement:
                    length = int(rng.integers(lower, upper+1))
                    start = int(rng.integers(0, n-length+1))
                    copied = min(length, remaining)
                    starts.append(start); lengths.append(copied)
                    remaining -= copied
                else:
                    ps, pl = partition(n, lower, upper, rng)
                    for index in rng.permutation(len(ps)):
                        copied = min(int(pl[index]), remaining)
                        starts.append(int(ps[index])); lengths.append(copied)
                        remaining -= copied
                        if not remaining:
                            break
            self.descriptors.append((np.asarray(starts), np.asarray(lengths)))
        self.pointer = np.zeros(paths, dtype=np.int64)
        self.within = np.zeros(paths, dtype=np.int64)
        self.position, self.horizon = 0, horizon

    def take(self, width):
        if width < 1 or self.position+width > self.horizon:
            raise ValueError('Requested columns outside remaining forecast')
        result = np.empty((len(self.descriptors), width), dtype=np.int64)
        for p, (starts, lengths) in enumerate(self.descriptors):
            cursor = 0
            while cursor < width:
                block = self.pointer[p]
                count = min(width-cursor, int(lengths[block]-self.within[p]))
                result[p, cursor:cursor+count] = starts[block]+self.within[p]+np.arange(count)
                self.within[p] += count; cursor += count
                if self.within[p] == lengths[block]:
                    self.pointer[p] += 1; self.within[p] = 0
        self.position += width
        return result
