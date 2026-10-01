"""CPU tests of Flash Next message-start states on CUDA (ashhart/TensorFold#169).

Run from the repository root:
    PYTHONPATH=src python -m pytest -q tests/test_flashnext_message_starts.py
No GPU: Triton is faked (as upstream's ``tests/test_cuda_geometry.py`` does) and the kernels are replaced by fakes that
track positions; the row copy and the store run on real CPU tensors.
"""

from __future__ import annotations

import importlib
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

OPEN, ASSIST = 900, 901          # a message's first token; the assistant role token after it
GAP = 256                        # tensorfold.cuda.markers.MIN_GAP


@pytest.fixture
def mods(monkeypatch):
    """The patched modules, imported over a fake Triton (only the kernels' decorators are touched)."""

    lang = ModuleType("triton.language")
    lang.constexpr = object
    triton = ModuleType("triton")
    triton.language = lang
    triton.jit = lambda fn: fn
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    before = set(sys.modules)
    try:
        multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
        decode = importlib.import_module("tensorfold.families.qwen4_exp.cuda.decode")
        engine = importlib.import_module("tensorfold.families.qwen4_exp.cuda.engine")
        kvcache = importlib.import_module("tensorfold.families.qwen4_exp.cuda.kvcache")
        yield SimpleNamespace(multi=multi, decode=decode, engine=engine, kvcache=kvcache)
    finally:
        for name in [n for n in sys.modules if n not in before and n.startswith("tensorfold.families.")]:
            sys.modules.pop(name, None)


# -- prompts -------------------------------------------------------------------------------------------------------
def system(tag: int, n: int) -> list[int]:
    return [OPEN] + [1000 + tag] * (n - 1)


def task(sys_ids: list[int], tag: int, user: int = 400) -> list[int]:
    """A new conversation: the system block, a user message, the generation prompt."""

    return sys_ids + [OPEN] + [5000 + tag] * (user - 1) + [OPEN, ASSIST, 7]


def turn(prev: list[int], tag: int, reply: int = 40, more: int = 300) -> list[int]:
    """The next turn: the previous prompt's history, its reply re-rendered, a tool result, the generation prompt."""

    history = prev[:-3]                                       # the reply replaces the generation prompt
    return history + [OPEN, ASSIST] + [8000 + tag] * reply + [OPEN] + [9000 + tag] * (more - 1) + [OPEN, ASSIST, 7]


# -- 1. the chunk plan ---------------------------------------------------------------------------------------------
def test_chunks_without_cuts_are_the_stock_plan(mods):
    for begin, n, rows in ((0, 5000, 2048), (30726, 35669, 2048), (0, 2048, 2048), (100, 101, 2048)):
        stock = [(s, min(s + rows, n)) for s in range(begin, n, rows)]
        assert list(mods.decode.chunks(begin, n, rows)) == stock


def test_chunks_end_at_every_cut_and_never_exceed_rows(mods):
    got = list(mods.decode.chunks(0, 35669, 2048, [30726, 35662, 0, 35669, 99999]))
    assert got[0][0] == 0 and got[-1][1] == 35669
    assert all(b - a <= 2048 and a < b for a, b in got)
    assert all(x[1] == y[0] for x, y in zip(got, got[1:]))
    ends = {b for _, b in got}
    assert 30726 in ends and 35662 in ends


# -- 2. prefill keeps states at stops, the MTP head one entry short ---------------------------------------------------
class TrackState:
    """A slot's committed counts, and the snapshot fields prefill reads and writes."""

    def __init__(self) -> None:
        self.pos, self.mtp_len, self.mtp_drafted = 0, 0, 0

    def reset(self, w) -> None:
        self.pos, self.mtp_len, self.mtp_drafted = 0, 0, 0

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n

    def snapshot(self) -> dict:
        return {"pos": self.pos, "mtp_len": self.mtp_len - self.mtp_drafted}

    def restore(self, snap: dict) -> None:
        self.pos, self.mtp_len, self.mtp_drafted = snap["pos"], snap["mtp_len"], 0


def tracked_prefill(mods, monkeypatch, prompt, stops, resume=None, rows=2048):
    """Run decode.prefill over fakes: a row's stream carries its position; the MTP head records (position, token)."""

    st = TrackState()
    pairs, chunks = [], []
    pb = SimpleNamespace(streams=torch.zeros((rows + 1, 1)))

    def forward(w, st_, pb_, chunk, logits=True):
        chunks.append((st_.pos, st_.pos + len(chunk)))
        pb_.streams[:len(chunk), 0] = torch.arange(st_.pos, st_.pos + len(chunk), dtype=torch.float32)
        return torch.zeros((1, 4)) if logits else None

    def mtp_forward(w, st_, pb_, nxt, streams):
        for j, tok in enumerate(nxt):
            pairs.append((st_.mtp_len + j, int(streams[j, 0]), tok))
        pb_.streams.fill_(-1.0)                       # the head writes the same scratch, as the real one does

    def commit(w, st_, pb_, R, keep, at=0):
        st_.pos += keep

    monkeypatch.setattr(mods.decode, "forward", forward)
    monkeypatch.setattr(mods.decode, "mtp_forward", mtp_forward)
    monkeypatch.setattr(mods.decode, "commit", commit)
    e = SimpleNamespace(w=SimpleNamespace(mtp=object()), st=st, pbuf=pb, mbuf=object(), prefill_rows=rows,
                        reset=lambda: st.reset(None), sample=lambda last, pos, sampling: [42])
    kept = []
    first = mods.decode.prefill(e, prompt, None, mtp=True, resume=resume, stops=stops,
                                keep=lambda p, snap, tail: kept.append((p, snap, None if tail is None else
                                                                        float(tail[0, 0]))))
    return first, st, pairs, chunks, kept


def test_prefill_keeps_each_stop_with_the_mtp_head_one_short(mods, monkeypatch):
    prompt = list(range(10, 10 + 5000))
    first, st, pairs, chunks, kept = tracked_prefill(mods, monkeypatch, prompt, [1500, 4000])
    assert first == 42 and st.pos == 5000 and st.mtp_len == 4999          # as stock: every position but the last
    assert [c for c in chunks] == [(0, 1500), (1500, 3548), (3548, 4000), (4000, 5000)]
    assert [(p, s["pos"], s["mtp_len"], t) for p, s, t in kept] == [(1500, 1500, 1499, 1499.0),
                                                                    (4000, 4000, 3999, 3999.0)]
    # the head saw (stream j, token j + 1) once for every j below the last position, in order
    assert [(m, int(s), tok) for m, s, tok in pairs] == [(j, j, prompt[j + 1]) for j in range(4999)]


def test_resuming_a_kept_stop_takes_the_last_pair_again_and_matches_a_fresh_run(mods, monkeypatch):
    prompt = list(range(10, 10 + 5000))
    _, _, fresh_pairs, _, kept = tracked_prefill(mods, monkeypatch, prompt, [1500])
    p, snap, tail = kept[0]
    other = prompt[:1500] + [77] * 3000                   # another conversation after the same block
    resume = {"state": snap, "tail": torch.full((1, 1), tail)}
    _, st, pairs, chunks, _ = tracked_prefill(mods, monkeypatch, other, [], resume=resume)
    assert chunks[0] == (1500, 3548) and st.pos == 4500 and st.mtp_len == 4499
    assert pairs[0] == (1499, 1499, 77)                   # (stream 1499, the new prompt's token 1500) at entry 1499
    assert [(m, int(s), tok) for m, s, tok in pairs] == [(j, j, other[j + 1]) for j in range(1499, 4499)]
    assert fresh_pairs[:1499] == [(j, j, prompt[j + 1]) for j in range(1499)]


def test_no_stops_is_the_stock_path(mods, monkeypatch):
    prompt = list(range(10, 10 + 5000))
    _, st, _, chunks, kept = tracked_prefill(mods, monkeypatch, prompt, [])
    assert kept == [] and chunks == [(0, 2048), (2048, 4096), (4096, 5000)]


# -- 3. copying a kept prefix, and the store, on real tensors -------------------------------------------------------
def real_state(mods, capacity: int = 64, dtype: str = "int8", layers: int = 3, seed: int = 0):
    """The caches a slot holds (CPU, small), filled with distinct values."""

    g = torch.Generator().manual_seed(seed)
    KV = mods.kvcache.KVCache

    def fill(t):
        if t.dtype in (torch.int8, torch.uint8):
            t.copy_(torch.randint(0, 100, t.shape, generator=g).to(t.dtype))
        else:
            t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
        return t

    st = SimpleNamespace(kc=[], ikc=[], pooled=[])
    for _ in range(layers):
        kv = KV(capacity, 2, 64, "cpu", dtype)
        for t in (kv.k, kv.v, kv.ks, kv.vs):
            fill(t)
        st.kc.append(kv)
        st.ikc.append(fill(torch.empty((capacity, 32), dtype=torch.bfloat16)))
        st.pooled.append(fill(torch.empty((capacity // 4, 32), dtype=torch.bfloat16)))
    st.mtp_kc = KV(capacity, 2, 64, "cpu", dtype)
    for t in (st.mtp_kc.k, st.mtp_kc.v, st.mtp_kc.ks, st.mtp_kc.vs):
        fill(t)
    st.mtp_ikc = fill(torch.empty((capacity, 32), dtype=torch.bfloat16))
    st.mtp_pooled = fill(torch.empty((capacity // 4, 32), dtype=torch.bfloat16))
    snap = {"pos": 0, "rec": fill(torch.empty((3, 2, 4, 4))), "conv": fill(torch.empty((3, 3, 8), dtype=torch.bfloat16)),
            "ple_tail": fill(torch.empty((9, 16), dtype=torch.bfloat16)), "ple_history": None, "mtp_len": 0}
    return st, snap


def caches(st):
    out = []
    for kv, ikc, pooled in zip(st.kc, st.ikc, st.pooled):
        out.append(("kv", kv.k, kv.v, kv.ks, kv.vs, ikc, pooled))
    out.append(("mtp", st.mtp_kc.k, st.mtp_kc.v, st.mtp_kc.ks, st.mtp_kc.vs, st.mtp_ikc, st.mtp_pooled))
    return out


@pytest.mark.parametrize("dtype", ["int8", "bf16", "int4"])
@pytest.mark.parametrize("n", [1, 2, 5, 31, 37])
def test_copy_prefix_moves_exactly_the_rows_a_state_reads(mods, dtype, n):
    src, _ = real_state(mods, dtype=dtype, seed=1)
    dst, _ = real_state(mods, dtype=dtype, seed=2)
    before = [[t.clone() for t in group[1:]] for group in caches(dst)]
    mods.multi.copy_prefix(src, dst, n, 4)
    for (kind, *a), (_, *b), old in zip(caches(src), caches(dst), before):
        rows = n if kind == "kv" else n - 1
        blocks = rows // 4
        k, v, ks, vs, ikc, pooled = range(6)
        for i, r in ((k, rows), (v, rows), (ikc, rows), (pooled, blocks)):
            assert torch.equal(b[i][:r], a[i][:r]) and torch.equal(b[i][r:], old[i][r:])
        quantized = dtype != "bf16"
        for i in (ks, vs):
            r = rows if quantized else 0
            assert torch.equal(b[i][:r], a[i][:r]) and torch.equal(b[i][r:], old[i][r:])


def test_store_keeps_rows_and_snapshot_and_restores_them(mods):
    multi = mods.multi
    src, snap = real_state(mods, seed=3)
    snap = {**snap, "pos": 37, "mtp_len": 36, "start": True, "shared": True}
    tail = torch.randn((1, 16)).to(torch.bfloat16)
    store = multi.Store(4 << 20, "cpu")
    ids = list(range(37))
    entry = store.add(ids, src, snap, tail, 4)
    assert entry is not None and store.find(ids) is entry and isinstance(entry[1], multi.Rows)
    assert store.add(ids, src, snap, tail, 4) is entry               # held already: no second copy
    for name in ("rec", "conv", "ple_tail"):
        assert torch.equal(entry[2][name], snap[name]) and entry[2][name].data_ptr() != snap[name].data_ptr()
    assert entry[2]["pos"] == 37 and entry[2]["mtp_len"] == 36 and entry[2]["shared"]
    assert torch.equal(entry[3], tail)
    src_rows = [[t.clone() for t in group[1:]] for group in caches(src)]
    for group in caches(src):                            # the slot moves on; the store's copy must not
        for t in group[1:]:
            t.zero_()
    dst, _ = real_state(mods, seed=4)
    multi.copy_prefix(entry[1], dst, 37, 4)
    for (kind, *b), saved in zip(caches(dst), src_rows):
        rows = 37 if kind == "kv" else 36
        for t, s, r in zip(b, saved, (rows, rows, rows, rows, rows, rows // 4)):
            assert torch.equal(t[:r], s[:r])
    own, own_tail = multi._owned(entry[2], entry[3])
    assert own["rec"].data_ptr() != entry[2]["rec"].data_ptr() and torch.equal(own["rec"], entry[2]["rec"])
    assert torch.equal(own_tail, tail)


def test_store_evicts_least_recently_used_and_never_overflows(mods):
    multi = mods.multi
    src, snap = real_state(mods, capacity=256, seed=5)
    one = multi._size(multi._spec(src, 200, 4, snap, None))
    store = multi.Store(3 * one + 100, "cpu")
    adds = [store.add([k] * 200, src, snap, None, 4) for k in range(3)]
    assert all(a is not None for a in adds) and len(store.entries) == 3
    store.touch(adds[0])                                 # 1 is now the least recently used
    fourth = store.add([9] * 200, src, snap, None, 4)
    assert fourth is not None and [e[0][0] for e in store.entries] == [2, 0, 9]
    spans = sorted((e[1].at, e[1].at + e[1].nbytes) for e in store.entries)
    assert all(b <= c for (_, b), (c, _) in zip(spans, spans[1:])) and spans[-1][1] <= store.size
    assert store.add([5] * 250, src, snap, None, 4) is not None      # a bigger one: evicts until a gap fits
    assert multi.Store(0, "cpu").add([1] * 10, src, snap, None, 4) is None
    assert multi.Store(one - 1, "cpu").add([1] * 200, src, snap, None, 4) is None


# -- 4. which slot, and from where: admissions over fake slots --------------------------------------------------------
class Slot:
    """A fake slot: what its caches hold (``seq``) and a snapshot of its position."""

    def __init__(self, name: str) -> None:
        self.name, self.pos, self.seq = name, 0, []

    def snapshot(self) -> dict:
        return {"pos": self.pos, "rec": torch.zeros(1), "conv": torch.zeros(1), "ple_tail": torch.zeros(1),
                "ple_history": None, "mtp_len": self.pos - 1}


class FakeRows:
    def __init__(self, seq) -> None:
        self.seq = list(seq)


class Harness:
    """A MultiDecoder over fake slots; prefill, drafting and the row copy are replaced by fakes that check every resume
    against what the slot (or the copied source) really holds."""

    def __init__(self, mods, monkeypatch, slots: int = 2, store_entries: int = 0) -> None:
        multi = mods.multi
        self.multi, self.log = multi, []
        d = multi.MultiDecoder.__new__(multi.MultiDecoder)
        d.w = SimpleNamespace(cfg=SimpleNamespace(index_ratio=4))
        d.depth, d.confidence, d.capacity, d.eos = 6, 0.6, 10**6, ()
        d.buf = d.mbuf = d.pbuf = None
        d.free = [Slot(f"s{i}") for i in range(slots)]
        d.slots = list(d.free)
        d.streams, d.next_id, d.kept, d.keep, d.used = {}, 0, [], 8, {}
        d.starts = mods.engine.starts_from((OPEN,), (OPEN, ASSIST))
        d.store = FakeStore(store_entries, self)
        self.d = d
        monkeypatch.setattr(multi, "_slot", lambda w, st, *a: SimpleNamespace(st=st, last_streams=torch.zeros((1, 4))))
        monkeypatch.setattr(multi, "draft", lambda *a, **k: [])
        monkeypatch.setattr(multi, "prefill", self.prefill)
        monkeypatch.setattr(multi, "copy_prefix", self.copy)
        monkeypatch.setattr(multi, "Rows", FakeRows)

    def copy(self, src, dst, n, ratio) -> None:
        self.log.append(("copy", getattr(src, "name", "store"), dst.name, n))
        assert id(dst) not in self.d._busy()
        dst.seq = list(src.seq[:n])                      # the rows that come over

    def prefill(self, e, prompt, sampling, *, mtp, resume, stops, keep, **kw):
        st = e.st
        begin = 0 if resume is None else resume["state"]["pos"]
        assert st.seq[:begin] == list(prompt[:begin]), "resumed from rows the slot does not hold"
        st.seq = list(prompt[:begin])
        for p in stops:
            if begin < p < len(prompt):
                st.seq, st.pos = list(prompt[:p]), p
                keep(p, st.snapshot(), torch.zeros((1, 4)))
        st.seq, st.pos = list(prompt), len(prompt)
        e.last_streams = torch.zeros((1, 4))
        return 7

    def submit(self, prompt, *, draft: bool = True, busy: bool = False):
        """Admit a request; it finishes at once unless ``busy`` (it then decodes until ``done``)."""

        s = self.multi.Stream(list(prompt), 50 if busy else 1, None, draft=draft)
        self.d.admit(s)
        if not busy:
            assert s.done
            self.finish(s)
        return s

    def finish(self, s) -> None:
        s.st.seq = list(s.st.seq) + [-1] * 5            # its reply's rows, past the prompt
        self.d.finish([s])
        self.check()

    def check(self) -> None:
        d = self.d
        busy = d._busy()
        held = {id(k[1]) for k in d.kept}
        ids = [id(f) for f in d.free]
        assert len(ids) == len(set(ids)), "a slot twice in the free list"
        for st in d.slots:
            places = (id(st) in ids) + (id(st) in held or id(st) in busy)
            assert places == 1, f"{st.name} is free and held/busy, or neither"
        for k in d.kept:
            assert k[1].seq[:len(k[0])] == k[0], "a kept state its slot no longer holds"
        for st in d.slots:
            assert sum(1 for k in d.kept if k[1] is st) <= 2
        assert len(d.kept) <= d.keep
        stored = {id(e[2][name]) for e in d.store.entries for name in ("rec", "conv", "ple_tail")}
        assert not any(id(k[2][name]) in stored for k in d.kept for name in ("rec", "conv", "ple_tail")
                       if name in k[2]), "a slot's state shares the store's buffer, which is reused"


class FakeStore:
    """The store's bookkeeping with whole prompts in place of rows; ``size`` entries at most (0: off)."""

    def __init__(self, size: int, h: Harness) -> None:
        self.size, self.h, self.entries = size, h, []
        self.buf = object() if size else None

    def find(self, ids):
        return next((e for e in self.entries if e[0] == ids), None)

    def touch(self, entry) -> None:
        self.entries = [e for e in self.entries if e is not entry] + [entry]

    def add(self, ids, st, snap, tail, ratio):
        held = self.find(ids)
        if held is not None:
            self.touch(held)
            return held
        assert st.seq[:len(ids)] == ids, "stored rows the slot does not hold"
        if len(self.entries) >= self.size:
            self.entries.pop(0)
        own = {**snap, **{k: snap[k].clone() for k in ("rec", "conv", "ple_tail")}}    # the real store copies too
        entry = (list(ids), FakeRows(ids), own, tail)
        self.entries.append(entry)
        return entry


def test_new_tasks_on_a_clean_cache_resume_after_the_shared_block(mods, monkeypatch):
    h = Harness(mods, monkeypatch)
    X = system(1, 3000)
    got = [h.submit(task(X, i)).cached for i in range(4)]
    assert got == [0, 3000, 3000, 3000]
    assert ("copy", "s0", "s1", 3000) in h.log                    # the second task: into the free slot, from s0
    t1 = task(X, 10)
    assert h.submit(t1).cached == 3000
    t2 = turn(t1, 10)
    assert h.submit(t2).cached == len(t1) - 3                     # the next turn: at its reply's start (27B rule)


def test_two_interleaving_conversations_then_new_tasks(mods, monkeypatch):
    h = Harness(mods, monkeypatch)
    X = system(1, 3000)
    h.submit(task(X, 0))                                          # warm: the block is kept
    a, b = task(X, 1), task(X, 2)
    assert h.submit(a).cached == 3000 and h.submit(b).cached == 3000
    for k in range(2):
        a2, b2 = turn(a, 10 + k), turn(b, 20 + k)
        assert h.submit(a2).cached == len(a) - 3                  # each conversation keeps its own slot
        assert h.submit(b2).cached == len(b) - 3
        a, b = a2, b2
    assert [h.submit(task(X, 30 + i)).cached for i in range(4)] == [3000] * 4


def test_two_system_blocks_each_keep_a_slot(mods, monkeypatch):
    h = Harness(mods, monkeypatch)
    X, Y = system(1, 3000), system(2, 2000)
    got = [h.submit(task(B, i)).cached for i, B in enumerate([X, Y, X, Y, X, X, Y, Y])]
    assert got == [0, 0, 3000, 2000, 3000, 3000, 2000, 2000]


def test_three_blocks_on_two_slots_need_the_store(mods, monkeypatch):
    blocks = [system(1, 3000), system(2, 2000), system(3, 2500)]
    order = [0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0]
    without = Harness(mods, monkeypatch, store_entries=0)
    lost = sum(without.submit(task(blocks[b], i)).cached == 0 for i, b in enumerate(order))
    assert lost > 3                                               # two slots cannot hold three blocks
    with_store = Harness(mods, monkeypatch, store_entries=4)
    got = [with_store.submit(task(blocks[b], i)).cached for i, b in enumerate(order)]
    assert got[:3] == [0, 0, 0] and got[3:] == [len(blocks[b]) for b in order[3:]]
    assert any(entry[1] == "store" for entry in with_store.log)


def test_a_new_task_copies_its_block_from_a_busy_slot(mods, monkeypatch):
    h = Harness(mods, monkeypatch)
    X = system(1, 3000)
    long = h.submit(task(X, 0), busy=True)                        # decoding in s0, holding the block
    s = h.submit(task(X, 1))
    assert s.cached == 3000 and ("copy", "s0", "s1", 3000) in h.log
    h.finish(long)


def test_serial_requests_neither_resume_nor_keep(mods, monkeypatch):
    h = Harness(mods, monkeypatch)
    X = system(1, 3000)
    h.submit(task(X, 0))
    s = h.submit(task(X, 1), draft=False)
    assert s.cached == 0 and not any(k[1] is s.st for k in h.d.kept)
    assert h.submit(task(X, 2)).cached == 3000                    # the block survived in the other slot


def test_short_prompts_and_markerless_prompts_keep_their_end_as_stock(mods, monkeypatch):
    h = Harness(mods, monkeypatch, store_entries=2)
    plain = list(range(1, 200))
    s = h.submit(plain)
    assert [len(k[0]) for k in h.d.kept] == [len(plain)]          # the prompt end, as stock keeps it
    assert h.submit(plain + [5, 6]).cached == len(plain)
    long_user = [OPEN] + [3] * 3000 + [OPEN, ASSIST, 7]           # no system block: no shared state
    h.submit(long_user)
    assert h.d.store.entries == []


def test_in_place_resume_drops_the_states_past_it_at_once(mods, monkeypatch):
    h = Harness(mods, monkeypatch, slots=1)
    X = system(1, 3000)
    first = task(X, 0)
    h.submit(first)
    assert sorted(len(k[0]) for k in h.d.kept) == [3000, len(first) - 3]
    st, resume, cached = h.d._slot_for(task(X, 1), True)
    assert cached == 3000 and "copy" not in resume
    assert [len(k[0]) for k in h.d.kept if k[1] is st] == [3000]     # the old reply's start goes before any prefill


def test_a_system_block_nothing_else_keeps_is_spared(mods, monkeypatch):
    multi = mods.multi
    h = Harness(mods, monkeypatch, slots=2, store_entries=4)
    d, (a, b) = h.d, h.d.slots
    Y, Z = system(2, 2000), system(3, 2500)
    shared = {"shared": True, "start": True, "rec": torch.zeros(1), "conv": torch.zeros(1), "ple_tail": torch.zeros(1)}
    d.free, d.used = [], {id(a): 5, id(b): 9}               # b is the more recently used
    d.kept = [(Y, a, dict(shared), None), (Z, b, dict(shared), None), (Z + [1] * 300, b, {}, None)]
    d.store.entries = [(Z, FakeRows(Z), dict(shared), None)]
    st, resume, cached = d._slot_for(task(system(4, 1000), 0), True)       # a block nothing keeps: fresh
    assert st is b and resume is None and cached == 0      # b loses two states the store and a reply cover, a its only Y
    assert multi.MIN_GAP == GAP


def turn38(prev, tag, more):
    """A next turn the way Qwen3.8's template renders it: the history keeps the previous prompt whole, then the reply, a tool result of ``more`` tokens and the generation prompt."""

    return list(prev) + [8000 + tag] * 40 + [OPEN] + [9000 + tag] * (more - 1) + [OPEN, ASSIST, 7]


def two_long_conversations(h):
    X = system(1, 3000)
    A = turn38(task(X, 1), 11, 1900)                              # scaled: about 50k against a 30.7k block
    B = turn38(task(X, 2), 12, 11900)                             # about 150k
    for p in (task(X, 1), A, task(X, 2), B):
        h.submit(p)
    return A, B


def test_an_interruption_costs_one_reread_not_a_cascade(mods, monkeypatch):
    """Two long conversations resident in the two slots; a new task with another system block comes between their turns. The
    slot it takes, and the slot the displaced conversation then takes, are the ones whose loss costs the fewest tokens to read
    again: the interrupter's leftovers go, the long conversation stays resident, rep after rep."""

    h = Harness(mods, monkeypatch, slots=2, store_entries=6)
    A, B = two_long_conversations(h)
    for rep in range(3):
        A, B = turn38(A, 100 + rep, 50), turn38(B, 200 + rep, 50)
        pa, pb = len(A), len(B)
        assert h.submit(A).cached >= pa - 100 and h.submit(B).cached >= pb - 100      # both resident
        h.submit(task(system(9, 2300), 50 + rep))                                         # the interrupter
        A, B = turn38(A, 300 + rep, 50), turn38(B, 400 + rep, 50)
        assert h.submit(A).cached == 3000                         # the cheaper one reads its own part once, from the block
        assert h.submit(B).cached == pb                           # the long one never left its slot


def test_v1_slot_rule_cascades_in_the_same_scenario(mods, monkeypatch):
    """The first version of the rule (fewest states lost, then a block nothing else keeps): kept here as a recorded CPU
    simulation of why it was replaced. One interruption and both conversations re-read from the block on every turn."""

    h = Harness(mods, monkeypatch, slots=2, store_entries=6)

    def v1_cost(st, own, lost, source):
        alone = sum(1 for k in lost if k[2].get("shared")
                    and not any(x[1] is not st and x[0] == k[0] for x in h.d.kept)
                    and not any(e[0] == k[0] for e in h.d.store.entries))
        return (alone, len(lost))

    real = mods.multi.MultiDecoder._rebuild
    monkeypatch.setattr(mods.multi.MultiDecoder, "_rebuild", lambda self, entry, st, staying: 1)   # each lost state costs 1
    A, B = two_long_conversations(h)
    falls = 0
    for rep in range(3):
        A, B = turn38(A, 100 + rep, 50), turn38(B, 200 + rep, 50)
        h.submit(A); h.submit(B)
        h.submit(task(system(9, 2300), 50 + rep))
        A, B = turn38(A, 300 + rep, 50), turn38(B, 400 + rep, 50)
        falls += (h.submit(A).cached == 3000) + (h.submit(B).cached == 3000)
    assert falls == 6                                             # both, every rep
    assert real is not None and v1_cost is not None


def test_without_the_store_a_slots_own_block_counts_as_kept(mods, monkeypatch):
    """Store off: resuming in place at the slot's block loses only the conversation's own part (the block stays in that slot),
    which is cheaper than destroying another system block's only copy."""

    h = Harness(mods, monkeypatch, slots=2, store_entries=0)
    X, W = system(1, 3000), system(2, 2000)
    A = turn38(task(X, 1), 11, 1000)
    for p in (task(X, 1), A, task(W, 2)):
        h.submit(p)
    s = h.submit(task(X, 3))
    assert s.cached == 3000                                       # in place at X in A's slot
    assert any(len(k[0]) == 2000 and k[0] == W for k in h.d.kept)  # W's only copy survived


def test_without_the_store_a_block_another_slot_keeps_is_free_to_lose(mods, monkeypatch):
    """Store off, three slots: the cheapest victim is the conversation whose block another slot also keeps, not the only copy of
    a third system block."""

    h = Harness(mods, monkeypatch, slots=3, store_entries=0)
    X, W = system(1, 3000), system(2, 2000)
    A = turn38(task(X, 1), 11, 500)
    B = turn38(task(X, 2), 12, 5000)
    for p in (task(X, 1), A, task(X, 2), B, task(W, 3)):
        h.submit(p)
    h.submit(task(system(5, 2500), 4))                            # a new block: someone's slot must go
    assert any(len(k[0]) == 2000 and k[0] == W for k in h.d.kept)  # W's only copy survived
    assert any(len(k[0]) > 8000 for k in h.d.kept)                # so did B, the long one


def test_a_finished_long_conversation_holds_its_slot(mods, monkeypatch):
    """DOCUMENTED, not fixed (follow-up: a recency term in the victim choice). A long conversation that has finished still costs
    many tokens to read again, so while new tasks keep coming, they all take the other slot and the finished one keeps its slot."""

    h = Harness(mods, monkeypatch, slots=2, store_entries=6)
    A, B = two_long_conversations(h)                              # B (long) now finishes: no more turns
    slots_used = []
    for i in range(6):
        s = h.submit(task(system(10 + i % 2, 2000), 60 + i))          # new tasks with two other setups
        slots_used.append(h.d._name(s.st))
    held_by_b = [h.d._name(k[1]) for k in h.d.kept if len(k[0]) > 10000 and B[:len(k[0])] == k[0]]   # B's kept state
    assert held_by_b and all(name != held_by_b[0] for name in slots_used)      # B's slot was never taken


def test_a_probe_order_forces_every_resume_path(mods, monkeypatch):
    """A token-identity probe's order at measured prompt lengths (two system blocks, one also with thinking on): fresh, a copy from an
    idle slot, in place, a next turn in place, a copy from the slot B1 is still decoding in (B0 holds the other block's slot
    while B1 is admitted), and a restore from the store. Without B0 this rule puts B1 in the other slot, A4 resumes in place,
    and no copy from a busy slot happens."""

    h = Harness(mods, monkeypatch, slots=2, store_entries=4)
    X, Xt, Y = system(1, 30726), system(2, 30764), system(3, 23641)

    def new(block, tag, total):
        return task(block, tag, total - len(block) - 3)

    A1, A2, A3, A4 = new(X, 1, 35732), new(X, 2, 35707), new(X, 3, 35699), new(X, 4, 35711)
    B1, T1, T2 = new(X, 5, 35701), new(Xt, 6, 35733), new(Xt, 7, 35768)
    Y1, Y2 = new(Y, 8, 28771), new(Y, 9, 28763)
    for p in (A1, A2, A3, A4, T1, T2, Y1, Y2):                   # the serial references first
        assert h.submit(p, draft=False).cached == 0
    assert h.d.kept == [] and h.d.store.entries == []

    def go(p, **kw):
        n = len(h.log)
        s = h.submit(p, **kw)
        copies = [e for e in h.log[n:] if e[0] == "copy"]
        how = "fresh" if s.cached == 0 else (f"copy {copies[0][1]}" if copies else "in place")
        return s, how, h.d._name(s.st)

    assert go(A1)[1] == "fresh"
    _, how, _ = go(A2)
    assert how.startswith("copy s")
    assert go(A3)[1] == "in place"
    A5 = turn38(A3, 10, 256)
    assert go(A5)[1] == "in place" and h.submit(A5, draft=False).cached == 0     # the next turn, then its reference
    assert go(Y1)[1] == "fresh"
    b0, how, b0_slot = go(turn38(Y1, 11, 300), busy=True)          # Y1's next turn, decoding: holds the second block's slot
    assert how == "in place"
    b1, how, b1_slot = go(B1, busy=True)                           # the only idle slot holds the first block
    assert how == "in place" and b1_slot != b0_slot
    h.finish(b0)
    s, how, slot = go(A4)                                          # B1 still decoding
    assert how == f"copy {b1_slot.replace('slot ', 's')}" and slot == b0_slot and id(b1.st) in h.d._busy()
    h.finish(b1)
    assert go(T1)[1] == "fresh"
    assert go(T2)[1] in ("in place", "copy s0", "copy s1")
    assert go(Y2)[1] == "copy store"


def test_aligned_system_block_state_sits_on_the_prefill_grid(mods, monkeypatch):
    monkeypatch.setattr(mods.multi, "ALIGN_CUTS", 1024)
    h = Harness(mods, monkeypatch, store_entries=2)
    X = system(1, 3000)
    first = task(X, 0)
    h.submit(first)
    lens = sorted(len(k[0]) for k in h.d.kept)
    assert 2048 in lens and 3000 not in lens                      # 3000 floored to the 1024 grid
    assert [len(e[0]) for e in h.d.store.entries] == [2048]       # the store keeps the aligned block too
    assert h.submit(task(X, 1)).cached == 2048                    # a new task resumes there
    t2 = turn(first, 5)
    assert h.submit(t2).cached == len(first) - 3                  # a next turn still resumes at its generation start


def test_alignment_below_the_minimum_gap_keeps_no_system_state(mods, monkeypatch):
    monkeypatch.setattr(mods.multi, "ALIGN_CUTS", 1024)
    h = Harness(mods, monkeypatch, store_entries=2)
    X = system(1, 600)                                            # floors to 0 on the grid: no state worth keeping
    h.submit(task(X, 0))
    assert h.d.store.entries == [] and all(not k[2].get("shared") for k in h.d.kept)
    assert h.submit(task(X, 1)).cached == 0


def test_alignment_off_is_the_default(mods):
    assert mods.multi.ALIGN_CUTS == 0


def test_the_upstream_slot_tests_still_hold(mods):
    multi = mods.multi
    old, fresh = object(), object()
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.streams, dec.free, dec.kept, dec.keep = {}, [fresh], [([1, 2, 3], old, {}, None)], 8
    chosen, resume, cached = dec._slot_for([1, 2, 3], True)
    assert chosen is fresh and resume is None and cached == 0
    dec._remember([1, 2, 3], chosen, {}, None)
    dec.streams = {0: SimpleNamespace(st=chosen)}
    other, _, _ = dec._slot_for([9, 9], True)
    assert other is old
    busy = object()
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.streams, dec.free, dec.kept, dec.keep = {0: SimpleNamespace(st=busy)}, [], [([1], old, {}, None),
                                                                                ([2], old, {}, None),
                                                                                ([3], busy, {}, None)], 8
    dec._remember([1], fresh, {}, None)
    assert old not in dec.free
    dec._remember([3], fresh, {}, None)
    assert busy not in dec.free


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("slots,store", [(2, 0), (2, 3), (3, 0), (5, 4)])
def test_random_traffic_keeps_every_invariant(mods, monkeypatch, seed, slots, store):
    """New tasks over several blocks, next turns, serial requests and overlapping streams: no slot is handed out twice,
    nothing resumes from rows its slot does not hold (the fakes assert it), and each block resumes once seen."""

    rng = random.Random(seed)
    h = Harness(mods, monkeypatch, slots=slots, store_entries=store)
    blocks = [system(b, rng.choice([600, 1500, 3000])) for b in range(4)]
    talks, live, n = [], [], 0
    for _ in range(60):
        n += 1
        op = rng.random()
        if live and (op < 0.25 or len(live) >= slots):
            h.finish(live.pop(rng.randrange(len(live))))
            continue
        busy = rng.random() < 0.3
        if op < 0.55 or not talks:
            prompt = task(rng.choice(blocks), n, user=rng.choice([300, 600]))
            talks.append(prompt)
        elif op < 0.85:
            i = rng.randrange(len(talks))
            prompt = talks[i] = turn(talks[i], n, more=rng.choice([100, 400]))
        else:
            prompt = task(rng.choice(blocks), n)
            s = h.submit(prompt, draft=False, busy=busy)
            if busy:
                live.append(s)
            continue
        s = h.submit(prompt, busy=busy)
        if busy:
            live.append(s)
        h.check()
    for s in live:
        h.finish(s)


# -- 5. the engine side: markers, admission bytes, the store setting -------------------------------------------------
def test_starts_name_the_system_block_only_when_one_precedes(mods):
    starts = mods.engine.starts_from((OPEN,), (OPEN, ASSIST))
    X = system(1, 3000)
    found, shared = starts(task(X, 1))
    assert found == [3000, 3000 + 400] and shared == 3000
    assert starts([OPEN] + [3] * 3000 + [OPEN, ASSIST, 7]) == ([3001], None)   # a lone user message: no block
    assert starts(list(range(1, 100))) == ([], None)


def test_start_snapshots_count_the_states_two_a_slot_adds(mods):
    t = {"layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12, "hidden_size": 2560,
         "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "linear_num_key_heads": 16,
         "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
         "linear_conv_kernel_dim": 4, "hc_count": 4, "ple_conv_kernel_size": 4, "ngram_size": 3}
    one = 36 * 48 * 128 * 128 * 4 + 36 * 3 * (2 * 16 * 128 + 48 * 128) * 2 + 3 * 3 * 4 * 2560 * 2
    assert mods.engine._start_snapshots(t, 2) == 2 * one          # --parallel 2: two more, about 220 MiB
    assert mods.engine._start_snapshots(t, 5) == 3 * one          # KEEP 8 bounds it
    assert mods.engine._start_snapshots(t, 1) == 1 * one


def test_store_setting(mods, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_PREFIX_STORE_GIB", raising=False)
    assert mods.engine.store_bytes() == 0
    monkeypatch.setenv("TENSORFOLD_PREFIX_STORE_GIB", "1.5")
    assert mods.engine.store_bytes() == int(1.5 * 2**30)
    monkeypatch.setenv("TENSORFOLD_PREFIX_STORE_GIB", "lots")
    with pytest.raises(ValueError):
        mods.engine.store_bytes()
