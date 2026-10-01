"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""
# Message-start states (ashhart/TensorFold#169). Prefill keeps states at the chat template's
# message starts (``starts``), a slot keeps two (its shortest, a system block, and its newest), a kept prefix's rows
# are copied into whichever slot a request gets, and a store outside the slots (TENSORFOLD_PREFIX_STORE_GIB) keeps
# system blocks, so a new conversation that shares only its system block resumes after it.

from __future__ import annotations

import os
import time

import numpy as np
import torch

from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import PREFILL_ROWS, WARM_TAIL, Engine, draft, prefill
from .forward import commit, compute, stage
from .mtp import mtp_compute, mtp_stage
from .state import Buffers, State
from ..cuda import CONFIDENCE, DEPTH

PER_SLOT = 2            # kept states a slot holds: its shortest (a system block) and its newest
ALIGN = 256             # bytes: where each of a stored state's tensors starts in the store's buffer
LOG = os.environ.get("TENSORFOLD_RESUME_LOG", "") not in ("", "0")      # one line of numbers per admission
# TENSORFOLD_RESUME_ALIGN=2048 (default 0: off) puts the system block's kept state on the prefill grid below the message start, so a
# resumed prompt's chunks are exactly the stock plan's suffix: equal to a fresh run by construction, for about 1,000 tokens read again
ALIGN_CUTS = int(os.environ.get("TENSORFOLD_RESUME_ALIGN", "0") or 0)


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, pbuf.rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    return e


# -- a kept prefix's rows: in a slot, or stored outside the slots ---------------------------------------------------
def _kv_parts(cache, rows: int) -> list:
    return [(t, rows) for t in ([cache.k, cache.v] + ([cache.ks, cache.vs] if cache.quantized else []))]


def _parts(st, n: int, ratio: int) -> list:
    """(tensor, rows) of every cache a state of ``n`` tokens reads below them: each attention layer's keys and values
    (and their scales), indexer keys and complete pooled blocks, then the MTP head's, which hold ``n - 1`` entries."""

    out = []
    for kv, ikc, pooled in zip(st.kc, st.ikc, st.pooled):
        out += _kv_parts(kv, n) + [(ikc, n), (pooled, n // ratio)]
    if getattr(st, "mtp_kc", None) is not None:
        m = n - 1
        out += _kv_parts(st.mtp_kc, m) + [(st.mtp_ikc, m), (st.mtp_pooled, m // ratio)]
    return out


def copy_prefix(src, dst, n: int, ratio: int) -> None:
    """Rows [0, n) of every attention cache (the MTP head's [0, n - 1)), ``src`` to ``dst``, slots or stored rows alike:
    with the kept snapshot restored over them, ``dst`` resumes the prefix exactly as ``src`` would."""

    a, b = _parts(src, n, ratio), _parts(dst, n, ratio)
    if len(a) != len(b):
        raise RuntimeError("a kept prefix's caches do not match the slot's")
    for (x, rows), (y, _) in zip(a, b):
        if rows:
            y[:rows].copy_(x[:rows])


class _KV:
    """A stored copy of a KV cache's first rows: the attributes ``copy_prefix`` reads."""

    def __init__(self, k, v, ks, vs, quantized: bool) -> None:
        self.k, self.v, self.ks, self.vs, self.quantized = k, v, ks, vs, quantized


def _aligned(nbytes: int) -> int:
    return -(-int(nbytes) // ALIGN) * ALIGN


def _spec(st: State, n: int, ratio: int, snap: dict, tail) -> list[tuple[torch.dtype, tuple[int, ...]]]:
    """The tensors a stored state of ``n`` tokens holds, in order: its rows, then its snapshot's DeltaNet, conv and
    n-gram states and its MTP tail."""

    out = [(t.dtype, (rows, *t.shape[1:])) for t, rows in _parts(st, n, ratio)]
    out += [(snap[k].dtype, tuple(snap[k].shape)) for k in ("rec", "conv", "ple_tail")]
    return out + ([(tail.dtype, tuple(tail.shape))] if tail is not None else [])


def _size(spec) -> int:
    return sum(_aligned(torch.empty((), dtype=d).element_size() * int(np.prod(s))) for d, s in spec)


class Rows:
    """A system block's kept state outside the slots: its attention rows and snapshot, views of the store's buffer."""

    def __init__(self, buf: torch.Tensor, at: int, spec, st: State, tail: bool) -> None:
        views, off = [], at
        for dtype, shape in spec:
            nbytes = torch.empty((), dtype=dtype).element_size() * int(np.prod(shape))
            views.append(buf[off:off + nbytes].view(dtype).view(shape))
            off += _aligned(nbytes)
        self.at, self.nbytes = at, off - at
        it = iter(views)
        quantized = bool(st.kc and st.kc[0].quantized)

        def kv() -> _KV:
            k, v = next(it), next(it)
            ks, vs = (next(it), next(it)) if quantized else (None, None)
            return _KV(k, v, ks, vs, quantized)

        self.kc, self.ikc, self.pooled = [], [], []
        for _ in st.kc:
            self.kc.append(kv())
            self.ikc.append(next(it))
            self.pooled.append(next(it))
        if getattr(st, "mtp_kc", None) is not None:
            self.mtp_kc, self.mtp_ikc, self.mtp_pooled = kv(), next(it), next(it)
        self.rec, self.conv, self.ple_tail = next(it), next(it), next(it)
        self.tail = next(it) if tail else None


class Store:
    """System-block states kept outside the slots, in one buffer allocated at startup: first fit, and the least
    recently used leaves first when a new one needs the room."""

    def __init__(self, nbytes: int, device) -> None:
        self.size = max(0, int(nbytes)) // ALIGN * ALIGN
        # zeros, not empty: the pages are committed at startup (as the slots' are), so the memory the store costs shows in the first
        # reading and the startup estimate, not later as it fills
        self.buf = torch.zeros((self.size,), dtype=torch.uint8, device=device) if self.size else None
        self.entries: list[tuple[list[int], Rows, dict, torch.Tensor | None]] = []      # least recently used first

    def find(self, ids: list[int]):
        return next((e for e in self.entries if len(e[0]) == len(ids) and e[0] == ids), None)

    def touch(self, entry) -> None:
        if any(e is entry for e in self.entries):
            self.entries = [e for e in self.entries if e is not entry] + [entry]

    def _fit(self, need: int) -> int | None:
        start = 0
        for a, b in sorted((e[1].at, e[1].at + e[1].nbytes) for e in self.entries):
            if a - start >= need:
                return start
            start = max(start, b)
        return start if self.size - start >= need else None

    def add(self, ids: list[int], st: State, snap: dict, tail, ratio: int):
        """Store the state of slot ``st`` after ``ids`` (its rows below ``len(ids)`` and ``snap``), unless held."""

        if self.buf is None:
            return None
        held = self.find(ids)
        if held is not None:
            self.touch(held)
            return held
        spec = _spec(st, len(ids), ratio, snap, tail)
        need = _size(spec)
        if need > self.size:
            return None
        at = self._fit(need)
        while at is None and self.entries:
            self.entries.pop(0)
            at = self._fit(need)
        if at is None:
            return None
        rows = Rows(self.buf, at, spec, st, tail is not None)
        copy_prefix(st, rows, len(ids), ratio)
        rows.rec.copy_(snap["rec"])
        rows.conv.copy_(snap["conv"])
        rows.ple_tail.copy_(snap["ple_tail"])
        if tail is not None:
            rows.tail.copy_(tail)
        kept = {**snap, "rec": rows.rec, "conv": rows.conv, "ple_tail": rows.ple_tail,
                "ple_history": None if snap["ple_history"] is None else snap["ple_history"].copy()}
        entry = (list(ids), rows, kept, rows.tail)
        self.entries.append(entry)
        return entry


def _owned(snap: dict, tail):
    """A stored snapshot's copy with tensors of its own (the store's buffer is reused when an entry leaves)."""

    own = {**snap, "rec": snap["rec"].clone(), "conv": snap["conv"].clone(), "ple_tail": snap["ple_tail"].clone(),
           "ple_history": None if snap["ple_history"] is None else snap["ple_history"].copy()}
    return own, None if tail is None else tail.clone()


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", starts=None,
                 store_bytes: int = 0) -> None:
        if w.comm is not None:
            raise ValueError("concurrent Flash Next runs on one GPU for now")
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        self.buf = Buffers(w, rows, capacity)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, PREFILL_ROWS, capacity, prefill=True)
        self.free = [State(w, capacity, depth + 1, kv_dtype) for _ in range(slots)]   # sized by the startup admission
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.streams: dict[int, Stream] = {}
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep
        self.starts = starts          # prompt -> (message starts to keep, the system block's end among them), or None
        self.slots = list(self.free)
        self.used: dict[int, int] = {}                     # each slot's last admission (least recently used goes first)
        self.store = Store(store_bytes if starts is not None else 0, w.device)

    def _busy(self) -> set[int]:
        return {id(s.st) for s in self.streams.values()}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _rebuild(self, entry, st, staying) -> int:
        """Tokens someone would read again to get ``entry`` back once slot ``st`` loses it: its length less the longest prefix
        of it still kept, by another slot, the store, or ``st``'s own states that stay (``staying``)."""

        ids, n = entry[0], len(entry[0])
        store = getattr(self, "store", None)
        keep = 0
        for k in [*(k for k in self.kept if k[1] is not st), *staying, *(store.entries if store is not None else ())]:
            m = len(k[0])
            if keep < m <= n and k[0][-1] == ids[m - 1] and ids[:m] == k[0]:
                keep = m
        return n - keep

    def _slot_for(self, prompt: list[int], reuse: bool):
        """(slot, resume or None, cached tokens). The longest kept state the prompt strictly extends, a slot's or the
        store's, resumes in its own idle slot when that loses nothing; else in a free slot; else in the idle slot whose
        loss costs least: the fewest tokens anyone would read again for the states it drops (each one's length less the
        longest prefix of it kept elsewhere), then no copy, then the least recently used. So a one-off request's leftovers
        go before a running conversation, and a system block nothing else keeps costs its whole length.
        ``resume["copy"]`` names where the prefix's rows come from when not in that slot."""

        busy = self._busy()
        store = getattr(self, "store", None)
        best = None
        for k in [*self.kept, *(store.entries if store is not None else ())] if reuse else []:
            ids = k[0]
            if len(ids) < len(prompt) and (best is None or len(ids) > len(best[0])) and prompt[:len(ids)] == ids:
                best = k
        slots = list(self.free)
        for k in self.kept:
            if id(k[1]) not in busy and all(s is not k[1] for s in slots):
                slots.append(k[1])
        if not slots:
            raise RuntimeError("no free stream slot")
        used = getattr(self, "used", {})
        n = 0 if best is None else len(best[0])
        options = []
        for st in slots:
            mine = [k for k in self.kept if k[1] is st]
            own = None if best is None else next((k for k in mine if len(k[0]) == n and k[0] == best[0]), None)
            lost = [k for k in mine if own is None or len(k[0]) > n]
            source = None
            if best is not None and own is None:     # the same prefix in another slot first, else where it was found
                source = next((k for k in self.kept if k[1] is not st and len(k[0]) == n and k[0] == best[0]), best)
            staying = [k for k in mine if not any(k is x for x in lost)]
            cost = sum(self._rebuild(k, st, staying) for k in lost)
            options.append(((cost, int(source is not None), used.get(id(st), -1)), st, own, source))
        _, st, own, source = min(options, key=lambda o: o[0])
        self.free = [f for f in self.free if f is not st]
        if best is None:
            self._drop_kept(st)
            return st, None, 0
        if own is not None:                          # in place: the states past the resumed one are overwritten
            self.kept = [k for k in self.kept if k[1] is not st or len(k[0]) <= n]
            self._touch(own)
            return st, {"state": own[2], "tail": own[3]}, n
        self._drop_kept(st)
        if store is not None and any(e is source for e in store.entries):
            store.touch(source)
        else:
            self._touch(source)
        return st, {"state": source[2], "tail": source[3], "copy": source[1], "ids": source[0]}, n

    def _touch(self, entry) -> None:
        if any(k is entry for k in self.kept):
            self.kept = [k for k in self.kept if k is not entry] + [entry]

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        """Keep slot ``st``'s state after ``ids``: a slot keeps its shortest and its newest, ``keep`` in all."""

        gone = [k[1] for k in self.kept if k[1] is st and k[0] == ids]
        entry = (ids, st, snap, tail)
        self.kept = [k for k in self.kept if not (k[1] is st and k[0] == ids)] + [entry]
        mine = [k for k in self.kept if k[1] is st]
        if len(mine) > PER_SLOT:
            base = min(mine, key=lambda k: len(k[0]))
            self.kept = [k for k in self.kept if k[1] is not st or k is base or k is entry]
        while len(self.kept) > self.keep:
            gone.append(self.kept.pop(0)[1])
        busy = self._busy()
        for old in gone:           # a displaced idle slot no kept entry holds goes back to the free list
            if old is not st and id(old) not in busy and all(k[1] is not old for k in self.kept) and \
                    all(f is not old for f in self.free):
                self.free.append(old)

    def _points(self, prompt: list[int]) -> tuple[list[int], int | None]:
        """The prompt's message starts and its system block's end among them (none without template markers)."""

        if getattr(self, "starts", None) is None:
            return [], None
        try:
            found, shared = self.starts(prompt)
            return [int(p) for p in found], (None if shared is None else int(shared))
        except Exception:                            # noqa: BLE001  (a prompt the markers cannot read keeps its end)
            return [], None

    def _store_shared(self, prompt: list[int], st: State, shared: int | None) -> None:
        """The store keeps the prompt's system block too, from the slot that now holds its rows."""

        if shared is None or self.store.buf is None:
            return
        ids = prompt[:shared]
        entry = next((k for k in self.kept if k[1] is st and len(k[0]) == shared and k[0] == ids), None)
        if entry is not None:
            self.store.add(ids, st, entry[2], entry[3], self.w.cfg.index_ratio)

    def _name(self, x) -> str:
        slots = getattr(self, "slots", [])
        return next((f"slot {i}" for i, s in enumerate(slots) if s is x), "store" if isinstance(x, Rows) else "?")

    def live(self) -> int:
        return len(self.streams)

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill, its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        s = Stream([0] * min(PREFILL_ROWS + WARM_TAIL, self.capacity - self.depth - 2), 2)
        self.admit(s)
        if not s.done:
            self.round()
        self.streams.pop(s.sid, None)
        self._drop_kept(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Prefill a request in a slot (resuming the longest kept state its prompt extends, keeping its message starts),
        draft its first chain and emit its first token."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        t0 = time.perf_counter()
        prompt = list(s.prompt)
        st, resume, s.cached = self._slot_for(prompt, s.draft)
        if id(st) in self._busy():
            raise RuntimeError("a live stream's slot was chosen for another request")
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        found, shared = self._points(prompt) if s.draft and getattr(s, "vision", None) is None else ([], None)
        if ALIGN_CUTS and shared is not None:         # the system block's state sits on the prefill grid (see ALIGN_CUTS)
            aligned = shared // ALIGN_CUTS * ALIGN_CUTS
            found = [aligned if p == shared else p for p in found]
            found = [p for p in found if p >= MIN_GAP]
            shared = aligned if aligned >= MIN_GAP else None
        stops = [p for p in found if s.cached + MIN_GAP <= p < len(prompt)]

        def keep(p: int, snap: dict, tail) -> None:
            self._remember(prompt[:p], st, {**snap, "start": True, "shared": p == shared}, tail)

        source = None if resume is None else resume.get("copy")
        try:
            if source is not None:                   # the kept prefix's rows are elsewhere: they come over first
                copy_prefix(source, st, s.cached, self.w.cfg.index_ratio)
                snap, tail = (_owned(resume["state"], resume["tail"]) if isinstance(source, Rows)
                              else (resume["state"], resume["tail"]))
                self._remember(list(resume["ids"]), st, snap, tail)
            first = prefill(e, s.prompt, s.sampling, mtp=mtp, resume=resume, stops=stops,
                            keep=keep if stops else None,
                            **({} if s.constraint is None else {"constraint": s.constraint}))
        except Exception:
            self._drop_kept(st)
            if all(f is not st for f in self.free):
                self.free.append(st)
            raise
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        self.used[id(st)] = s.sid
        if s.draft and not (stops and len(prompt) - stops[-1] < MIN_GAP):   # the end, unless a start sits just before it
            # the prompt's state; the MTP head has absorbed every position but the last
            self._remember(prompt, st, st.snapshot(), e.last_streams.clone() if mtp else None)
        self._store_shared(prompt, st, shared)
        if LOG:
            how = "fresh" if resume is None else (f"copied from {self._name(source)}" if source is not None
                                                  else "in place")
            print(f"[tensorfold] resume: request {s.sid} prompt {len(prompt)} cached {s.cached} ({how}) into "
                  f"{self._name(st)}; stops {stops}; {self._name(st)} keeps "
                  f"{[len(k[0]) for k in self.kept if k[1] is st]}; store keeps "
                  f"{[len(k[0]) for k in self.store.entries]}", flush=True)
        s.context = list(s.prompt)
        s.drafts = draft(e, e.last_streams, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                         self.confidence) if mtp and s.count > 1 else []
        s.prefill_s, s.started = time.perf_counter() - t0, time.perf_counter()
        self.streams[s.sid] = s
        s.take([first], self._ends(s))

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams; returns the ones that finished."""

        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return []
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        logits = compute(self.w, segs, self.buf)
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        for s, (_, a0, a1) in zip(live, segs):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid])
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        kept = []
        for s, (_, tokens), (st, a0, a1), rows in zip(live, windows, segs, sampled):
            path, end = accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last])
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        return failed + [s for s in live if s.done]

    def _draft_all(self, streams: list) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.w, self.mbuf, windows)
        logits = mtp_compute(self.w, segs, self.mbuf)
        for (s, _, keep), (st, a0, a1) in zip(todo, segs):
            st.set_mtp_len(st.mtp_len + len(keep))
        active = [(s, a1 - 1) for s, (_, _, a1) in zip([t[0] for t in todo], segs)]
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active])
            nxt = []
            for (s, row), (d, p) in zip(active, picks):
                low = self.confidence > 0 and p < self.confidence
                if low and j > 0:
                    continue
                s.drafts.append(d)
                if not low and j + 1 < room[s.sid]:
                    nxt.append((s, row, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            segs = mtp_stage(self.w, self.mbuf, windows)
            logits = mtp_compute(self.w, segs, self.mbuf)
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        k = min(k, row.shape[1])
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        got = torch.cat([vals, idx.float(), top, col.float(), lse], dim=1).cpu().numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt end is kept stays with it, the rest are free again."""

        for s in done:
            self.streams.pop(s.sid, None)
            if not any(k[1] is s.st for k in self.kept):
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            self.streams.pop(s.sid, None)
            self._drop_kept(s.st)
            self.free.append(s.st)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
