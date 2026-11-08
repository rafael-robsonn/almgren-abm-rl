use crate::types::{AgentId, Fill, OrderId, Price, Qty, RejectReason, Side, Timestamp};

const NIL: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Pending,
    Resting,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u32,
    state: SlotState,
    owner: AgentId,
    side: Side,
    price: Price,
    qty: Qty,
    prev: u32,
    next: u32,
}

#[derive(Clone, Copy, Debug)]
struct Level {
    head: u32,
    tail: u32,
    qty: Qty,
    orders: u32,
}

impl Level {
    const EMPTY: Level = Level {
        head: NIL,
        tail: NIL,
        qty: 0,
        orders: 0,
    };
}

#[derive(Clone, Debug)]
struct Ladder {
    side: Side,
    levels: Vec<Level>,
    bits: Vec<u64>,
    best: Option<Price>,
    qty: Qty,
    orders: u64,
}

impl Ladder {
    fn new(side: Side, n_ticks: usize) -> Self {
        Self {
            side,
            levels: vec![Level::EMPTY; n_ticks],
            bits: vec![0; n_ticks.div_ceil(64)],
            best: None,
            qty: 0,
            orders: 0,
        }
    }

    #[inline]
    fn mark(&mut self, p: usize) {
        self.bits[p >> 6] |= 1u64 << (p & 63);
    }

    #[inline]
    fn unmark(&mut self, p: usize) {
        self.bits[p >> 6] &= !(1u64 << (p & 63));
    }

    fn first_at_or_above(&self, p: usize) -> Option<usize> {
        let mut w = p >> 6;
        if w >= self.bits.len() {
            return None;
        }
        let mut word = self.bits[w] & (!0u64 << (p & 63));
        loop {
            if word != 0 {
                return Some((w << 6) + word.trailing_zeros() as usize);
            }
            w += 1;
            if w >= self.bits.len() {
                return None;
            }
            word = self.bits[w];
        }
    }

    fn first_at_or_below(&self, p: usize) -> Option<usize> {
        let mut w = p >> 6;
        let mut word = self.bits[w] & (!0u64 >> (63 - (p & 63)));
        loop {
            if word != 0 {
                return Some((w << 6) + 63 - word.leading_zeros() as usize);
            }
            if w == 0 {
                return None;
            }
            w -= 1;
            word = self.bits[w];
        }
    }

    #[inline]
    fn better(&self, a: Price, b: Price) -> bool {
        match self.side {
            Side::Buy => a > b,
            Side::Sell => a < b,
        }
    }

    #[inline]
    fn worse_than(&self, p: Price) -> Option<Price> {
        match self.side {
            Side::Buy if p > 0 => self.first_at_or_below(p as usize - 1).map(|x| x as Price),
            Side::Buy => None,
            Side::Sell => self.first_at_or_above(p as usize + 1).map(|x| x as Price),
        }
    }

    #[inline]
    fn level_emptied(&mut self, p: Price) {
        self.unmark(p as usize);
        if self.best == Some(p) {
            self.best = self.worse_than(p);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderInfo {
    pub id: OrderId,
    pub owner: AgentId,
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimitOutcome {
    pub filled: Qty,
    pub resting: Qty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CancelOutcome {
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModifyOutcome {
    pub side: Side,
    pub old_price: Price,
    pub old_qty: Qty,
    pub requeued: bool,
    pub filled: Qty,
    pub resting: Qty,
}

/// Price-time priority limit order book over the integer tick range `[0, n_ticks)`.
///
/// Invariants (checked by [`OrderBook::check_invariants`]): every occupied level is FIFO,
/// level and side aggregates equal the sum of resting quantities, the occupancy bitset marks
/// exactly the non-empty levels, and `best_bid < best_ask` whenever both sides are non-empty.
#[derive(Clone, Debug)]
pub struct OrderBook {
    slots: Vec<Slot>,
    free: Vec<u32>,
    bids: Ladder,
    asks: Ladder,
    n_ticks: usize,
}

#[inline]
fn release_slot(slots: &mut [Slot], free: &mut Vec<u32>, idx: u32) {
    let s = &mut slots[idx as usize];
    s.state = SlotState::Free;
    s.generation = s.generation.wrapping_add(1);
    s.prev = NIL;
    s.next = NIL;
    s.qty = 0;
    free.push(idx);
}

#[inline]
fn unlink(slots: &mut [Slot], ladder: &mut Ladder, idx: u32) {
    let (prev, next, price, qty) = {
        let s = &slots[idx as usize];
        (s.prev, s.next, s.price, s.qty)
    };
    let level = &mut ladder.levels[price as usize];
    if prev != NIL {
        slots[prev as usize].next = next;
    } else {
        level.head = next;
    }
    if next != NIL {
        slots[next as usize].prev = prev;
    } else {
        level.tail = prev;
    }
    level.orders -= 1;
    level.qty -= qty;
    ladder.orders -= 1;
    ladder.qty -= qty;
    if level.orders == 0 {
        ladder.level_emptied(price);
    }
}

#[inline]
fn append(slots: &mut [Slot], ladder: &mut Ladder, idx: u32) {
    let (price, qty) = {
        let s = &slots[idx as usize];
        (s.price, s.qty)
    };
    let level = &mut ladder.levels[price as usize];
    let tail = level.tail;
    {
        let s = &mut slots[idx as usize];
        s.prev = tail;
        s.next = NIL;
        s.state = SlotState::Resting;
    }
    if tail != NIL {
        slots[tail as usize].next = idx;
    } else {
        level.head = idx;
    }
    level.tail = idx;
    let was_empty = level.orders == 0;
    level.orders += 1;
    level.qty += qty;
    ladder.orders += 1;
    ladder.qty += qty;
    if was_empty {
        ladder.mark(price as usize);
    }
    if ladder.best.is_none_or(|b| ladder.better(price, b)) {
        ladder.best = Some(price);
    }
}

impl OrderBook {
    pub fn new(n_ticks: usize) -> Self {
        assert!(n_ticks >= 2, "price ladder needs at least two ticks");
        Self {
            slots: Vec::with_capacity(4096),
            free: Vec::with_capacity(4096),
            bids: Ladder::new(Side::Buy, n_ticks),
            asks: Ladder::new(Side::Sell, n_ticks),
            n_ticks,
        }
    }

    pub fn n_ticks(&self) -> usize {
        self.n_ticks
    }

    #[inline]
    fn ladder(&self, side: Side) -> &Ladder {
        match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        }
    }

    #[inline]
    fn valid_price(&self, price: Price) -> bool {
        price >= 0 && (price as usize) < self.n_ticks
    }

    /// Reserves an order handle. The handle stays valid until the order is fully executed,
    /// cancelled, expired or released.
    pub fn allocate(&mut self, owner: AgentId) -> OrderId {
        let idx = match self.free.pop() {
            Some(i) => i,
            None => {
                let i = u32::try_from(self.slots.len()).expect("order slab exhausted");
                assert!(i != NIL, "order slab exhausted");
                self.slots.push(Slot {
                    generation: 0,
                    state: SlotState::Free,
                    owner,
                    side: Side::Buy,
                    price: 0,
                    qty: 0,
                    prev: NIL,
                    next: NIL,
                });
                i
            }
        };
        let s = &mut self.slots[idx as usize];
        s.state = SlotState::Pending;
        s.owner = owner;
        OrderId::new(idx, s.generation)
    }

    pub fn release(&mut self, id: OrderId) {
        if self.slot_in_state(id, SlotState::Pending).is_some() {
            release_slot(&mut self.slots, &mut self.free, id.slot());
        }
    }

    #[inline]
    fn slot_in_state(&self, id: OrderId, state: SlotState) -> Option<u32> {
        let idx = id.slot();
        let s = self.slots.get(idx as usize)?;
        (s.generation == id.generation() && s.state == state).then_some(idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn match_incoming(
        &mut self,
        taker_order: OrderId,
        taker: AgentId,
        side: Side,
        limit: Option<Price>,
        qty: Qty,
        ts: Timestamp,
        fills: &mut Vec<Fill>,
    ) -> Qty {
        let OrderBook {
            slots, free, bids, asks, ..
        } = self;
        let book = match side {
            Side::Buy => asks,
            Side::Sell => bids,
        };
        let mut remaining = qty;
        while remaining > 0 {
            let Some(best) = book.best else { break };
            if let Some(l) = limit {
                let crosses = match side {
                    Side::Buy => best <= l,
                    Side::Sell => best >= l,
                };
                if !crosses {
                    break;
                }
            }
            let p = best as usize;
            let mut cur = book.levels[p].head;
            let mut taken_level: Qty = 0;
            let mut removed: u32 = 0;
            while remaining > 0 && cur != NIL {
                let s = &mut slots[cur as usize];
                let take = s.qty.min(remaining);
                s.qty -= take;
                remaining -= take;
                taken_level += take;
                let done = s.qty == 0;
                fills.push(Fill {
                    ts,
                    price: best,
                    qty: take,
                    taker_side: side,
                    taker,
                    taker_order,
                    maker: s.owner,
                    maker_order: OrderId::new(cur, s.generation),
                    maker_done: done,
                });
                if !done {
                    break;
                }
                let next = s.next;
                release_slot(slots, free, cur);
                removed += 1;
                if next != NIL {
                    slots[next as usize].prev = NIL;
                }
                cur = next;
            }
            let level = &mut book.levels[p];
            level.qty -= taken_level;
            level.orders -= removed;
            book.qty -= taken_level;
            book.orders -= removed as u64;
            if level.orders == 0 {
                level.head = NIL;
                level.tail = NIL;
                book.level_emptied(best);
            } else {
                level.head = cur;
            }
        }
        remaining
    }

    fn rest(&mut self, idx: u32) {
        let side = self.slots[idx as usize].side;
        let ladder = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        append(&mut self.slots, ladder, idx);
    }

    /// Executes a limit order against the opposite side, then rests any remainder.
    /// `id` must be a pending handle returned by [`OrderBook::allocate`].
    #[allow(clippy::too_many_arguments)]
    pub fn limit(
        &mut self,
        id: OrderId,
        side: Side,
        price: Price,
        qty: Qty,
        ts: Timestamp,
        fills: &mut Vec<Fill>,
    ) -> Result<LimitOutcome, RejectReason> {
        let Some(idx) = self.slot_in_state(id, SlotState::Pending) else {
            return Err(RejectReason::UnknownOrder);
        };
        if qty == 0 {
            release_slot(&mut self.slots, &mut self.free, idx);
            return Err(RejectReason::ZeroQuantity);
        }
        if !self.valid_price(price) {
            release_slot(&mut self.slots, &mut self.free, idx);
            return Err(RejectReason::PriceOutOfRange);
        }
        let owner = self.slots[idx as usize].owner;
        let remaining = self.match_incoming(id, owner, side, Some(price), qty, ts, fills);
        if remaining == 0 {
            release_slot(&mut self.slots, &mut self.free, idx);
        } else {
            let s = &mut self.slots[idx as usize];
            s.side = side;
            s.price = price;
            s.qty = remaining;
            self.rest(idx);
        }
        Ok(LimitOutcome {
            filled: qty - remaining,
            resting: remaining,
        })
    }

    /// Immediate-or-cancel execution at any price; returns the filled quantity and
    /// releases the handle.
    pub fn market(
        &mut self,
        id: OrderId,
        side: Side,
        qty: Qty,
        ts: Timestamp,
        fills: &mut Vec<Fill>,
    ) -> Result<Qty, RejectReason> {
        let Some(idx) = self.slot_in_state(id, SlotState::Pending) else {
            return Err(RejectReason::UnknownOrder);
        };
        let owner = self.slots[idx as usize].owner;
        let remaining = if qty == 0 {
            0
        } else {
            self.match_incoming(id, owner, side, None, qty, ts, fills)
        };
        release_slot(&mut self.slots, &mut self.free, idx);
        if qty == 0 {
            return Err(RejectReason::ZeroQuantity);
        }
        Ok(qty - remaining)
    }

    pub fn cancel(&mut self, owner: AgentId, target: OrderId) -> Result<CancelOutcome, RejectReason> {
        let Some(idx) = self.slot_in_state(target, SlotState::Resting) else {
            return Err(RejectReason::UnknownOrder);
        };
        let s = self.slots[idx as usize];
        if s.owner != owner {
            return Err(RejectReason::NotOwner);
        }
        let ladder = match s.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        unlink(&mut self.slots, ladder, idx);
        release_slot(&mut self.slots, &mut self.free, idx);
        Ok(CancelOutcome {
            side: s.side,
            price: s.price,
            qty: s.qty,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn modify(
        &mut self,
        owner: AgentId,
        target: OrderId,
        price: Price,
        qty: Qty,
        ts: Timestamp,
        fills: &mut Vec<Fill>,
    ) -> Result<ModifyOutcome, RejectReason> {
        let Some(idx) = self.slot_in_state(target, SlotState::Resting) else {
            return Err(RejectReason::UnknownOrder);
        };
        let s = self.slots[idx as usize];
        if s.owner != owner {
            return Err(RejectReason::NotOwner);
        }
        if !self.valid_price(price) {
            return Err(RejectReason::PriceOutOfRange);
        }
        if qty == 0 {
            let c = self.cancel(owner, target)?;
            return Ok(ModifyOutcome {
                side: c.side,
                old_price: c.price,
                old_qty: c.qty,
                requeued: false,
                filled: 0,
                resting: 0,
            });
        }
        if price == s.price && qty <= s.qty {
            let delta = s.qty - qty;
            let ladder = match s.side {
                Side::Buy => &mut self.bids,
                Side::Sell => &mut self.asks,
            };
            ladder.levels[price as usize].qty -= delta;
            ladder.qty -= delta;
            self.slots[idx as usize].qty = qty;
            return Ok(ModifyOutcome {
                side: s.side,
                old_price: s.price,
                old_qty: s.qty,
                requeued: false,
                filled: 0,
                resting: qty,
            });
        }
        let ladder = match s.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        unlink(&mut self.slots, ladder, idx);
        self.slots[idx as usize].state = SlotState::Pending;
        let remaining = self.match_incoming(target, owner, s.side, Some(price), qty, ts, fills);
        if remaining == 0 {
            release_slot(&mut self.slots, &mut self.free, idx);
        } else {
            let slot = &mut self.slots[idx as usize];
            slot.price = price;
            slot.qty = remaining;
            self.rest(idx);
        }
        Ok(ModifyOutcome {
            side: s.side,
            old_price: s.price,
            old_qty: s.qty,
            requeued: true,
            filled: qty - remaining,
            resting: remaining,
        })
    }

    pub fn submit_limit(
        &mut self,
        owner: AgentId,
        side: Side,
        price: Price,
        qty: Qty,
        ts: Timestamp,
        fills: &mut Vec<Fill>,
    ) -> (OrderId, Result<LimitOutcome, RejectReason>) {
        let id = self.allocate(owner);
        (id, self.limit(id, side, price, qty, ts, fills))
    }

    pub fn submit_market(
        &mut self,
        owner: AgentId,
        side: Side,
        qty: Qty,
        ts: Timestamp,
        fills: &mut Vec<Fill>,
    ) -> (OrderId, Result<Qty, RejectReason>) {
        let id = self.allocate(owner);
        (id, self.market(id, side, qty, ts, fills))
    }

    #[inline]
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.best
    }

    #[inline]
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.best
    }

    #[inline]
    pub fn best(&self, side: Side) -> Option<Price> {
        self.ladder(side).best
    }

    #[inline]
    pub fn mid(&self) -> Option<f64> {
        match (self.bids.best, self.asks.best) {
            (Some(b), Some(a)) => Some(0.5 * (b + a) as f64),
            _ => None,
        }
    }

    #[inline]
    pub fn spread(&self) -> Option<Price> {
        match (self.bids.best, self.asks.best) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    #[inline]
    pub fn level_qty(&self, side: Side, price: Price) -> Qty {
        if !self.valid_price(price) {
            return 0;
        }
        self.ladder(side).levels[price as usize].qty
    }

    #[inline]
    pub fn best_qty(&self, side: Side) -> Qty {
        self.best(side).map_or(0, |p| self.level_qty(side, p))
    }

    pub fn side_qty(&self, side: Side) -> Qty {
        self.ladder(side).qty
    }

    pub fn side_orders(&self, side: Side) -> u64 {
        self.ladder(side).orders
    }

    /// Occupied levels of `side` from the touch outward as `(price, aggregate qty)`.
    pub fn levels(&self, side: Side) -> Levels<'_> {
        let ladder = self.ladder(side);
        Levels {
            ladder,
            next: ladder.best,
        }
    }

    pub fn depth(&self, side: Side, max_levels: usize, out: &mut Vec<(Price, Qty)>) {
        out.clear();
        out.extend(self.levels(side).take(max_levels));
    }

    /// Aggregate quantity within `ticks` ticks of the touch, inclusive of the touch.
    pub fn qty_within(&self, side: Side, ticks: Price) -> Qty {
        let Some(best) = self.best(side) else { return 0 };
        let mut total = 0;
        for (p, q) in self.levels(side) {
            if (p - best).abs() > ticks {
                break;
            }
            total += q;
        }
        total
    }

    pub fn order(&self, id: OrderId) -> Option<OrderInfo> {
        let idx = self.slot_in_state(id, SlotState::Resting)?;
        let s = &self.slots[idx as usize];
        Some(OrderInfo {
            id,
            owner: s.owner,
            side: s.side,
            price: s.price,
            qty: s.qty,
        })
    }

    pub fn is_resting(&self, id: OrderId) -> bool {
        self.slot_in_state(id, SlotState::Resting).is_some()
    }

    /// Resting orders at one level in time priority.
    pub fn queue(&self, side: Side, price: Price) -> Vec<OrderInfo> {
        let mut out = Vec::new();
        if !self.valid_price(price) {
            return out;
        }
        let mut cur = self.ladder(side).levels[price as usize].head;
        while cur != NIL {
            let s = &self.slots[cur as usize];
            out.push(OrderInfo {
                id: OrderId::new(cur, s.generation),
                owner: s.owner,
                side: s.side,
                price: s.price,
                qty: s.qty,
            });
            cur = s.next;
        }
        out
    }

    pub fn check_invariants(&self) -> Result<(), String> {
        for ladder in [&self.bids, &self.asks] {
            let mut total_qty = 0;
            let mut total_orders = 0;
            let mut best: Option<Price> = None;
            for (p, level) in ladder.levels.iter().enumerate() {
                let marked = ladder.bits[p >> 6] & (1u64 << (p & 63)) != 0;
                if marked != (level.orders > 0) {
                    return Err(format!("{:?} level {p}: bitset/occupancy mismatch", ladder.side));
                }
                let mut cur = level.head;
                let mut prev = NIL;
                let mut q = 0;
                let mut n = 0;
                while cur != NIL {
                    let s = &self.slots[cur as usize];
                    if s.state != SlotState::Resting || s.price != p as Price || s.side != ladder.side {
                        return Err(format!("{:?} level {p}: foreign slot {cur}", ladder.side));
                    }
                    if s.prev != prev {
                        return Err(format!("{:?} level {p}: broken back-link at {cur}", ladder.side));
                    }
                    if s.qty == 0 {
                        return Err(format!("{:?} level {p}: empty resting order", ladder.side));
                    }
                    q += s.qty;
                    n += 1;
                    prev = cur;
                    cur = s.next;
                }
                if level.tail != prev || q != level.qty || n != level.orders {
                    return Err(format!("{:?} level {p}: aggregate mismatch", ladder.side));
                }
                if n > 0 && best.is_none_or(|b| ladder.better(p as Price, b)) {
                    best = Some(p as Price);
                }
                total_qty += q;
                total_orders += n as u64;
            }
            if best != ladder.best {
                return Err(format!("{:?}: best {:?} expected {:?}", ladder.side, ladder.best, best));
            }
            if total_qty != ladder.qty || total_orders != ladder.orders {
                return Err(format!("{:?}: side aggregate mismatch", ladder.side));
            }
        }
        if let (Some(b), Some(a)) = (self.bids.best, self.asks.best)
            && b >= a
        {
            return Err(format!("crossed book: bid {b} ask {a}"));
        }
        Ok(())
    }
}

pub struct Levels<'a> {
    ladder: &'a Ladder,
    next: Option<Price>,
}

impl Iterator for Levels<'_> {
    type Item = (Price, Qty);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let p = self.next?;
        self.next = self.ladder.worse_than(p);
        Some((p, self.ladder.levels[p as usize].qty))
    }
}
