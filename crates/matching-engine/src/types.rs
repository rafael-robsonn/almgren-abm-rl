pub type Price = i64;
pub type Qty = u64;
pub type Timestamp = u64;
pub type AgentId = u32;

pub const NANOS_PER_SEC: Timestamp = 1_000_000_000;

#[inline]
pub fn secs(s: f64) -> Timestamp {
    (s * NANOS_PER_SEC as f64).round().max(0.0) as Timestamp
}

#[inline]
pub fn to_secs(t: Timestamp) -> f64 {
    t as f64 / NANOS_PER_SEC as f64
}

/// Order handle. The low 32 bits index a slab slot and the high 32 bits carry the slot
/// generation, so a handle is never resolved after its order has left the book.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrderId(pub u64);

impl OrderId {
    #[inline]
    pub(crate) fn new(slot: u32, generation: u32) -> Self {
        Self(((generation as u64) << 32) | slot as u64)
    }

    #[inline]
    pub(crate) fn slot(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub(crate) fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    #[inline]
    pub fn sign(self) -> i64 {
        match self {
            Side::Buy => 1,
            Side::Sell => -1,
        }
    }
}

/// Inbound order-entry message. `Modify` sets the new resting quantity: a pure quantity
/// reduction at an unchanged price keeps time priority, any other change re-queues the order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Limit { side: Side, price: Price, qty: Qty },
    Market { side: Side, qty: Qty },
    Cancel { target: OrderId },
    Modify { target: OrderId, price: Price, qty: Qty },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fill {
    pub ts: Timestamp,
    pub price: Price,
    pub qty: Qty,
    pub taker_side: Side,
    pub taker: AgentId,
    pub taker_order: OrderId,
    pub maker: AgentId,
    pub maker_order: OrderId,
    pub maker_done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    PriceOutOfRange,
    ZeroQuantity,
    UnknownOrder,
    NotOwner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Liquidity {
    Maker,
    Taker,
}

/// Execution report addressed to the owner of `order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Report {
    Fill {
        order: OrderId,
        side: Side,
        price: Price,
        qty: Qty,
        liquidity: Liquidity,
        done: bool,
    },
    Rested {
        order: OrderId,
        side: Side,
        price: Price,
        qty: Qty,
    },
    Cancelled {
        order: OrderId,
        qty: Qty,
    },
    Modified {
        order: OrderId,
        price: Price,
        qty: Qty,
    },
    Expired {
        order: OrderId,
        qty: Qty,
    },
    Rejected {
        order: OrderId,
        reason: RejectReason,
    },
}

impl Report {
    pub fn order(&self) -> OrderId {
        match *self {
            Report::Fill { order, .. }
            | Report::Rested { order, .. }
            | Report::Cancelled { order, .. }
            | Report::Modified { order, .. }
            | Report::Expired { order, .. }
            | Report::Rejected { order, .. } => order,
        }
    }
}
