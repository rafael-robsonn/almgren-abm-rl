use crate::book::OrderBook;
use crate::types::{OrderId, Price, Qty, Side, Timestamp};

/// Message-level (L3) book event. The stream is complete: replaying `Add`, `Execute` and
/// `Cancel` in order reproduces every resting order, so any L2 view can be rebuilt from it.
/// A requeueing modify is logged as a full `Cancel`, the executions it triggers, then `Add`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L3Event {
    Add {
        ts: Timestamp,
        order: OrderId,
        side: Side,
        price: Price,
        qty: Qty,
    },
    Execute {
        ts: Timestamp,
        maker: OrderId,
        taker: OrderId,
        taker_side: Side,
        price: Price,
        qty: Qty,
    },
    Cancel {
        ts: Timestamp,
        order: OrderId,
        qty: Qty,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct L2Snapshot {
    pub ts: Timestamp,
    pub bids: Vec<(Price, Qty)>,
    pub asks: Vec<(Price, Qty)>,
}

impl L2Snapshot {
    pub fn capture(book: &OrderBook, depth: usize, ts: Timestamp) -> Self {
        Self {
            ts,
            bids: book.levels(Side::Buy).take(depth).collect(),
            asks: book.levels(Side::Sell).take(depth).collect(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HistoryConfig {
    pub l3: bool,
    pub l2_interval: Option<Timestamp>,
    pub l2_depth: usize,
}

#[derive(Clone, Debug, Default)]
pub struct History {
    pub config: HistoryConfig,
    pub l3: Vec<L3Event>,
    pub l2: Vec<L2Snapshot>,
    next_l2: Timestamp,
}

impl History {
    pub fn new(config: HistoryConfig) -> Self {
        Self {
            next_l2: config.l2_interval.unwrap_or(Timestamp::MAX),
            config,
            l3: Vec::new(),
            l2: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn record_l3(&mut self, event: L3Event) {
        if self.config.l3 {
            self.l3.push(event);
        }
    }

    #[inline]
    pub(crate) fn sample_l2(&mut self, book: &OrderBook, now: Timestamp) {
        let Some(interval) = self.config.l2_interval else { return };
        while self.next_l2 <= now {
            self.l2
                .push(L2Snapshot::capture(book, self.config.l2_depth, self.next_l2));
            self.next_l2 += interval;
        }
    }

    pub fn clear(&mut self) {
        self.l3.clear();
        self.l2.clear();
    }
}
