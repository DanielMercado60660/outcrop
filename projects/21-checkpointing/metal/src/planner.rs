//! Exact recursive-split schedules without an nt-by-capacity cost table.
//! Full-storage and zero-slot intervals emit their known plans directly.
use outcrop_common::Result;

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Advance(usize), Save(usize, usize), Restore(usize, Option<usize>),
    Drop(usize, usize), Reverse(usize),
}

pub struct Plan { pub events: Vec<Event>, pub forward_steps: i64 }

/// Binomial coefficient, saturated at limit. Callers have n < 2^64 and
/// limit <= 2^63, so intermediate multiplies fit u128. Saturation keeps
/// enormous binomial capacities cheap and exactly comparable.
fn binomial(n: u128, k: u128, limit: usize) -> usize {
    if k > n { return 0; }
    let k = k.min(n - k);
    let mut value = 1u128;
    for j in 1..=k {
        value = value * (n - k + j) / j;
        if value >= limit as u128 { return limit; }
    }
    value as usize
}

/// Smallest minimizing split for the API's recurrence.
///
/// Write G(N,c)=F(N,c,0). Its increasing marginal costs r=1,2,... have
/// multiplicities C(c+r,c). The last marginal level is the smallest r
/// with N <= C(c+r+1,c+1), and G(N,c)=r*N-C(c+r+1,c+2).
/// Also F(n,c,e)=G(n+e,c), including e=1's otherwise-unneeded final step.
///
/// For a split, put x=k-1 and y=n+e-k-1. The cost is
/// 1+x+G(x+1,c)+G(y+1,c-1), with x+y=n+e-2. Merge the sorted marginal
/// costs of the two terms; Pascal's identity gives the multiplicities
/// above. To minimize k, take all suffix marginals <=r before tied prefix
/// marginals. The prefix must still contain all its marginals <r. Hence
/// k=max(C(c+r-1,c+1), n+e-C(c+r,c)); at r=1 the first bound is 1.
/// This handles both end conventions without a split search.
fn optimal_split(n: usize, c: usize, end_needed: bool) -> usize {
    let length = n + usize::from(end_needed);
    let (mut low, mut high) = (1, length - 1);
    while low < high {
        let mid = low + (high - low) / 2;
        if binomial(c as u128 + mid as u128 + 1, c as u128 + 1, length) >= length {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let r = low as u128;
    let c = c as u128;
    let prefix = if r == 1 { 1 } else { binomial(c + r - 1, c + 1, length) };
    let suffix = binomial(c + r, c, length);
    prefix.max(length.saturating_sub(suffix))
}

#[derive(Clone, Copy)]
struct Interval {
    start: usize, n: usize, c: usize, end_needed: bool, initial: Option<usize>,
}

enum Task {
    Descend(Interval),
    // The suffix is finished: drop its initial restart, restore the parent,
    // then descend the prefix. A heap stack avoids O(nt) call depth for
    // budgets just below full storage.
    Prefix(Interval, usize),
}

impl Plan {
    pub fn new(nt: usize, capacity: usize) -> Result<Self> {
        if nt == 0 || nt > i64::MAX as usize || capacity >= nt {
            return Err("invalid planner interval/capacity".into());
        }
        let mut events = Vec::new();
        let mut free: Vec<usize> = (0..capacity).rev().collect();
        let mut tasks = vec![Task::Descend(Interval {
            start: 0, n: nt, c: capacity, end_needed: true, initial: None,
        })];
        while let Some(task) = tasks.pop() {
            let interval = match task {
                Task::Descend(interval) => interval,
                Task::Prefix(interval, slot) => {
                    events.push(Event::Drop(interval.start + interval.n, slot));
                    free.push(slot);
                    events.push(Event::Restore(interval.start, interval.initial));
                    interval
                }
            };
            let Interval { start, n, c, end_needed, initial } = interval;
            if n == 1 {
                if end_needed {
                    events.push(Event::Advance(start + 1));
                    events.push(Event::Restore(start, initial));
                }
                events.push(Event::Reverse(start));
            } else if c == 0 {
                if end_needed {
                    events.push(Event::Advance(start + n));
                    events.push(Event::Restore(start, initial));
                }
                for i in (0..n).rev() {
                    if i > 0 { events.push(Event::Advance(start + i)); }
                    events.push(Event::Reverse(start + i));
                    if i > 0 { events.push(Event::Restore(start, initial)); }
                }
            } else if c >= n - 1 {
                let mut saved = Vec::with_capacity(n - 1);
                for i in 1..n {
                    events.push(Event::Advance(start + i));
                    let slot = free.pop().ok_or("planner exhausted free slots")?;
                    events.push(Event::Save(start + i, slot));
                    saved.push((start + i, slot));
                }
                if end_needed { events.push(Event::Advance(start + n)); }
                for &(t, slot) in saved.iter().rev() {
                    events.push(Event::Restore(t, Some(slot)));
                    events.push(Event::Reverse(t));
                    events.push(Event::Drop(t, slot));
                    free.push(slot);
                }
                events.push(Event::Restore(start, initial));
                events.push(Event::Reverse(start));
            } else {
                let k = optimal_split(n, c, end_needed);
                if k == 0 || k >= n { return Err("invalid planner split".into()); }
                events.push(Event::Advance(start + k));
                let slot = free.pop().ok_or("planner exhausted free slots")?;
                events.push(Event::Save(start + k, slot));
                tasks.push(Task::Prefix(Interval {
                    start, n: k, c, end_needed: false, initial,
                }, slot));
                tasks.push(Task::Descend(Interval {
                    start: start + k, n: n - k, c: c - 1, end_needed, initial: Some(slot),
                }));
            }
        }
        let (mut current, mut forward_steps) = (0, 0i64);
        for &event in &events {
            match event {
                Event::Advance(t) => {
                    let steps = t.checked_sub(current).ok_or("invalid planner advance")?;
                    forward_steps = forward_steps.checked_add(i64::try_from(steps)?)
                        .ok_or("forward step count exceeds int64")?;
                    current = t;
                }
                Event::Restore(t, _) => current = t,
                _ => {}
            }
        }
        Ok(Self { events, forward_steps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent, literal locked recurrence; only used on small intervals.
    fn reference(nt: usize) -> Vec<Vec<[i64; 2]>> {
        let mut cost = vec![vec![[0; 2]; nt]; nt + 1];
        for n in 1..=nt {
            for c in 0..nt {
                for e in 0..=1 {
                    cost[n][c][e] = if n == 1 { e as i64 }
                        else if c == 0 { (n * (n - 1) / 2 + e * n) as i64 }
                        else if c >= n - 1 { (n - 1 + e) as i64 }
                        else { (1..n).map(|k| k as i64 + cost[n-k][c-1][e] + cost[k][c][0]).min().unwrap() };
                }
            }
        }
        cost
    }

    fn audit(plan: &Plan, nt: usize, capacity: usize) {
        let (mut current, mut next, mut work, mut reached_end) = (0, nt, 0i64, false);
        let mut slots = vec![None; capacity];
        for &event in &plan.events {
            match event {
                Event::Advance(t) => {
                    assert!(current < t && t <= nt);
                    work += (t - current) as i64;
                    current = t;
                    reached_end |= t == nt;
                }
                Event::Save(t, slot) => {
                    assert!(t > 0 && t < nt);
                    assert_eq!(t, current);
                    assert!(slots[slot].replace(t).is_none());
                }
                Event::Restore(t, slot) => {
                    assert!(reached_end);
                    assert_eq!(Some(t), slot.map_or(Some(0), |s| slots[s]));
                    current = t;
                }
                Event::Drop(t, slot) => assert_eq!(slots[slot].take(), Some(t)),
                Event::Reverse(t) => {
                    assert!(reached_end && next > 0);
                    assert_eq!(t, next - 1);
                    assert_eq!(t, current);
                    next -= 1;
                }
            }
        }
        assert_eq!(next, 0);
        assert!(slots.iter().all(Option::is_none));
        assert_eq!(work, plan.forward_steps);
    }

    #[test]
    fn binomial_splits_match_exact_recurrence_and_smallest_ties() {
        let cost = reference(48);
        for n in 1..=48 {
            for c in 0..n {
                let plan = Plan::new(n, c).unwrap();
                audit(&plan, n, c);
                assert_eq!(plan.forward_steps, cost[n][c][1], "n={n} c={c}");
                if c > 0 && c < n - 1 {
                    for e in 0..=1 {
                        let k = (1..n).find(|&k| k as i64 + cost[n-k][c-1][e] + cost[k][c][0] == cost[n][c][e]).unwrap();
                        assert_eq!(optimal_split(n, c, e == 1), k, "n={n} c={c} e={e}");
                    }
                }
            }
        }
    }

    #[test]
    fn long_plans_need_neither_quadratic_tables_nor_recursive_call_depth() {
        for capacity in [0, 1, 7, 5_000, 9_998, 9_999] {
            let plan = Plan::new(10_000, capacity).unwrap();
            audit(&plan, 10_000, capacity);
            assert!(plan.events.len() <= 6 * 10_000);
            if capacity == 0 { assert_eq!(plan.forward_steps, 50_005_000); }
            if capacity == 9_999 { assert_eq!(plan.forward_steps, 10_000); }
        }
    }
}
