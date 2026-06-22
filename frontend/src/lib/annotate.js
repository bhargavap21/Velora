// obs = [inventory_remaining, time_remaining, current_price, vwap_so_far,
//         spread, volatility, volume_imbalance, participation_rate_lag]
//
// Returns { label, text, color, activeDims } where activeDims is the set of
// obs indices that triggered the rule — used by the chart to highlight them.
export function annotate(obs) {
  // current_price (idx 2) and vwap_so_far (idx 3) not used by annotation rules
  const [inv, time, /* current_price */, /* vwap_so_far */, spread, , , lag] = obs

  if (inv > 0.7 && time < 0.3)
    return { label: 'URGENCY',  text: 'urgency mode: agent forced aggressive execution', color: '#f97316', activeDims: [0, 1] }
  if (spread > 0.6)
    return { label: 'CAUTION',  text: 'wide spread — agent backing off to reduce cost',  color: '#fbbf24', activeDims: [4] }
  if (lag > 0.8)
    return { label: 'MOMENTUM', text: 'momentum — agent riding prior volume',            color: '#34d399', activeDims: [7] }

  return { label: 'BALANCED', text: 'balanced execution', color: '#6ee7b7', activeDims: [] }
}
