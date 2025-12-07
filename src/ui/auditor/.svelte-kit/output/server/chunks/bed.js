import { d as derived, w as writable } from "./index.js";
const connected = writable(false);
const connectionError = writable(null);
const ticks = writable([]);
const bedStatus = writable(null);
const introspection = writable(null);
const recentViolations = writable([]);
const recentSteering = writable([]);
const latestTick = derived(
  ticks,
  ($ticks) => $ticks.length > 0 ? $ticks[$ticks.length - 1] : null
);
derived(
  ticks,
  ($ticks) => $ticks.filter((t) => t.tick_type === "output")
);
derived(
  recentViolations,
  ($violations) => $violations.length
);
function disconnect() {
  connected.set(false);
}
export {
  connectionError as a,
  bedStatus as b,
  connected as c,
  disconnect as d,
  recentSteering as e,
  introspection as i,
  latestTick as l,
  recentViolations as r,
  ticks as t
};
