import { s as store_get, a as attr_class, b as attr, e as ensure_array_like, c as stringify, d as attr_style, u as unsubscribe_stores } from "../../chunks/index2.js";
import { t as ticks, l as latestTick, r as recentViolations, e as recentSteering } from "../../chunks/bed.js";
import { W as escape_html } from "../../chunks/context.js";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let viewMode = "stream";
    let autoScroll = true;
    let conceptSummary = (() => {
      const data = store_get($$store_subs ??= {}, "$ticks", ticks);
      const summaries = {};
      for (const item of data.slice(-100)) {
        const activations = item.concept_activations;
        for (const [concept, score] of Object.entries(activations)) {
          if (!summaries[concept]) {
            summaries[concept] = { total: 0, max: 0, count: 0, recent: [] };
          }
          summaries[concept].total += score;
          summaries[concept].max = Math.max(summaries[concept].max, score);
          summaries[concept].count += 1;
          summaries[concept].recent.push(score);
          if (summaries[concept].recent.length > 10) {
            summaries[concept].recent.shift();
          }
        }
      }
      return Object.entries(summaries).map(([concept_id, s]) => {
        const avg = s.total / s.count;
        const recent = s.recent;
        let trend = "stable";
        if (recent.length >= 3) {
          const firstHalf = recent.slice(0, Math.floor(recent.length / 2));
          const secondHalf = recent.slice(Math.floor(recent.length / 2));
          const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
          const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
          if (secondAvg > firstAvg * 1.1) trend = "rising";
          else if (secondAvg < firstAvg * 0.9) trend = "falling";
        }
        return {
          concept_id,
          total_activations: s.total,
          avg_score: avg,
          max_score: s.max,
          tick_count: s.count,
          trend
        };
      }).sort((a, b) => b.avg_score - a.avg_score).slice(0, 15);
    })();
    function formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        fractionalSecondDigits: 3
      });
    }
    function getTickTypeClass(type) {
      const classes = {
        input: "type-input",
        output: "type-output",
        introspect: "type-introspect",
        steering: "type-steering",
        workspace: "type-workspace",
        system: "type-system"
      };
      return classes[type] || "";
    }
    function topActivations(tick) {
      const entries = Object.entries(tick.concept_activations);
      return entries.sort((a, b) => b[1] - a[1]).slice(0, 3);
    }
    function getTrendIcon(trend) {
      if (trend === "rising") return "↑";
      if (trend === "falling") return "↓";
      return "→";
    }
    $$renderer2.push(`<div class="stream-page svelte-1uha8ag"><div class="controls svelte-1uha8ag"><div class="view-toggle svelte-1uha8ag"><button${attr_class("svelte-1uha8ag", void 0, { "active": viewMode === "stream" })}>Experience Stream</button> <button${attr_class("svelte-1uha8ag", void 0, { "active": viewMode === "audit" })}>CAT Audit Log</button></div> <div class="control-buttons svelte-1uha8ag">`);
    {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<label class="svelte-1uha8ag"><input type="checkbox"${attr("checked", autoScroll, true)}/> Auto-scroll</label> <button class="svelte-1uha8ag">Clear</button> <span class="tick-count svelte-1uha8ag">${escape_html(store_get($$store_subs ??= {}, "$ticks", ticks).length)} ticks</span>`);
    }
    $$renderer2.push(`<!--]--></div></div> <div class="panels svelte-1uha8ag"><div class="stream-panel svelte-1uha8ag">`);
    {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(store_get($$store_subs ??= {}, "$ticks", ticks));
      for (let $$index_2 = 0, $$length = each_array.length; $$index_2 < $$length; $$index_2++) {
        let tick = each_array[$$index_2];
        $$renderer2.push(`<div${attr_class(`tick ${stringify(getTickTypeClass(tick.tick_type))}`, "svelte-1uha8ag")}><div class="tick-header svelte-1uha8ag"><span class="tick-id svelte-1uha8ag">#${escape_html(tick.tick_id)}</span> <span class="tick-time svelte-1uha8ag">${escape_html(formatTime(tick.timestamp))}</span> <span class="tick-type svelte-1uha8ag">${escape_html(tick.tick_type)}</span> `);
        if (tick.tier !== null) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="tick-tier svelte-1uha8ag">T${escape_html(tick.tier)}</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div> <div class="tick-content svelte-1uha8ag">`);
        if (tick.content) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="content-text svelte-1uha8ag">${escape_html(tick.content)}</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div> `);
        if (Object.keys(tick.concept_activations).length > 0) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="tick-activations svelte-1uha8ag"><!--[-->`);
          const each_array_1 = ensure_array_like(topActivations(tick));
          for (let $$index = 0, $$length2 = each_array_1.length; $$index < $$length2; $$index++) {
            let [concept, score] = each_array_1[$$index];
            $$renderer2.push(`<span class="activation svelte-1uha8ag"${attr_style(`opacity: ${stringify(0.3 + score * 0.7)}`)}>${escape_html(concept)}: ${escape_html(score.toFixed(2))}</span>`);
          }
          $$renderer2.push(`<!--]--></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--> `);
        if (tick.hush_violations.length > 0) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="tick-violations svelte-1uha8ag"><!--[-->`);
          const each_array_2 = ensure_array_like(tick.hush_violations);
          for (let $$index_1 = 0, $$length2 = each_array_2.length; $$index_1 < $$length2; $$index_1++) {
            let v = each_array_2[$$index_1];
            $$renderer2.push(`<span class="violation svelte-1uha8ag">! ${escape_html(v.simplex_term)}</span>`);
          }
          $$renderer2.push(`<!--]--></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div>`);
      }
      $$renderer2.push(`<!--]--> `);
      if (store_get($$store_subs ??= {}, "$ticks", ticks).length === 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="empty-state svelte-1uha8ag"><p>Waiting for experience stream...</p> <p class="hint svelte-1uha8ag">Generate text to see ticks appear here.</p></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div> <div class="sidebar svelte-1uha8ag"><div class="panel concept-summary svelte-1uha8ag"><h3 class="svelte-1uha8ag">Concept Summary</h3> `);
    if (conceptSummary.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<p class="empty svelte-1uha8ag">No concept data yet</p>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="concept-list svelte-1uha8ag"><!--[-->`);
      const each_array_6 = ensure_array_like(conceptSummary);
      for (let $$index_6 = 0, $$length = each_array_6.length; $$index_6 < $$length; $$index_6++) {
        let cs = each_array_6[$$index_6];
        $$renderer2.push(`<div class="concept-row svelte-1uha8ag"><span class="concept-name svelte-1uha8ag">${escape_html(cs.concept_id)}</span> <span class="concept-stats svelte-1uha8ag"><span class="avg svelte-1uha8ag">${escape_html(cs.avg_score.toFixed(2))}</span> <span${attr_class("trend svelte-1uha8ag", void 0, {
          "rising": cs.trend === "rising",
          "falling": cs.trend === "falling"
        })}>${escape_html(getTrendIcon(cs.trend))}</span> <span class="count svelte-1uha8ag">(${escape_html(cs.tick_count)})</span></span></div>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--></div> `);
    {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="panel latest-tick svelte-1uha8ag"><h3 class="svelte-1uha8ag">Latest Tick</h3> `);
      if (store_get($$store_subs ??= {}, "$latestTick", latestTick)) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="detail-grid svelte-1uha8ag"><span class="label svelte-1uha8ag">ID:</span> <span>#${escape_html(store_get($$store_subs ??= {}, "$latestTick", latestTick).tick_id)}</span> <span class="label svelte-1uha8ag">Type:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$latestTick", latestTick).tick_type)}</span> <span class="label svelte-1uha8ag">State:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$latestTick", latestTick).workspace_state || "-")}</span> <span class="label svelte-1uha8ag">Tier:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$latestTick", latestTick).tier ?? "-")}</span> <span class="label svelte-1uha8ag">H-Norm:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$latestTick", latestTick).hidden_state_norm?.toFixed(2) || "-")}</span></div> `);
        if (Object.keys(store_get($$store_subs ??= {}, "$latestTick", latestTick).simplex_activations).length > 0) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<h4 class="svelte-1uha8ag">Simplex</h4> <div class="simplex-list svelte-1uha8ag"><!--[-->`);
          const each_array_7 = ensure_array_like(Object.entries(store_get($$store_subs ??= {}, "$latestTick", latestTick).simplex_activations));
          for (let $$index_7 = 0, $$length = each_array_7.length; $$index_7 < $$length; $$index_7++) {
            let [term, value] = each_array_7[$$index_7];
            const dev = store_get($$store_subs ??= {}, "$latestTick", latestTick).simplex_deviations[term];
            $$renderer2.push(`<div class="simplex-item svelte-1uha8ag"><span class="term svelte-1uha8ag">${escape_html(term)}</span> <span class="value svelte-1uha8ag">${escape_html(value.toFixed(3))}</span> `);
            if (dev !== null && dev !== void 0) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<span${attr_class("deviation svelte-1uha8ag", void 0, { "warning": Math.abs(dev) > 1.5 })}>${escape_html(dev > 0 ? "+" : "")}${escape_html(dev.toFixed(2))}σ</span>`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]--></div>`);
          }
          $$renderer2.push(`<!--]--></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<p class="empty svelte-1uha8ag">No ticks yet</p>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="panel alerts svelte-1uha8ag"><h3 class="svelte-1uha8ag">Alerts `);
      if (store_get($$store_subs ??= {}, "$recentViolations", recentViolations).length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="badge svelte-1uha8ag">${escape_html(store_get($$store_subs ??= {}, "$recentViolations", recentViolations).length)}</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></h3> `);
      if (store_get($$store_subs ??= {}, "$recentViolations", recentViolations).length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="alert-list svelte-1uha8ag"><!--[-->`);
        const each_array_8 = ensure_array_like(store_get($$store_subs ??= {}, "$recentViolations", recentViolations).slice(-5));
        for (let $$index_8 = 0, $$length = each_array_8.length; $$index_8 < $$length; $$index_8++) {
          let v = each_array_8[$$index_8];
          $$renderer2.push(`<div class="alert violation svelte-1uha8ag"><span class="alert-type svelte-1uha8ag">VIOLATION</span> <span class="alert-term svelte-1uha8ag">${escape_html(v.simplex_term)}</span> <span class="alert-value svelte-1uha8ag">${escape_html(v.deviation?.toFixed(2))}σ</span></div>`);
        }
        $$renderer2.push(`<!--]--></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (store_get($$store_subs ??= {}, "$recentSteering", recentSteering).length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="alert-list svelte-1uha8ag"><!--[-->`);
        const each_array_9 = ensure_array_like(store_get($$store_subs ??= {}, "$recentSteering", recentSteering).slice(-5));
        for (let $$index_9 = 0, $$length = each_array_9.length; $$index_9 < $$length; $$index_9++) {
          let s = each_array_9[$$index_9];
          $$renderer2.push(`<div class="alert steering svelte-1uha8ag"><span class="alert-type svelte-1uha8ag">STEERING</span> <span class="alert-term svelte-1uha8ag">${escape_html(s.simplex_term)}</span> <span class="alert-value svelte-1uha8ag">×${escape_html(s.strength.toFixed(2))}</span></div>`);
        }
        $$renderer2.push(`<!--]--></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (store_get($$store_subs ??= {}, "$recentViolations", recentViolations).length === 0 && store_get($$store_subs ??= {}, "$recentSteering", recentSteering).length === 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<p class="empty svelte-1uha8ag">No alerts</p>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (store_get($$store_subs ??= {}, "$recentViolations", recentViolations).length > 0 || store_get($$store_subs ??= {}, "$recentSteering", recentSteering).length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button class="clear-btn svelte-1uha8ag">Clear</button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--></div></div></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
