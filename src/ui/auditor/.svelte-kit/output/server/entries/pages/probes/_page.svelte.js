import { s as store_get, e as ensure_array_like, a as attr_class, b as attr, u as unsubscribe_stores } from "../../../chunks/index2.js";
import { t as ticks, i as introspection } from "../../../chunks/bed.js";
import { W as escape_html } from "../../../chunks/context.js";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let conceptTraces = (() => {
      const traces = {};
      const recentTicks = store_get($$store_subs ??= {}, "$ticks", ticks).slice(-100);
      for (const tick of recentTicks) {
        for (const [concept, score] of Object.entries(tick.concept_activations)) {
          if (!traces[concept]) traces[concept] = [];
          traces[concept].push({ tick: tick.tick_id, score });
        }
      }
      return Object.entries(traces).sort((a, b) => b[1].length - a[1].length).slice(0, 20);
    })();
    let simplexTraces = (() => {
      const traces = {};
      const recentTicks = store_get($$store_subs ??= {}, "$ticks", ticks).slice(-100);
      for (const tick of recentTicks) {
        for (const [term, score] of Object.entries(tick.simplex_activations)) {
          if (!traces[term]) traces[term] = [];
          traces[term].push({
            tick: tick.tick_id,
            score,
            deviation: tick.simplex_deviations[term] ?? null
          });
        }
      }
      return Object.entries(traces);
    })();
    function sparkline(data, width = 100, height = 24) {
      if (data.length === 0) return "";
      const min = Math.min(...data.map((d) => d.score));
      const max = Math.max(...data.map((d) => d.score));
      const range = max - min || 1;
      const points = data.map((d, i) => {
        const x = i / (data.length - 1 || 1) * width;
        const y = height - (d.score - min) / range * height;
        return `${x},${y}`;
      }).join(" ");
      return `M ${points.replace(/ /g, " L ")}`;
    }
    function stats(data) {
      if (data.length === 0) return { min: 0, max: 0, avg: 0 };
      const scores = data.map((d) => d.score);
      return {
        min: Math.min(...scores),
        max: Math.max(...scores),
        avg: scores.reduce((a, b) => a + b, 0) / scores.length
      };
    }
    $$renderer2.push(`<div class="probes-page svelte-1yyefww"><div class="header svelte-1yyefww"><h2 class="svelte-1yyefww">Probe Activations</h2> <button class="svelte-1yyefww">Refresh</button></div> <div class="panels svelte-1yyefww"><div class="panel simplex-panel svelte-1yyefww"><h3 class="svelte-1yyefww">Simplex Probes (Proprioception)</h3> `);
    if (simplexTraces.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<p class="empty svelte-1yyefww">No simplex data yet</p>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="probe-list svelte-1yyefww"><!--[-->`);
      const each_array = ensure_array_like(simplexTraces);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let [term, data] = each_array[$$index];
        const s = stats(data);
        const lastDev = data[data.length - 1]?.deviation;
        $$renderer2.push(`<div class="probe-row svelte-1yyefww"><div class="probe-info svelte-1yyefww"><span class="probe-name svelte-1yyefww">${escape_html(term)}</span> <span class="probe-stats svelte-1yyefww">${escape_html(s.avg.toFixed(3))} (±${escape_html((s.max - s.min).toFixed(3))})</span> `);
        if (lastDev !== null && lastDev !== void 0) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span${attr_class("probe-deviation svelte-1yyefww", void 0, {
            "warning": Math.abs(lastDev) > 1.5,
            "danger": Math.abs(lastDev) > 2.5
          })}>${escape_html(lastDev > 0 ? "+" : "")}${escape_html(lastDev.toFixed(2))}σ</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div> <svg class="sparkline svelte-1yyefww" width="120" height="28" viewBox="0 0 100 24"><path${attr("d", sparkline(data))} fill="none" stroke="#58a6ff" stroke-width="1.5"></path></svg></div>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--></div> <div class="panel concept-panel svelte-1yyefww"><h3 class="svelte-1yyefww">Concept Activations</h3> `);
    if (conceptTraces.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<p class="empty svelte-1yyefww">No concept data yet</p>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="probe-list svelte-1yyefww"><!--[-->`);
      const each_array_1 = ensure_array_like(conceptTraces);
      for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
        let [concept, data] = each_array_1[$$index_1];
        const s = stats(data);
        $$renderer2.push(`<div class="probe-row svelte-1yyefww"><div class="probe-info svelte-1yyefww"><span class="probe-name svelte-1yyefww">${escape_html(concept)}</span> <span class="probe-stats svelte-1yyefww">avg: ${escape_html(s.avg.toFixed(3))} | ${escape_html(data.length)} hits</span></div> <svg class="sparkline svelte-1yyefww" width="120" height="28" viewBox="0 0 100 24"><path${attr("d", sparkline(data))} fill="none" stroke="#3fb950" stroke-width="1.5"></path></svg></div>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--></div></div> `);
    if (store_get($$store_subs ??= {}, "$introspection", introspection)) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="introspection-panel svelte-1yyefww"><h3 class="svelte-1yyefww">Introspection Report</h3> <div class="report-grid svelte-1yyefww"><span class="label svelte-1yyefww">BE ID:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).be_id)}</span> <span class="label svelte-1yyefww">XDB ID:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).xdb_id)}</span> <span class="label svelte-1yyefww">Current Tick:</span> <span>#${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).current_tick)}</span> <span class="label svelte-1yyefww">Workspace:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).workspace_state || "-")}</span> <span class="label svelte-1yyefww">Tier:</span> <span>T${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).tier ?? "?")}</span></div> `);
      if (store_get($$store_subs ??= {}, "$introspection", introspection).recent_violations.length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="violations-section"><h4 class="svelte-1yyefww">Recent Violations (${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).recent_violations.length)})</h4> <div class="violation-list svelte-1yyefww"><!--[-->`);
        const each_array_2 = ensure_array_like(store_get($$store_subs ??= {}, "$introspection", introspection).recent_violations.slice(-5));
        for (let $$index_2 = 0, $$length = each_array_2.length; $$index_2 < $$length; $$index_2++) {
          let v = each_array_2[$$index_2];
          $$renderer2.push(`<div class="violation-item svelte-1yyefww"><span class="v-term svelte-1yyefww">${escape_html(v.simplex_term)}</span> <span class="v-value svelte-1yyefww">${escape_html(v.deviation?.toFixed(2))}σ</span> <span class="v-threshold svelte-1yyefww">threshold: ${escape_html(v.threshold)}</span></div>`);
        }
        $$renderer2.push(`<!--]--></div></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (store_get($$store_subs ??= {}, "$introspection", introspection).recent_steering.length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="steering-section"><h4 class="svelte-1yyefww">Recent Steering (${escape_html(store_get($$store_subs ??= {}, "$introspection", introspection).recent_steering.length)})</h4> <div class="steering-list svelte-1yyefww"><!--[-->`);
        const each_array_3 = ensure_array_like(store_get($$store_subs ??= {}, "$introspection", introspection).recent_steering.slice(-5));
        for (let $$index_3 = 0, $$length = each_array_3.length; $$index_3 < $$length; $$index_3++) {
          let s = each_array_3[$$index_3];
          $$renderer2.push(`<div class="steering-item svelte-1yyefww"><span class="s-term svelte-1yyefww">${escape_html(s.simplex_term)}</span> <span class="s-strength svelte-1yyefww">×${escape_html(s.strength.toFixed(2))}</span> <span class="s-target svelte-1yyefww">${escape_html(s.target_pole)}</span></div>`);
        }
        $$renderer2.push(`<!--]--></div></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
