import { s as store_get, u as unsubscribe_stores } from "../../chunks/index2.js";
import { V as ssr_context, W as escape_html } from "../../chunks/context.js";
import "clsx";
import { d as disconnect, c as connected, b as bedStatus, a as connectionError } from "../../chunks/bed.js";
function onDestroy(fn) {
  /** @type {SSRContext} */
  ssr_context.r.on_destroy(fn);
}
function _layout($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let { children } = $$props;
    onDestroy(() => {
      disconnect();
    });
    $$renderer2.push(`<div class="app svelte-12qhfyh"><header class="svelte-12qhfyh"><h1 class="svelte-12qhfyh">BED Auditor</h1> <nav class="svelte-12qhfyh"><a href="/" class="svelte-12qhfyh">Stream</a> <a href="/probes" class="svelte-12qhfyh">Probes</a> <a href="/xdb" class="svelte-12qhfyh">XDB</a> <a href="/audit" class="svelte-12qhfyh">Audit</a> <a href="/tools" class="svelte-12qhfyh">Tools</a></nav> <div class="status svelte-12qhfyh">`);
    if (store_get($$store_subs ??= {}, "$connected", connected)) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="connected svelte-12qhfyh">Connected</span> `);
      if (store_get($$store_subs ??= {}, "$bedStatus", bedStatus)) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="be-id svelte-12qhfyh">${escape_html(store_get($$store_subs ??= {}, "$bedStatus", bedStatus).be_id)}</span> <span class="tier svelte-12qhfyh">T${escape_html(store_get($$store_subs ??= {}, "$bedStatus", bedStatus).tier ?? "?")}</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<span class="disconnected svelte-12qhfyh">Disconnected</span> `);
      if (store_get($$store_subs ??= {}, "$connectionError", connectionError)) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="error svelte-12qhfyh">${escape_html(store_get($$store_subs ??= {}, "$connectionError", connectionError))}</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div></header> <main class="svelte-12qhfyh">`);
    children($$renderer2);
    $$renderer2.push(`<!----></main></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _layout as default
};
