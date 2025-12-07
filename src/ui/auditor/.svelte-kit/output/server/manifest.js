export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.CrwxLHKX.js",app:"_app/immutable/entry/app.CGvLQ9ur.js",imports:["_app/immutable/entry/start.CrwxLHKX.js","_app/immutable/chunks/iecM6VJV.js","_app/immutable/chunks/DN0K50FJ.js","_app/immutable/chunks/v0FPBf_t.js","_app/immutable/entry/app.CGvLQ9ur.js","_app/immutable/chunks/DN0K50FJ.js","_app/immutable/chunks/BGMlQYQP.js","_app/immutable/chunks/v0FPBf_t.js","_app/immutable/chunks/BMmI4BWs.js","_app/immutable/chunks/CaPF1CxJ.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js')),
			__memo(() => import('./nodes/3.js')),
			__memo(() => import('./nodes/4.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			},
			{
				id: "/probes",
				pattern: /^\/probes\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 3 },
				endpoint: null
			},
			{
				id: "/xdb",
				pattern: /^\/xdb\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 4 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
