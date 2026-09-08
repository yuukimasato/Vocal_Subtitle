// node_modules/abslink/src/types.js
var WireValueType = {
  RAW: "RAW",
  PROXY: "PROXY",
  THROW: "THROW",
  HANDLER: "HANDLER"
};
var MessageType = {
  GET: "GET",
  SET: "SET",
  APPLY: "APPLY",
  CONSTRUCT: "CONSTRUCT",
  RELEASE: "RELEASE"
};

// node_modules/abslink/src/abslink.js
var proxyMarker = /* @__PURE__ */ Symbol("Abslink.proxy");
var releaseProxy = /* @__PURE__ */ Symbol("Abslink.releaseProxy");
var finalizer = /* @__PURE__ */ Symbol("Abslink.finalizer");
var throwMarker = /* @__PURE__ */ Symbol("Abslink.thrown");
var isObject = (val) => typeof val === "object" && val !== null || typeof val === "function";
var proxyTransferHandler = {
  canHandle: (val) => isObject(val) && proxyMarker in val,
  serialize(obj, ep) {
    const markerID = obj[proxyMarker];
    expose(obj, ep, markerID);
    return [markerID, []];
  },
  deserialize(markerID, ep) {
    return wrap(ep, markerID);
  }
};
var throwTransferHandler = {
  canHandle: (value) => isObject(value) && throwMarker in value,
  serialize({ value }) {
    let serialized;
    if (value instanceof Error) {
      serialized = {
        isError: true,
        value: {
          message: value.message,
          name: value.name,
          stack: value.stack
        }
      };
    } else {
      serialized = { isError: false, value };
    }
    return [serialized, []];
  },
  deserialize(serialized) {
    if (serialized.isError) {
      throw Object.assign(new Error(serialized.value.message), serialized.value);
    }
    throw serialized.value;
  }
};
var transferHandlers = /* @__PURE__ */ new Map([
  ["proxy", proxyTransferHandler],
  ["throw", throwTransferHandler]
]);
function filterPath(path, obj) {
  let parent = obj;
  const parentPath = path.slice(0, -1);
  for (const segment of parentPath) {
    if (Object.prototype.hasOwnProperty.call(parent, segment)) {
      parent = parent[segment];
    }
  }
  const lastSegment = path[path.length - 1];
  const RawValue = lastSegment ? parent[lastSegment] : parent;
  return { parent, RawValue, lastSegment };
}
function expose(obj, ep, rootMarkerID) {
  ep.on("message", function callback(data) {
    if (!data)
      return;
    const { id, type, path, markerID } = {
      path: [],
      ...data
    };
    if (markerID !== rootMarkerID)
      return;
    const argumentList = (data.argumentList ?? []).map((v) => fromWireValue(v, ep));
    let returnValue;
    try {
      const { parent, RawValue, lastSegment } = filterPath(path, obj);
      switch (type) {
        case MessageType.GET:
          returnValue = RawValue;
          break;
        case MessageType.SET:
          parent[lastSegment] = fromWireValue(data.value, ep);
          returnValue = true;
          break;
        case MessageType.APPLY:
          returnValue = RawValue.apply(parent, argumentList);
          break;
        case MessageType.CONSTRUCT:
          returnValue = new RawValue(...argumentList);
          break;
        case MessageType.RELEASE:
          returnValue = void 0;
          break;
        default:
          return;
      }
    } catch (value) {
      returnValue = { value, [throwMarker]: 0 };
    }
    Promise.resolve(returnValue).catch((value) => {
      return { value, [throwMarker]: 0 };
    }).then((returnValue2) => {
      if (type === MessageType.CONSTRUCT)
        returnValue2 = proxy(returnValue2);
      const [wireValue, transfer2] = toWireValue(returnValue2, ep);
      ep.postMessage({ ...wireValue, id, markerID: rootMarkerID }, transfer2);
      if (type === MessageType.RELEASE) {
        ep.off("message", callback);
        obj[finalizer]?.();
        ep.close?.();
      }
    }).catch((_) => {
      const [wireValue, transfer2] = toWireValue({
        value: new TypeError("Unserializable return value"),
        [throwMarker]: 0
      }, ep);
      ep.postMessage({ ...wireValue, id, markerID: rootMarkerID }, transfer2);
    });
  });
  return obj;
}
function wrap(endpoint, rootMarkerID) {
  const pendingListeners = /* @__PURE__ */ new Map();
  endpoint.on("message", (data) => {
    if (!data?.id) {
      return;
    }
    const resolver = pendingListeners.get(data.id);
    if (!resolver) {
      return;
    }
    try {
      resolver(data);
    } finally {
      pendingListeners.delete(data.id);
    }
  });
  return createProxy({ endpoint, pendingListeners, rootMarkerID });
}
function throwIfProxyReleased(isReleased) {
  if (isReleased) {
    throw new Error("Proxy has been released and is not useable");
  }
}
async function releaseEndpoint(epWithPendingListeners) {
  await requestResponseMessage(epWithPendingListeners, { type: MessageType.RELEASE });
  epWithPendingListeners.endpoint.close?.();
}
var proxyCounter = /* @__PURE__ */ new WeakMap();
var proxyFinalizers = "FinalizationRegistry" in globalThis && new FinalizationRegistry((epWithPendingListeners) => {
  const newCount = (proxyCounter.get(epWithPendingListeners) ?? 0) - 1;
  proxyCounter.set(epWithPendingListeners, newCount);
  if (newCount === 0) {
    releaseEndpoint(epWithPendingListeners).finally(() => {
      epWithPendingListeners.pendingListeners.clear();
    });
  }
});
function registerProxy(proxy2, epWithPendingListeners) {
  const newCount = (proxyCounter.get(epWithPendingListeners) ?? 0) + 1;
  proxyCounter.set(epWithPendingListeners, newCount);
  if (proxyFinalizers) {
    proxyFinalizers.register(proxy2, epWithPendingListeners, proxy2);
  }
}
function unregisterProxy(proxy2) {
  if (proxyFinalizers) {
    proxyFinalizers.unregister(proxy2);
  }
}
function createProxy(epWithPendingListeners, path = []) {
  let isProxyReleased = false;
  const propProxyCache = /* @__PURE__ */ new Map();
  const proxy2 = new Proxy(function() {
  }, {
    get(_target, prop) {
      throwIfProxyReleased(isProxyReleased);
      if (prop === releaseProxy) {
        return async () => {
          for (const subProxy of propProxyCache.values()) {
            subProxy[releaseProxy]();
          }
          propProxyCache.clear();
          unregisterProxy(proxy2);
          releaseEndpoint(epWithPendingListeners).finally(() => {
            epWithPendingListeners.pendingListeners.clear();
          });
          isProxyReleased = true;
        };
      }
      if (prop === "then") {
        if (path.length === 0) {
          return { then: () => proxy2 };
        }
        const r = requestResponseMessage(epWithPendingListeners, {
          type: MessageType.GET,
          path: path.map((p) => p.toString())
        }).then((v) => fromWireValue(v, epWithPendingListeners.endpoint));
        return r.then.bind(r);
      }
      const cachedProxy = propProxyCache.get(prop);
      if (cachedProxy) {
        return cachedProxy;
      }
      const propProxy = createProxy(epWithPendingListeners, [...path, prop]);
      propProxyCache.set(prop, propProxy);
      return propProxy;
    },
    set(_target, prop, rawValue) {
      throwIfProxyReleased(isProxyReleased);
      const [value, transfer2] = toWireValue(rawValue, epWithPendingListeners.endpoint);
      return requestResponseMessage(epWithPendingListeners, {
        type: MessageType.SET,
        path: [...path, prop].map((p) => p.toString()),
        value
      }, transfer2).then((v) => fromWireValue(v, epWithPendingListeners.endpoint));
    },
    apply(_target, _thisArg, rawArgumentList) {
      throwIfProxyReleased(isProxyReleased);
      const last = path[path.length - 1];
      if (last === "bind") {
        return createProxy(epWithPendingListeners, path.slice(0, -1));
      }
      const [argumentList, transfer2] = processArguments(rawArgumentList, epWithPendingListeners);
      return requestResponseMessage(epWithPendingListeners, {
        type: MessageType.APPLY,
        path: path.map((p) => p.toString()),
        argumentList
      }, transfer2).then((v) => fromWireValue(v, epWithPendingListeners.endpoint));
    },
    construct(_target, rawArgumentList) {
      throwIfProxyReleased(isProxyReleased);
      const [argumentList, transfer2] = processArguments(rawArgumentList, epWithPendingListeners);
      return requestResponseMessage(epWithPendingListeners, {
        type: MessageType.CONSTRUCT,
        path: path.map((p) => p.toString()),
        argumentList
      }, transfer2).then((v) => fromWireValue(v, epWithPendingListeners.endpoint));
    }
  });
  registerProxy(proxy2, epWithPendingListeners);
  return proxy2;
}
var transferCache = /* @__PURE__ */ new WeakMap();
function transfer(obj, transfers) {
  transferCache.set(obj, transfers);
  return obj;
}
function processArguments(argumentList, epWithPendingListeners) {
  const wireValues = [];
  const transferables = [];
  for (const argument of argumentList) {
    const [wireValue, transfer2] = toWireValue(argument, epWithPendingListeners.endpoint);
    wireValues.push(wireValue);
    transferables.push(...transfer2);
  }
  return [wireValues, transferables];
}
function proxy(obj) {
  return Object.assign(obj, { [proxyMarker]: randomId() });
}
function toWireValue(value, ep) {
  for (const [name, handler] of transferHandlers) {
    if (handler.canHandle(value)) {
      const [serializedValue, transfer2] = handler.serialize(value, ep);
      return [{
        type: WireValueType.HANDLER,
        name,
        value: serializedValue
      }, transfer2];
    }
  }
  return [{
    type: WireValueType.RAW,
    value
  }, transferCache.get(value) ?? []];
}
function fromWireValue(value, ep) {
  switch (value.type) {
    case WireValueType.HANDLER:
      return transferHandlers.get(value.name).deserialize(value.value, ep);
    case WireValueType.RAW:
      return value.value;
  }
}
function requestResponseMessage(ep, msg, transfer2) {
  return new Promise((resolve) => {
    const id = randomId();
    ep.pendingListeners.set(id, resolve);
    ep.endpoint.postMessage({ id, ...msg, markerID: ep.rootMarkerID }, transfer2);
  });
}
var hex = [];
var alphabet = "0123456789abcdef";
for (let i = 0; i < 256; i++) {
  hex[i] = alphabet[i >> 4 & 15] + alphabet[i & 15];
}
var step = 0;
var buffer = "";
function randomId() {
  let i = 0;
  if (!buffer || step + 16 > 256 * 2) {
    for (buffer = "", step = 0; i < 256; ++i) {
      buffer += hex[Math.random() * 256 | 0];
    }
  }
  return buffer.substring(step, ++step + 16);
}
export {
  MessageType,
  WireValueType,
  expose,
  finalizer,
  isObject,
  proxy,
  proxyMarker,
  releaseProxy,
  throwMarker,
  transfer,
  transferHandlers,
  wrap
};
