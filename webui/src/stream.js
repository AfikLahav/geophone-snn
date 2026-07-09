// WebSocket client for Geophone mode — connects to the backend and dispatches
// per-window detection messages. Same-origin, so it works on whatever host/port
// the FastAPI server is on.

export function createStream(onMessage) {
  let ws = null;

  function start(cfg) {
    stop();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    // capture the socket locally; a fast stop()/switch can null `ws` before this
    // socket finishes connecting, so guard every callback against a stale socket.
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    ws = sock;
    sock.onopen = () => { if (ws === sock) sock.send(JSON.stringify(cfg)); };
    sock.onmessage = (e) => {
      if (ws !== sock) return;
      try { onMessage(JSON.parse(e.data)); } catch (_) { /* ignore */ }
    };
    sock.onclose = () => { if (ws === sock) ws = null; };
    sock.onerror = () => { if (ws === sock) onMessage({ type: 'error' }); };
  }

  function stop() {
    if (ws) { try { ws.close(); } catch (_) {} ws = null; }
  }

  return { start, stop, isOpen: () => !!ws };
}
